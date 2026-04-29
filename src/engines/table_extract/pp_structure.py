"""
Wrapper de PaddleOCR PP-Structure V3 para extraer tablas como matrices.

Carga PPStructureV3 una vez por proceso (modelos ~600MB en GPU).
Procesa imagenes una por una y devuelve, por imagen, una lista de tablas
detectadas con su matriz [filas][columnas] y el HTML literal de PP-Structure.

API:
    extract_tables_from_image(image_path, page_num) -> list[dict]

Cada dict tiene:
    - "pagina": int
    - "matriz": list[list[str]]   # filas x cols
    - "html": str                  # HTML literal de PP-Structure
    - "bbox": [x1, y1, x2, y2] o None
    - "n_filas": int
    - "n_cols": int
    - "score": float               # confianza si PP-Structure la expone
"""
from __future__ import annotations
import logging
import os
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

# FLAGS_use_mkldnn debe estar antes de cualquier import de paddle
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

logger = logging.getLogger(__name__)

_engine = None


def get_engine():
    """Singleton de PPStructureV3. Se carga 1 sola vez por proceso."""
    global _engine
    if _engine is not None:
        return _engine

    try:
        from paddleocr import PPStructureV3
    except ImportError as e:
        logger.error(
            "PPStructureV3 no disponible en este venv: %s. "
            "Verificar paddleocr>=3.0", e,
        )
        raise

    logger.info("Inicializando PPStructureV3 (primera vez, carga de modelos)...")

    # PPStructureV3 acepta varios sub-pipelines. Para tablas solo necesitamos
    # ocr + table_recognition. Apagamos lo demas para reducir VRAM/latencia.
    init_kwargs_v3 = dict(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        use_seal_recognition=False,
        use_chart_recognition=False,
        use_formula_recognition=False,
        use_table_recognition=True,
    )

    try:
        _engine = PPStructureV3(**init_kwargs_v3)
    except TypeError:
        # Algunas versiones no aceptan algunos kwargs — fallback minimo
        logger.warning(
            "PPStructureV3 con kwargs avanzados no soportado, "
            "reintentando con init minimo"
        )
        _engine = PPStructureV3()

    logger.info("PPStructureV3 inicializado.")
    return _engine


# ── HTML -> matriz ───────────────────────────────────────────────────────────

class _TableHtmlToMatrix(HTMLParser):
    """
    Parser HTML que convierte el <table>...</table> de PP-Structure en matriz
    [filas][columnas] de strings, expandiendo correctamente colspan y rowspan
    para que cada celda merged tenga su valor replicado en cada fila/col que
    cubre (estandar para que el consumidor pueda mapear columnas por header).
    """

    def __init__(self):
        super().__init__()
        self._matrix: list[list[str]] = []
        self._current_row: list[str] = []
        self._cell_buffer: list[str] = []
        self._in_cell = False
        self._cell_attrs: dict = {}
        # rowspan pendiente: por cada columna, cuantas filas mas ocupa y con
        # que valor. Aplicado al inicio de cada nueva fila.
        self._rowspan_pending: dict[int, tuple[int, str]] = {}

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._current_row = []
            # aplicar rowspans pendientes al principio de la fila
            self._aplicar_rowspans()
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell_buffer = []
            self._cell_attrs = {k: v for k, v in attrs}

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._in_cell = False
            text = "".join(self._cell_buffer).strip()
            colspan = int(self._cell_attrs.get("colspan", 1) or 1)
            rowspan = int(self._cell_attrs.get("rowspan", 1) or 1)

            # Replicar por colspan en la fila actual
            for _ in range(max(1, colspan)):
                self._current_row.append(text)

            # Registrar rowspan pendiente para las proximas filas
            if rowspan > 1:
                col_actual = len(self._current_row) - colspan
                for offset in range(colspan):
                    col = col_actual + offset
                    self._rowspan_pending[col] = (rowspan - 1, text)

        elif tag == "tr":
            if self._current_row:
                self._matrix.append(self._current_row)
            self._current_row = []

    def handle_data(self, data):
        if self._in_cell:
            self._cell_buffer.append(data)

    def _aplicar_rowspans(self):
        """
        Inserta los valores de rowspan pendientes en su columna correspondiente
        antes de procesar las celdas literales de esta fila.
        """
        if not self._rowspan_pending:
            return
        # Construir fila base con None y luego rellenar
        max_col = max(self._rowspan_pending.keys()) + 1
        # Inicializar fila como lista de strings vacios y al final filtrarla
        # (esto es complejo porque las celdas literales se agregan despues).
        # Aproximacion: avanzar con padding hasta la columna del rowspan.
        for col in sorted(self._rowspan_pending.keys()):
            while len(self._current_row) < col:
                self._current_row.append("")
            count, text = self._rowspan_pending[col]
            # Insertar el texto del rowspan en la columna correcta
            if len(self._current_row) <= col:
                self._current_row.append(text)
            else:
                self._current_row.insert(col, text)
            # decrementar contador
            if count - 1 <= 0:
                del self._rowspan_pending[col]
            else:
                self._rowspan_pending[col] = (count - 1, text)

    @property
    def matrix(self) -> list[list[str]]:
        # Normalizar n_cols: rellenar filas cortas con ""
        if not self._matrix:
            return []
        max_cols = max(len(r) for r in self._matrix)
        return [
            r + [""] * (max_cols - len(r)) if len(r) < max_cols else r
            for r in self._matrix
        ]


def html_to_matrix(html: str) -> list[list[str]]:
    """Convierte un <table>...</table> a matriz [filas][cols]."""
    if not html or not isinstance(html, str):
        return []
    parser = _TableHtmlToMatrix()
    try:
        parser.feed(html)
    except Exception as e:
        logger.warning("HTML parse fallo: %s", e)
        return []
    return parser.matrix


# ── Extraccion de tablas por imagen ──────────────────────────────────────────

def _normalizar_resultado(res, page_num: int) -> list[dict]:
    """
    Convierte el output de PPStructureV3.predict() en una lista uniforme de
    dicts con {pagina, matriz, html, bbox, n_filas, n_cols, score}.

    PPStructureV3 puede devolver el resultado en distintas formas segun la
    version. Probamos varias rutas para ser robustos.
    """
    # Intentar obtener un dict serializable
    res_dict = None
    if hasattr(res, "json"):
        try:
            j = res.json
            if callable(j):
                j = j()
            if isinstance(j, dict):
                res_dict = j
        except Exception:
            res_dict = None

    if res_dict is None and hasattr(res, "__dict__"):
        res_dict = {k: v for k, v in res.__dict__.items() if not k.startswith("_")}

    if res_dict is None and isinstance(res, dict):
        res_dict = res

    if res_dict is None:
        return []

    # Buscar lista de tablas en multiples claves conocidas
    tablas_raw = []
    for key in (
        "table_res_list",
        "table_recognition_res_list",
        "tables",
        "parsing_res_list",
        "layout_parsing_res_list",
    ):
        if key in res_dict and res_dict[key]:
            tablas_raw = res_dict[key]
            break

    # Si no hay clave directa, intentar dentro de res_dict["res"] o similar
    if not tablas_raw:
        for key in ("res", "result", "results"):
            sub = res_dict.get(key)
            if isinstance(sub, dict):
                for k2 in ("table_res_list", "tables"):
                    if k2 in sub and sub[k2]:
                        tablas_raw = sub[k2]
                        break

    resultados: list[dict] = []

    for region in tablas_raw or []:
        # region puede ser dict o un objeto con atributos
        if not isinstance(region, dict):
            region = getattr(region, "__dict__", {}) or {}

        # Filtrar solo tablas si la region tiene tipo
        tipo = (region.get("type") or region.get("category") or "").lower()
        if tipo and tipo not in ("table", "tabla"):
            # Si parsing_res_list, las regiones que no son tablas las saltamos
            continue

        # Buscar HTML
        html = ""
        for key in ("html", "pred_html", "structure", "table_html"):
            v = region.get(key)
            if isinstance(v, str) and "<" in v:
                html = v
                break
        # html anidado en res / pred
        if not html:
            sub = region.get("res") or region.get("pred") or {}
            if isinstance(sub, dict):
                for key in ("html", "pred_html"):
                    v = sub.get(key)
                    if isinstance(v, str) and "<" in v:
                        html = v
                        break

        bbox = region.get("bbox") or region.get("box") or region.get("region_bbox")
        score = region.get("score") or region.get("confidence")

        if not html:
            logger.debug(
                "PP-Structure region en pag %d sin HTML — saltando", page_num,
            )
            continue

        matriz = html_to_matrix(html)
        if not matriz or len(matriz) < 2:
            logger.debug(
                "PP-Structure region en pag %d con matriz vacia/trivial — saltando",
                page_num,
            )
            continue

        resultados.append({
            "pagina": page_num,
            "matriz": matriz,
            "html": html[:8000],  # cap de seguridad
            "bbox": list(bbox) if bbox else None,
            "n_filas": len(matriz),
            "n_cols": max(len(r) for r in matriz),
            "score": float(score) if score is not None else None,
        })

    return resultados


def extract_tables_from_image(
    image_path: str,
    page_num: int,
) -> list[dict]:
    """
    Detecta y extrae tablas de una imagen de pagina.

    Returns:
        list de dicts (ver _normalizar_resultado).

    Si PP-Structure falla, devuelve [] y loguea el error.
    """
    if not os.path.exists(image_path):
        logger.error("Imagen no encontrada: %s", image_path)
        return []

    try:
        engine = get_engine()
    except Exception as e:
        logger.exception("PPStructureV3 init fallo: %s", e)
        return []

    try:
        # PPStructureV3 v3.x acepta `input=` o positional
        try:
            output = engine.predict(input=image_path)
        except TypeError:
            output = engine.predict(image_path)
    except Exception as e:
        logger.exception("PPStructureV3.predict fallo en pag %d: %s", page_num, e)
        return []

    resultados: list[dict] = []
    try:
        # output es iterable de StructureChatResult / dict
        for res in output:
            resultados.extend(_normalizar_resultado(res, page_num))
    except Exception as e:
        logger.exception(
            "PP-Structure: error procesando output de pag %d: %s",
            page_num, e,
        )
        return []

    logger.info(
        "PP-Structure pag %d: %d tablas extraidas",
        page_num, len(resultados),
    )
    return resultados
