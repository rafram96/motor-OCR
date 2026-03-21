from __future__ import annotations
import logging
import time
from typing import List

from models.document_result import DocumentResult
from models.page_result import PageResult
from segmentation.detector import es_candidata_separadora, evaluar_separadora
from segmentation.models.separator_page import SeparatorPage
from segmentation.models.professional_section import ProfessionalSection

logger = logging.getLogger(__name__)


def _format_eta(segundos: float) -> str:
    if segundos <= 0:
        return "0s"
    s = int(segundos)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def segment_document(doc: DocumentResult) -> List[ProfessionalSection]:
    """
    Divide un DocumentResult en secciones por profesional.

    Flujo:
    1. Filtrar páginas candidatas a separadora (pocas líneas)
    2. Confirmar cada candidata con Qwen (+ fallback fuzzy)
    3. Agrupar páginas entre separadoras consecutivas
    4. Retornar lista de ProfessionalSection ordenada por aparición

    Args:
        doc: DocumentResult del motor OCR con todas las páginas.

    Returns:
        Lista de ProfessionalSection. Puede estar vacía si no se detectaron
        separadoras (expediente sin la estructura esperada).
    """
    pages_ord = sorted(doc.pages, key=lambda p: p.page_number)

    # ── 1. Identificar separadoras ────────────────────────────────────────────
    candidatas: List[PageResult] = [
        p for p in pages_ord if es_candidata_separadora(p)
    ]

    logger.info(
        f"Segmentación: {len(pages_ord)} páginas totales, "
        f"{len(candidatas)} candidatas a separadora"
    )

    separadoras: List[SeparatorPage] = []
    total_candidatas = len(candidatas)
    progreso_cada = max(1, total_candidatas // 10) if total_candidatas else 1
    t_candidatas = time.time()

    for idx, page in enumerate(candidatas, start=1):
        sep = evaluar_separadora(page)
        if sep.es_separadora:
            separadoras.append(sep)

        if idx == 1 or idx % progreso_cada == 0 or idx == total_candidatas:
            elapsed = time.time() - t_candidatas
            promedio = elapsed / idx
            restante = max(0.0, promedio * (total_candidatas - idx))
            pct = (idx / total_candidatas) * 100 if total_candidatas else 100.0
            logger.info(
                f"Segmentación progreso (confirmación): {idx}/{total_candidatas} "
                f"({pct:.1f}%), ETA {_format_eta(restante)}"
            )

    logger.info(f"Separadoras confirmadas: {len(separadoras)}")

    if not separadoras:
        logger.warning(
            "No se detectaron separadoras — el documento puede no tener "
            "la estructura esperada o todas las candidatas fueron descartadas."
        )
        return []

    # ── 2. Agrupar páginas entre separadoras ──────────────────────────────────
    secciones: List[ProfessionalSection] = []

    for i, sep in enumerate(separadoras):
        inicio = sep.page_number
        fin = (
            separadoras[i + 1].page_number - 1
            if i + 1 < len(separadoras)
            else pages_ord[-1].page_number
        )

        paginas_seccion = [
            p for p in pages_ord
            if inicio <= p.page_number <= fin
        ]

        seccion = ProfessionalSection(
            section_index=i + 1,
            cargo=sep.cargo_normalizado,
            cargo_raw=sep.cargo_detectado,
            separator_page=inicio,
            pages=paginas_seccion,
            total_pages=len(paginas_seccion),
            has_tables=any(p.tiene_tabla for p in paginas_seccion),
        )
        secciones.append(seccion)

        logger.debug(
            f"  Sección {i+1}: '{sep.cargo_normalizado}' "
            f"págs {inicio}–{fin} ({len(paginas_seccion)} páginas) "
            f"[{sep.metodo}]"
        )

    logger.info(
        f"Segmentación completada: {len(secciones)} profesionales detectados"
    )
    return secciones