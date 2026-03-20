import os
import time
import logging
import numpy as np
from typing import Optional

# FLAGS_use_mkldnn debe estar antes de cualquier import de paddle
os.environ["FLAGS_use_mkldnn"] = "0"

from paddleocr import PaddleOCR

from config import (
    PADDLE_LANG,
    PADDLE_USE_TEXTLINE_ORIENT,
    PADDLE_USE_DOC_ORIENT,
    PADDLE_USE_DOC_UNWARPING,
    PADDLE_REC_SCORE_THRESH,
    PADDLE_DET_THRESH,
    PADDLE_DET_BOX_THRESH,
    UMBRAL_CONFIANZA_LINEA,
)
from models.page_result import PageResult

logger = logging.getLogger(__name__)

_ocr_instance: Optional[PaddleOCR] = None


def get_ocr() -> PaddleOCR:
    """
    Retorna la instancia singleton de PaddleOCR.
    Los modelos se cargan una sola vez por proceso (~500MB).
    """
    global _ocr_instance
    if _ocr_instance is None:
        logger.info("Inicializando PaddleOCR (primera vez, carga de modelos)...")
        _ocr_instance = PaddleOCR(
            use_textline_orientation=PADDLE_USE_TEXTLINE_ORIENT,
            use_doc_orientation_classify=PADDLE_USE_DOC_ORIENT,
            use_doc_unwarping=PADDLE_USE_DOC_UNWARPING,
            lang=PADDLE_LANG,
            text_rec_score_thresh=PADDLE_REC_SCORE_THRESH,
            text_det_thresh=PADDLE_DET_THRESH,
            text_det_box_thresh=PADDLE_DET_BOX_THRESH,
        )
        logger.info("PaddleOCR inicializado.")
    return _ocr_instance


def predict(image_path: str, page_number: int) -> PageResult:
    """
    Corre OCR sobre una imagen y retorna un PageResult con texto y métricas.

    Args:
        image_path:  Ruta absoluta a la imagen PNG.
        page_number: Número de página en el documento (base 1).

    Returns:
        PageResult con engine_used='paddle' y todas las métricas calculadas.
        Si paddle falla internamente, retorna un PageResult de error.
    """
    t_start = time.time()

    try:
        ocr = get_ocr()
        resultado = ocr.predict(image_path)
    except Exception as e:
        elapsed = time.time() - t_start
        logger.error(f"Página {page_number}: paddle falló — {e}")
        return PageResult.error_placeholder(
            page_number=page_number,
            image_path=image_path,
            reason=f"paddle_exception: {e}",
        )

    t_paddle = time.time() - t_start

    if not resultado:
        logger.warning(f"Página {page_number}: paddle devolvió resultado vacío")
        return PageResult.error_placeholder(
            page_number=page_number,
            image_path=image_path,
            reason="paddle_empty_result",
        )

    res = resultado[0]

    # ── Extraer campos del resultado raw ──────────────────────────────────────
    textos: list  = res.get("rec_texts", [])
    scores: list  = res.get("rec_scores", [])
    dt_polys: list = res.get("dt_polys", [])
    angle: int    = res.get("doc_preprocessor_res", {}).get("angle", 0)

    # Filtrar líneas vacías
    lineas_limpias = [t for t in textos if isinstance(t, str) and t.strip()]
    texto_completo = "\n".join(lineas_limpias)

    # ── Métricas de confianza ─────────────────────────────────────────────────
    det_count = len(dt_polys)
    rec_count = len(textos)
    tasa_descarte = (det_count - rec_count) / det_count if det_count > 0 else 0.0

    if scores:
        arr = np.array(scores, dtype=float)
        conf_promedio  = float(np.mean(arr))
        conf_mediana   = float(np.median(arr))
        conf_min       = float(np.min(arr))
        conf_max       = float(np.max(arr))
        conf_std       = float(np.std(arr))
        lineas_bajas   = int(np.sum(arr < UMBRAL_CONFIANZA_LINEA))
    else:
        # Página en blanco o completamente ilegible
        conf_promedio = conf_mediana = conf_min = conf_max = conf_std = None
        lineas_bajas  = 0

    elapsed = time.time() - t_start

    conf_str = f"{conf_promedio:.3f}" if conf_promedio is not None else "N/A"

    logger.debug(
        f"Página {page_number}: "
        f"conf={conf_str} "
        f"descarte={tasa_descarte*100:.1f}% "
        f"angle={angle}° "
        f"t={elapsed:.2f}s"
    )

    return PageResult(
        page_number=page_number,
        image_path=image_path,
        engine_used="paddle",
        fallback_reason=None,
        text=texto_completo,
        lines=lineas_limpias,
        conf_promedio=conf_promedio,
        conf_mediana=conf_mediana,
        conf_min=conf_min,
        conf_max=conf_max,
        conf_std=conf_std,
        lineas_baja_confianza=lineas_bajas,
        det_count=det_count,
        rec_count=rec_count,
        tasa_descarte=tasa_descarte,
        angle_detected=angle,
        tiene_tabla=False,          # lo detecta qwen_engine si se necesita
        tiempo_paddle=t_paddle,
        tiempo_qwen=None,
        tiempo_total=elapsed,
    )