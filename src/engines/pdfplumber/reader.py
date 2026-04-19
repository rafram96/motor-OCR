from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import List, Optional

import pdfplumber

from models.page_result import PageResult

logger = logging.getLogger(__name__)


def read_pdf_pages(
    pdf_path: str,
    pages: Optional[List[int]] = None,
) -> List[PageResult]:
    """
    Lee un PDF digital con pdfplumber y retorna una lista de PageResult.

    Cada PageResult tiene:
    - engine_used = "pdfplumber"
    - conf_promedio = 1.0 (pdfplumber no hace OCR, el texto viene del PDF)
    - image_path = "" (no se renderizan imagenes)
    - lines = lineas no vacias del texto extraido

    Args:
        pdf_path: Ruta al PDF (debe ser digital con capa de texto).
        pages: Numeros de pagina 1-based a extraer. None = todas.

    Returns:
        Lista de PageResult ordenada por page_number.
    """
    t_inicio = time.time()
    results: List[PageResult] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        total = len(pdf.pages)
        logger.info(f"pdfplumber abrio {Path(pdf_path).name}: {total} paginas")

        for idx, pdf_page in enumerate(pdf.pages, start=1):
            if pages is not None and idx not in pages:
                continue

            t_pagina = time.time()
            try:
                text = pdf_page.extract_text() or ""
            except Exception as e:
                logger.warning(f"Pagina {idx}: extract_text fallo - {e}")
                results.append(
                    PageResult.error_placeholder(idx, "", f"pdfplumber_exception: {e}")
                )
                continue

            lines = [l for l in text.splitlines() if l.strip()]

            results.append(
                PageResult(
                    page_number=idx,
                    image_path="",
                    engine_used="pdfplumber",
                    fallback_reason=None,
                    text=text,
                    lines=lines,
                    conf_promedio=1.0,
                    conf_mediana=1.0,
                    conf_min=1.0,
                    conf_max=1.0,
                    conf_std=0.0,
                    lineas_baja_confianza=0,
                    det_count=len(lines),
                    rec_count=len(lines),
                    tasa_descarte=0.0,
                    angle_detected=0,
                    tiene_tabla=False,
                    tiempo_paddle=None,
                    tiempo_qwen=None,
                    tiempo_total=time.time() - t_pagina,
                    line_scores=[],
                )
            )

    logger.info(
        f"pdfplumber leyo {len(results)} paginas en {time.time() - t_inicio:.1f}s"
    )
    return results
