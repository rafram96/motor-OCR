from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

from config import OUTPUT_DIR, SAVE_MARKDOWN
from models.document_result import DocumentResult
from segmentation.consolidator import consolidar_secciones
from segmentation.models.professional_section import ProfessionalSection
from segmentation.output.consolidation_writer import write_consolidation_report

from .markdown_writer import write_document_report_simple
from .reader import read_pdf_pages
from .segmenter import segment_textonly

logger = logging.getLogger(__name__)


def process_with_pdfplumber(
    pdf_path: str,
    pages: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
    keep_images: bool = False,
) -> Tuple[DocumentResult, List[ProfessionalSection]]:
    """
    Fast-path de segmentacion para PDFs digitales (texto nativo).

    Firma identica a process_and_segment de main.py para simetria.
    Genera los mismos archivos .md que el motor-OCR completo:
    - {nombre}_metricas_{ts}.md  (formato simplificado, sin metricas OCR)
    - {nombre}_texto_{ts}.md
    - {nombre}_profesionales_{ts}.md  (formato identico via consolidation_writer reusado)

    Args:
        pdf_path: Ruta al PDF digital (con capa de texto).
        pages: Paginas a procesar (None = todas).
        output_dir: Directorio de salida. Default: OUTPUT_DIR/{nombre}.
        keep_images: Ignorado (no se generan imagenes).

    Returns:
        (DocumentResult, List[ProfessionalSection])
    """
    pdf_path = str(pdf_path)
    base_name = Path(pdf_path).stem
    work_dir = str(Path(output_dir or OUTPUT_DIR) / base_name)
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    t_inicio = time.time()
    logger.info(f"Procesando con pdfplumber: {Path(pdf_path).name}")

    # ── 1. Leer PDF con pdfplumber ────────────────────────────────────────────
    page_results = read_pdf_pages(pdf_path, pages=pages)

    # ── 2. Construir DocumentResult ───────────────────────────────────────────
    doc = DocumentResult(
        pdf_path=pdf_path,
        total_pages=len(page_results),
        pages=page_results,
        tiempo_total=time.time() - t_inicio,
    )
    doc.compute_summary()

    logger.info(
        f"pdfplumber completado: {doc.total_pages} pags | "
        f"pdfplumber={doc.pages_pdfplumber} error={doc.pages_error} | "
        f"t={doc.tiempo_total:.1f}s"
    )

    # ── 3. Segmentar (fuzzy + qwen2.5:14b texto-only) ────────────────────────
    secciones_raw, descartadas, delimitadores = segment_textonly(doc)
    secciones = consolidar_secciones(secciones_raw, delimitadores)

    doc.tiempo_total = time.time() - t_inicio

    # ── 4. Markdown reports ──────────────────────────────────────────────────
    if SAVE_MARKDOWN:
        try:
            write_document_report_simple(doc, work_dir)
        except Exception as e:
            logger.warning(f"No se pudo generar Markdown OCR simple: {e}")

        try:
            write_consolidation_report(doc, secciones, work_dir)
        except Exception as e:
            logger.warning(f"No se pudo generar reporte de profesionales: {e}")

    logger.info(
        f"Segmentacion textonly completada: {len(secciones)} profesionales | "
        f"t_total={doc.tiempo_total:.1f}s"
    )

    return doc, secciones
