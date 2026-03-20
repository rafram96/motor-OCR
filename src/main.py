import os
import shutil
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from config import MAX_WORKERS, OUTPUT_DIR, SAVE_MARKDOWN
from models.page_result import PageResult
from models.document_result import DocumentResult
from pipeline.pdf_to_images import pdf_to_images
from pipeline.page_processor import process_page

logger = logging.getLogger(__name__)

# ── Worker initializer ────────────────────────────────────────────────────────
# Se ejecuta una vez por proceso worker al arrancar el pool.
# Carga los modelos de PaddleOCR en el contexto del worker,
# evitando recargarlos en cada llamada a process_page.

def _worker_init() -> None:
    import os as _os
    _os.environ["FLAGS_use_mkldnn"] = "0"
    from engines.paddle_engine import get_ocr
    get_ocr()   # fuerza la carga del modelo en este worker
    logger.debug("Worker OCR inicializado.")


def _process_page_worker(args: tuple) -> PageResult:
    """Wrapper serializable para ProcessPoolExecutor."""
    image_path, page_number = args
    return process_page(image_path, page_number)


# ── API pública ───────────────────────────────────────────────────────────────

def process_document(
    pdf_path: str,
    pages: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
    keep_images: bool = False,
) -> DocumentResult:
    """
    Procesa un PDF completo y retorna texto estructurado por página.

    Args:
        pdf_path:    Ruta al PDF escaneado.
        pages:       Lista de páginas a extraer (base 1). None = todas.
        output_dir:  Directorio para imágenes temporales y salida Markdown.
                     Por defecto usa OUTPUT_DIR de config.py.
        keep_images: Si True, conserva las imágenes PNG después de procesar.
                     Si False (por defecto), las elimina al terminar.

    Returns:
        DocumentResult con todas las páginas procesadas.
    """
    pdf_path  = str(pdf_path)
    base_name = Path(pdf_path).stem
    work_dir  = str(Path(output_dir or OUTPUT_DIR) / base_name)

    t_inicio = time.time()
    logger.info(f"Procesando: {os.path.basename(pdf_path)}")

    # ── 1. PDF → imágenes ─────────────────────────────────────────────────────
    try:
        image_paths = pdf_to_images(pdf_path, work_dir, pages=pages)
    except Exception as e:
        logger.error(f"Error convirtiendo PDF a imágenes: {e}")
        raise

    logger.info(f"  {len(image_paths)} páginas a procesar (modo secuencial)")

    # ── 2. Procesar páginas en paralelo ───────────────────────────────────────
    # Inferir número de página desde el nombre del archivo (pagina_0001.png → 1)
    def _page_number_from_path(path: str) -> int:
        stem = Path(path).stem          # "pagina_0001"
        try:
            return int(stem.split("_")[-1])
        except ValueError:
            return 0

    args = [
        (path, _page_number_from_path(path))
        for path in image_paths
    ]

    results: List[PageResult] = []

    # === PARALLEL_POOL_DISABLED_START ===
    # with ProcessPoolExecutor(
    #     max_workers=MAX_WORKERS,
    #     initializer=_worker_init,
    # ) as executor:
    #     future_to_page = {
    #         executor.submit(_process_page_worker, a): a[1]
    #         for a in args
    #     }
    #     for future in as_completed(future_to_page):
    #         page_num = future_to_page[future]
    #         try:
    #             result = future.result()
    #             results.append(result)
    #             status = (
    #                 f"✓ paddle conf={result.conf_promedio:.3f}"
    #                 if result.engine_used == "paddle" and result.conf_promedio
    #                 else f"⚠ {result.engine_used}"
    #             )
    #             logger.debug(f"  Página {page_num}: {status}")
    #         except Exception as e:
    #             logger.error(f"  Página {page_num}: excepción en worker — {e}")
    #             results.append(
    #                 PageResult.error_placeholder(
    #                     page_number=page_num,
    #                     image_path=args[page_num - 1][0] if page_num <= len(args) else "",
    #                     reason=f"worker_exception: {e}",
    #                 )
    #             )
    # === PARALLEL_POOL_DISABLED_END ===

    for i, (path, page_num) in enumerate(args, 1):
        logger.info(f"  [{i}/{len(args)}] Procesando página {page_num}...")
        result = process_page(path, page_num)
        results.append(result)
        conf_str = f"{result.conf_promedio:.3f}" if result.conf_promedio is not None else "N/A"
        logger.info(f"    → {result.engine_used} | conf={conf_str}")

    # ── 3. Construir DocumentResult ───────────────────────────────────────────
    results.sort(key=lambda x: x.page_number)

    doc = DocumentResult(
        pdf_path=pdf_path,
        total_pages=len(results),
        pages=results,
        tiempo_total=time.time() - t_inicio,
    )
    doc.compute_summary()

    logger.info(
        f"Completado: {doc.total_pages} págs | "
        f"paddle={doc.pages_paddle} qwen={doc.pages_qwen} error={doc.pages_error} | "
        f"conf={doc.conf_promedio_documento:.3f} | "
        f"t={doc.tiempo_total:.1f}s"
    )

    # ── 4. Markdown de métricas (opcional) ────────────────────────────────────
    if SAVE_MARKDOWN:
        try:
            from output.markdown_writer import write_document_report
            write_document_report(doc, work_dir)
        except Exception as e:
            logger.warning(f"No se pudo generar Markdown: {e}")

    # ── 5. Limpiar imágenes temporales ────────────────────────────────────────
    if not keep_images:
        pages_dir = Path(work_dir) / "pages"
        if pages_dir.exists():
            shutil.rmtree(str(pages_dir))
            logger.debug(f"Imágenes temporales eliminadas: {pages_dir}")

    return doc


# ── API pública — OCR + Segmentación ─────────────────────────────────────────

def process_and_segment(
    pdf_path: str,
    pages: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
    keep_images: bool = False,
) -> Tuple[DocumentResult, list]:
    """
    Flujo completo: OCR + segmentación por profesional.

    Args:
        pdf_path:    Ruta al PDF escaneado.
        pages:       Lista de páginas a extraer (base 1). None = todas.
        output_dir:  Directorio de salida.
        keep_images: Si True, conserva las imágenes PNG.

    Returns:
        (DocumentResult, List[ProfessionalSection])
    """
    from segmentation.segmenter import segment_document
    from segmentation.detector import es_candidata_separadora, evaluar_separadora
    from segmentation.output.markdown_writer import write_segmentation_report

    base_name = Path(pdf_path).stem
    work_dir  = str(Path(output_dir or OUTPUT_DIR) / base_name)

    # ── OCR ───────────────────────────────────────────────────────────────────
    doc = process_document(
        pdf_path=pdf_path,
        pages=pages,
        output_dir=output_dir,
        keep_images=keep_images,
    )

    # ── Segmentación ──────────────────────────────────────────────────────────
    secciones = segment_document(doc)

    # Recopilar candidatas descartadas para el reporte
    candidatas_descartadas = []
    for page in sorted(doc.pages, key=lambda p: p.page_number):
        if es_candidata_separadora(page):
            sep = evaluar_separadora(page)
            if not sep.es_separadora:
                candidatas_descartadas.append(sep)

    # ── Markdown segmentación (opcional) ──────────────────────────────────────
    if SAVE_MARKDOWN:
        try:
            write_segmentation_report(doc, secciones, candidatas_descartadas, work_dir)
        except Exception as e:
            logger.warning(f"No se pudo generar reporte de segmentación: {e}")

    return doc, secciones