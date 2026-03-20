import os
import sys
import shutil
import json
import pickle
import logging
import time
import tempfile
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

from config import MAX_WORKERS, OUTPUT_DIR, SAVE_MARKDOWN
from models.page_result import PageResult
from models.document_result import DocumentResult
from pipeline.pdf_to_images import pdf_to_images
from pipeline.page_processor import process_page

logger = logging.getLogger(__name__)


# ── Worker initializer ────────────────────────────────────────────────────────

def _worker_init() -> None:
    import os as _os
    _os.environ["FLAGS_use_mkldnn"] = "0"
    from engines.paddle_engine import get_ocr
    get_ocr()
    logger.debug("Worker OCR inicializado.")


def _process_page_worker(args: tuple) -> PageResult:
    """Wrapper serializable para ProcessPoolExecutor."""
    image_path, page_number = args
    return process_page(image_path, page_number)


# ── API pública — OCR ─────────────────────────────────────────────────────────

def process_document(
    pdf_path: str,
    pages: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
    keep_images: bool = False,
) -> DocumentResult:
    """
    Procesa un PDF completo y retorna texto estructurado por página.

    Flujo de dos pasadas para evitar conflictos de VRAM entre PaddleOCR y Qwen:
    1. Paddle procesa TODAS las páginas primero (libera VRAM al terminar)
    2. Qwen procesa solo las páginas que necesitan fallback

    Args:
        pdf_path:    Ruta al PDF escaneado.
        pages:       Lista de páginas a extraer (base 1). None = todas.
        output_dir:  Directorio para imágenes temporales y salida Markdown.
        keep_images: Si True, conserva las imágenes PNG después de procesar.

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

    logger.info(f"  {len(image_paths)} páginas a procesar")

    def _page_number_from_path(path: str) -> int:
        stem = Path(path).stem
        try:
            return int(stem.split("_")[-1])
        except ValueError:
            return 0

    args_list = [
        (path, _page_number_from_path(path))
        for path in image_paths
    ]

    # ── 2. Pasada 1 — Paddle para todas las páginas ───────────────────────────
    from pipeline.decision import debe_usar_qwen

    # Pasada 1 — Paddle en subproceso separado
    logger.info("  Pasada 1: PaddleOCR (subproceso)...")

    src_dir = str(Path(__file__).parent)
    args_json = json.dumps(args_list)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        output_pkl = tmp.name

    try:
        subprocess.run(
            [
                sys.executable,
                str(Path(src_dir) / "pipeline" / "paddle_worker.py"),
                src_dir,
                args_json,
                output_pkl,
            ],
            check=True,
        )
        with open(output_pkl, "rb") as f:
            resultados_paddle = pickle.load(f)
    finally:
        os.unlink(output_pkl)

    # ── 3. Pasada 2 — Qwen solo para páginas que lo necesitan ─────────────────

    paginas_qwen = [
        (r, path, page_num)
        for r, (path, page_num) in zip(resultados_paddle, args_list)
        if debe_usar_qwen(r)[0]
    ]

    if paginas_qwen:
        # ── Liberar VRAM de Paddle antes de Qwen ─────────────────────────────
        import paddle
        paddle.device.cuda.empty_cache()
        # También destruir la instancia singleton
        import engines.paddle_engine as _pe
        _pe._ocr_instance = None
        import gc
        gc.collect()
        logger.info("  VRAM liberada, iniciando Qwen...")

        from engines import qwen_engine
        logger.info(f"  Pasada 2: Qwen fallback para {len(paginas_qwen)} páginas...")
        qwen_map: dict = {}

        for paddle_r, path, page_num in tqdm(paginas_qwen, desc="Qwen fallback", unit="pág"):
            _, razon = debe_usar_qwen(paddle_r)
            try:
                qwen_r = qwen_engine.extract_text(
                    image_path=path,
                    page_number=page_num,
                    fallback_reason=razon,
                    tiempo_paddle=paddle_r.tiempo_paddle,
                )
            except Exception as e:
                logger.error(f"  Página {page_num}: error en qwen — {e}")
                qwen_r = PageResult.error_placeholder(page_num, path, f"qwen_exception: {e}")
            qwen_map[page_num] = qwen_r

        results = [
            qwen_map.get(r.page_number, r)
            for r in resultados_paddle
        ]
    else:
        logger.info("  Pasada 2: no se necesita Qwen.")
        results = resultados_paddle

    # ── 4. Construir DocumentResult ───────────────────────────────────────────
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

    # ── 5. Markdown OCR (opcional) ────────────────────────────────────────────
    if SAVE_MARKDOWN:
        try:
            from output.markdown_writer import write_document_report
            write_document_report(doc, work_dir)
        except Exception as e:
            logger.warning(f"No se pudo generar Markdown OCR: {e}")

    # ── 6. Limpiar imágenes temporales ────────────────────────────────────────
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