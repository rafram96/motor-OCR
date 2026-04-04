import os
import logging
import time
from pathlib import Path
from typing import List, Optional

from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import ImageEnhance

from config import PDF_DPI, PDF_IMAGE_FORMAT, PDF_JPEG_QUALITY, POPPLER_PATH, PDF_BATCH_SIZE, IMAGE_ENHANCE_CONTRAST

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


def _enhance(img):
    """Aplica mejora de contraste si está habilitada en config."""
    if IMAGE_ENHANCE_CONTRAST != 1.0:
        return ImageEnhance.Contrast(img).enhance(IMAGE_ENHANCE_CONTRAST)
    return img


def _save_kwargs() -> dict:
    """Kwargs extra para img.save() según el formato configurado."""
    if PDF_IMAGE_FORMAT.upper() == "JPEG":
        return {"quality": PDF_JPEG_QUALITY}
    return {}


def pdf_to_images(
    pdf_path: str,
    output_dir: str,
    dpi: int = PDF_DPI,
    pages: Optional[List[int]] = None,
) -> List[str]:
    """
    Convierte cada página de un PDF en una imagen PNG y las guarda en disco.

    Args:
        pdf_path:   Ruta al PDF de entrada.
        output_dir: Directorio donde guardar las imágenes. Se crea si no existe.
        dpi:        Resolución de las imágenes (300 por defecto).
        pages:      Lista de números de página a extraer (base 1).
                    Si es None, extrae todas las páginas.

    Returns:
        Lista de rutas absolutas a las imágenes, ordenadas por número de página.

    Raises:
        FileNotFoundError: Si el PDF no existe.
        RuntimeError:      Si no se pudo convertir ninguna página.
    """
    pdf_path = str(pdf_path)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

    pages_dir = Path(output_dir) / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    image_paths: List[str] = []
    t_start = time.time()

    if pages:
        # Extraer páginas específicas una por una
        logger.info(f"Extrayendo {len(pages)} páginas de {os.path.basename(pdf_path)}")
        pages_sorted = sorted(pages)
        total = len(pages_sorted)
        progreso_cada = max(1, total // 10)
        for idx, page_num in enumerate(pages_sorted, start=1):
            try:
                imgs = convert_from_path(
                    pdf_path,
                    dpi=dpi,
                    first_page=page_num,
                    last_page=page_num,
                    poppler_path=POPPLER_PATH
                )
                if imgs:
                    img = _enhance(imgs[0])
                    filename = f"pagina_{page_num:04d}.{PDF_IMAGE_FORMAT.lower()}"
                    ruta = pages_dir / filename
                    img.save(str(ruta), PDF_IMAGE_FORMAT, **_save_kwargs())
                    image_paths.append(str(ruta.resolve()))
                    logger.debug(f"  ✓ Página {page_num} → {filename}")
                else:
                    logger.warning(f"  ✗ Página {page_num}: convert_from_path devolvió vacío")
            except Exception as e:
                logger.error(f"  ✗ Página {page_num} falló: {e}")

            if idx == 1 or idx % progreso_cada == 0 or idx == total:
                elapsed = time.time() - t_start
                promedio = elapsed / idx
                restante = max(0.0, promedio * (total - idx))
                pct = (idx / total) * 100
                logger.info(
                    f"Extracción PDF progreso: {idx}/{total} ({pct:.1f}%), "
                    f"ETA {_format_eta(restante)}"
                )
    else:
        # Extraer todas las páginas en batches para no explotar RAM
        logger.info(f"Extrayendo todas las páginas de {os.path.basename(pdf_path)}")
        try:
            info = pdfinfo_from_path(pdf_path, poppler_path=POPPLER_PATH)
            total = int(info["Pages"])
        except Exception as e:
            raise RuntimeError(f"Error obteniendo info del PDF: {e}") from e

        logger.info(f"  {total} páginas detectadas, batch_size={PDF_BATCH_SIZE}")
        global_idx = 0
        progreso_cada = max(1, total // 10) if total else 1

        for batch_start in range(1, total + 1, PDF_BATCH_SIZE):
            batch_end = min(batch_start + PDF_BATCH_SIZE - 1, total)
            try:
                imgs = convert_from_path(
                    pdf_path,
                    dpi=dpi,
                    first_page=batch_start,
                    last_page=batch_end,
                    poppler_path=POPPLER_PATH,
                )
            except Exception as e:
                logger.error(f"  ✗ Batch {batch_start}-{batch_end} falló: {e}")
                continue

            for i, img in enumerate(imgs):
                page_num = batch_start + i
                global_idx += 1
                img = _enhance(img)
                filename = f"pagina_{page_num:04d}.{PDF_IMAGE_FORMAT.lower()}"
                ruta = pages_dir / filename
                img.save(str(ruta), PDF_IMAGE_FORMAT, **_save_kwargs())
                image_paths.append(str(ruta.resolve()))

                if global_idx == 1 or global_idx % progreso_cada == 0 or global_idx == total:
                    elapsed = time.time() - t_start
                    promedio = elapsed / global_idx
                    restante = max(0.0, promedio * (total - global_idx))
                    pct = (global_idx / total) * 100 if total else 100.0
                    logger.info(
                        f"Guardado de imágenes progreso: {global_idx}/{total} ({pct:.1f}%), "
                        f"ETA {_format_eta(restante)}"
                    )

            del imgs  # libera RAM del batch

        logger.info(f"  ✓ {len(image_paths)} páginas extraídas")

    if not image_paths:
        raise RuntimeError(f"No se extrajo ninguna imagen de {pdf_path}")

    # Garantizar orden por número de página
    image_paths.sort()
    return image_paths
