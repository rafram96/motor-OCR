import os
import logging
import time
from pathlib import Path
from typing import List, Optional

from pdf2image import convert_from_path

from config import PDF_DPI, PDF_IMAGE_FORMAT, POPPLER_PATH

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
                    filename = f"pagina_{page_num:04d}.{PDF_IMAGE_FORMAT.lower()}"
                    ruta = pages_dir / filename
                    imgs[0].save(str(ruta), PDF_IMAGE_FORMAT)
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
        # Extraer todas las páginas de una vez (más eficiente)
        logger.info(f"Extrayendo todas las páginas de {os.path.basename(pdf_path)}")
        try:
            imgs = convert_from_path(pdf_path, dpi=dpi, poppler_path=POPPLER_PATH)
            total = len(imgs)
            progreso_cada = max(1, total // 10) if total else 1
            for i, img in enumerate(imgs, start=1):
                filename = f"pagina_{i:04d}.{PDF_IMAGE_FORMAT.lower()}"
                ruta = pages_dir / filename
                img.save(str(ruta), PDF_IMAGE_FORMAT)
                image_paths.append(str(ruta.resolve()))
                if i == 1 or i % progreso_cada == 0 or i == total:
                    elapsed = time.time() - t_start
                    promedio = elapsed / i
                    restante = max(0.0, promedio * (total - i))
                    pct = (i / total) * 100 if total else 100.0
                    logger.info(
                        f"Guardado de imágenes progreso: {i}/{total} ({pct:.1f}%), "
                        f"ETA {_format_eta(restante)}"
                    )
            logger.info(f"  ✓ {len(imgs)} páginas extraídas")
        except Exception as e:
            raise RuntimeError(f"Error convirtiendo PDF: {e}") from e

    if not image_paths:
        raise RuntimeError(f"No se extrajo ninguna imagen de {pdf_path}")

    # Garantizar orden por número de página
    image_paths.sort()
    return image_paths