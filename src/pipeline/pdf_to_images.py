from __future__ import annotations

from pathlib import Path
from typing import Any


def pdf_to_images(pdf_path: str | Path, *, dpi: int = 200) -> list[Any]:
    """Renderiza un PDF a una lista de imágenes (PIL.Image.Image).

    Importa dependencias de forma lazy para permitir que el proyecto se importe
    aun si falta instalar PyMuPDF/Pillow.
    """

    try:
        import fitz  # type: ignore  # PyMuPDF
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "No se pudo importar `fitz` (PyMuPDF). Instala `pymupdf`."
        ) from exc

    try:
        from PIL import Image  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("No se pudo importar `PIL`. Instala `Pillow`.") from exc

    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)

    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    images: list[Any] = []
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    finally:
        doc.close()

    return images
