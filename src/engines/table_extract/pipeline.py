"""
Pipeline de extraccion de tablas: PDF -> imagenes -> PP-Structure -> matrices.

Llamado desde subprocess_wrapper.py cuando mode == "table_extract".

Renderiza solo las paginas pedidas (no todo el PDF) usando pdf_to_images
existente (con POPPLER_PATH y PDF_DPI del config global).
"""
from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import List, Optional

from pipeline.pdf_to_images import pdf_to_images

from .pp_structure import extract_tables_from_image

logger = logging.getLogger(__name__)


def extract_tables_from_pdf(
    pdf_path: str,
    paginas: List[int],
    output_dir: Optional[str] = None,
) -> dict:
    """
    Extrae todas las tablas de las paginas especificadas de un PDF.

    Args:
        pdf_path: ruta absoluta al PDF
        paginas: lista de numeros de pagina (1-based) a procesar
        output_dir: directorio para imagenes intermedias (default: temp/{stem})

    Returns:
        dict con:
          - "pdf_path": str
          - "paginas_solicitadas": list[int]
          - "tablas": list[dict]   # ver pp_structure.extract_tables_from_image
          - "tiempo_total": float
          - "n_paginas_procesadas": int
          - "errores": list[str]
    """
    pdf_path = str(pdf_path)
    if not Path(pdf_path).exists():
        return {
            "pdf_path": pdf_path,
            "paginas_solicitadas": paginas,
            "tablas": [],
            "tiempo_total": 0.0,
            "n_paginas_procesadas": 0,
            "errores": [f"PDF no encontrado: {pdf_path}"],
        }

    base_name = Path(pdf_path).stem
    work_dir = Path(output_dir) if output_dir else Path.cwd() / "tmp_tables" / base_name
    work_dir.mkdir(parents=True, exist_ok=True)

    paginas_unicas = sorted(set(p for p in paginas if isinstance(p, int) and p > 0))

    if not paginas_unicas:
        return {
            "pdf_path": pdf_path,
            "paginas_solicitadas": paginas,
            "tablas": [],
            "tiempo_total": 0.0,
            "n_paginas_procesadas": 0,
            "errores": ["Lista de paginas vacia o invalida"],
        }

    logger.info(
        "[table_extract] Procesando %d pags de %s -> %s",
        len(paginas_unicas), Path(pdf_path).name, work_dir,
    )

    t0 = time.time()
    errores: list[str] = []

    # ── 1. Renderizar las paginas como imagenes ────────────────────────────
    try:
        image_paths = pdf_to_images(
            pdf_path=pdf_path,
            output_dir=str(work_dir),
            pages=paginas_unicas,
        )
    except Exception as e:
        logger.exception("[table_extract] pdf_to_images fallo: %s", e)
        return {
            "pdf_path": pdf_path,
            "paginas_solicitadas": paginas,
            "tablas": [],
            "tiempo_total": time.time() - t0,
            "n_paginas_procesadas": 0,
            "errores": [f"pdf_to_images: {e}"],
        }

    # Mapear ruta de imagen a numero de pagina (1-based)
    img_a_pagina: dict[str, int] = {}
    for img_path in image_paths:
        stem = Path(img_path).stem  # "pagina_0042"
        try:
            num = int(stem.split("_")[-1])
            img_a_pagina[img_path] = num
        except (ValueError, IndexError):
            errores.append(f"Nombre de imagen no parseable: {img_path}")

    # ── 2. PP-Structure por imagen ──────────────────────────────────────────
    todas_tablas: list[dict] = []
    n_procesadas = 0

    for img_path, num_pag in sorted(img_a_pagina.items(), key=lambda kv: kv[1]):
        try:
            tablas = extract_tables_from_image(img_path, num_pag)
            todas_tablas.extend(tablas)
            n_procesadas += 1
        except Exception as e:
            logger.exception(
                "[table_extract] Pagina %d fallo: %s", num_pag, e,
            )
            errores.append(f"pag {num_pag}: {e}")

    elapsed = time.time() - t0
    logger.info(
        "[table_extract] OK: %d tablas extraidas de %d/%d paginas en %.1fs",
        len(todas_tablas), n_procesadas, len(paginas_unicas), elapsed,
    )

    return {
        "pdf_path": pdf_path,
        "paginas_solicitadas": paginas_unicas,
        "tablas": todas_tablas,
        "tiempo_total": elapsed,
        "n_paginas_procesadas": n_procesadas,
        "errores": errores,
    }
