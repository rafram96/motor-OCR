"""
Wrapper script para invocar motor-OCR desde subprocess.

Usado por extractor-Bases_TDR para procesar PDFs escaneados.
No modificar: este archivo es generado y mantenido por extractor-Bases_TDR.
"""

import json
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import process_document, process_and_segment

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: subprocess_wrapper.py <args_file> <results_file>")
        sys.exit(1)

    args_file = sys.argv[1]
    results_file = sys.argv[2]

    try:
        # Cargar argumentos
        with open(args_file) as f:
            args = json.load(f)

        # Determinar modo: ocr_only o segmentation
        mode = args.pop("mode", "segmentation")  # Default: segmentation
        pdf_name = Path(args['pdf_path']).name

        if mode == "ocr_only":
            # Solo OCR, sin segmentación
            print(f"[subprocess_wrapper] Iniciando OCR (mode=ocr_only) con PDF: {pdf_name}")
            doc = process_document(**args)

            # Serializar DocumentResult a JSON (evita problemas de pickle con imports)
            result_data = {
                "mode": "ocr_only",
                "total_pages": doc.total_pages,
                "pages_paddle": doc.pages_paddle,
                "pages_qwen": doc.pages_qwen,
                "pages_error": doc.pages_error,
                "conf_promedio_documento": doc.conf_promedio_documento,
                "tiempo_total": doc.tiempo_total,
                "full_text": doc.full_text,
            }

            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False)

            print(
                f"[subprocess_wrapper] OK: {doc.total_pages} páginas procesadas "
                f"({doc.pages_paddle} Paddle, {doc.pages_qwen} Qwen, {doc.pages_error} errores)"
            )

        else:
            # OCR + Segmentación por profesionales
            print(f"[subprocess_wrapper] Iniciando OCR + Segmentación con PDF: {pdf_name}")
            doc, secciones = process_and_segment(**args)

            # Serializar a JSON
            result_data = {
                "mode": "segmentation",
                "doc": {
                    "total_pages": doc.total_pages,
                    "pages_paddle": doc.pages_paddle,
                    "pages_qwen": doc.pages_qwen,
                    "pages_error": doc.pages_error,
                    "conf_promedio_documento": doc.conf_promedio_documento,
                    "tiempo_total": doc.tiempo_total,
                    "full_text": doc.full_text,
                },
                "secciones": [
                    {
                        "section_index": sec.section_index,
                        "cargo": sec.cargo,
                        "numero": sec.numero,
                        "total_pages": sec.total_pages,
                        "full_text": sec.full_text,
                    }
                    for sec in secciones
                ],
            }

            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False)

            print(
                f"[subprocess_wrapper] OK: {doc.total_pages} páginas, {len(secciones)} profesionales "
                f"({doc.pages_paddle} Paddle, {doc.pages_qwen} Qwen, {doc.pages_error} errores)"
            )

        sys.exit(0)

    except Exception as e:
        print(f"[subprocess_wrapper] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
