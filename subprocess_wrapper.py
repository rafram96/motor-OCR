"""
Wrapper script para invocar motor-OCR desde subprocess.

Usado por extractor-Bases_TDR para procesar PDFs escaneados.
No modificar: este archivo es generado y mantenido por extractor-Bases_TDR.
"""

import json
import logging
import pickle
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

            # Guardar solo DocumentResult
            with open(results_file, "wb") as f:
                pickle.dump(doc, f)

            print(
                f"[subprocess_wrapper] OK: {doc.total_pages} páginas procesadas "
                f"({doc.pages_paddle} Paddle, {doc.pages_qwen} Qwen, {doc.pages_error} errores)"
            )

        else:
            # OCR + Segmentación por profesionales
            print(f"[subprocess_wrapper] Iniciando OCR + Segmentación con PDF: {pdf_name}")
            doc, secciones = process_and_segment(**args)

            # Guardar (DocumentResult, List[ProfessionalSection])
            with open(results_file, "wb") as f:
                pickle.dump((doc, secciones), f)

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
