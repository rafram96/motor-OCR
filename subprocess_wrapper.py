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


def _serializar_secciones(secciones):
    return [
        {
            "section_index": sec.section_index,
            "cargo": sec.cargo,
            "cargo_raw": sec.cargo_raw,
            "numero": sec.numero,
            "total_pages": sec.total_pages,
            "page_numbers": sec.page_numbers,
            "bloques_origen": [
                {"start": b.start, "end": b.end}
                for b in sec.bloques_origen
            ],
            "es_tipo_b": sec.es_tipo_b,
            "full_text": sec.full_text,
        }
        for sec in secciones
    ]


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

        # Determinar modo: ocr_only | segmentation | pdfplumber_segmentation
        mode = args.pop("mode", "segmentation")
        pdf_name = Path(args['pdf_path']).name

        if mode == "ocr_only":
            # Solo OCR, sin segmentación
            print(f"[subprocess_wrapper] Iniciando OCR (mode=ocr_only) con PDF: {pdf_name}")
            doc = process_document(**args)

            result_data = {
                "mode": "ocr_only",
                "total_pages": doc.total_pages,
                "pages_paddle": doc.pages_paddle,
                "pages_qwen": doc.pages_qwen,
                "pages_pdfplumber": doc.pages_pdfplumber,
                "pages_error": doc.pages_error,
                "conf_promedio_documento": doc.conf_promedio_documento,
                "tiempo_total": doc.tiempo_total,
                "full_text": doc.full_text,
                "engine": "motor_ocr",
            }

            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False)

            print(
                f"[subprocess_wrapper] OK: {doc.total_pages} páginas procesadas "
                f"({doc.pages_paddle} Paddle, {doc.pages_qwen} Qwen, {doc.pages_error} errores)"
            )

        elif mode == "pdfplumber_segmentation":
            # Fast-path: pdfplumber + segmentación texto-only (para PDFs digitales)
            from engines.pdfplumber import process_with_pdfplumber

            print(f"[subprocess_wrapper] Iniciando PDFPLUMBER + Segmentación con PDF: {pdf_name}")
            doc, secciones = process_with_pdfplumber(**args)

            result_data = {
                "mode": "pdfplumber_segmentation",
                "doc": {
                    "total_pages": doc.total_pages,
                    "pages_paddle": doc.pages_paddle,
                    "pages_qwen": doc.pages_qwen,
                    "pages_pdfplumber": doc.pages_pdfplumber,
                    "pages_error": doc.pages_error,
                    "conf_promedio_documento": doc.conf_promedio_documento,
                    "tiempo_total": doc.tiempo_total,
                    "full_text": doc.full_text,
                    "engine": "pdfplumber",
                },
                "secciones": _serializar_secciones(secciones),
            }

            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False)

            print(
                f"[subprocess_wrapper] OK: {doc.total_pages} páginas, {len(secciones)} profesionales "
                f"(pdfplumber={doc.pages_pdfplumber}, errores={doc.pages_error})"
            )

        elif mode == "table_extract":
            # Extracción de tablas (B.1 / B.2 del TDR) con PP-Structure V3.
            # Usado por Alpamayo Capa 2 del pipeline 3-capas.
            # Args esperados: {pdf_path, paginas: [int], output_dir?}
            from engines.table_extract import extract_tables_from_pdf

            paginas = args.get("paginas") or args.get("pages") or []
            print(
                f"[subprocess_wrapper] Iniciando TABLE_EXTRACT con PDF: {pdf_name} "
                f"(paginas={paginas})"
            )

            result_data = extract_tables_from_pdf(
                pdf_path=args["pdf_path"],
                paginas=paginas,
                output_dir=args.get("output_dir"),
            )
            result_data["mode"] = "table_extract"

            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False)

            n_tablas = len(result_data.get("tablas", []))
            n_procesadas = result_data.get("n_paginas_procesadas", 0)
            tiempo = result_data.get("tiempo_total", 0.0)
            print(
                f"[subprocess_wrapper] OK: {n_tablas} tablas extraidas de "
                f"{n_procesadas} paginas en {tiempo:.1f}s"
            )

        else:
            # OCR + Segmentación por profesionales (flujo completo motor-OCR)
            print(f"[subprocess_wrapper] Iniciando OCR + Segmentación con PDF: {pdf_name}")
            doc, secciones = process_and_segment(**args)

            result_data = {
                "mode": "segmentation",
                "doc": {
                    "total_pages": doc.total_pages,
                    "pages_paddle": doc.pages_paddle,
                    "pages_qwen": doc.pages_qwen,
                    "pages_pdfplumber": doc.pages_pdfplumber,
                    "pages_error": doc.pages_error,
                    "conf_promedio_documento": doc.conf_promedio_documento,
                    "tiempo_total": doc.tiempo_total,
                    "full_text": doc.full_text,
                    "engine": "motor_ocr",
                },
                "secciones": _serializar_secciones(secciones),
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
