from __future__ import annotations
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from models.document_result import DocumentResult
from models.page_result import PageResult

logger = logging.getLogger(__name__)


def write_document_report_simple(
    doc: DocumentResult,
    output_dir: str,
) -> tuple[str, str]:
    """
    Version simplificada de write_document_report para PDFs procesados con pdfplumber.
    No incluye metricas de OCR (angulo, tasa_descarte, lineas bajas) porque pdfplumber
    extrae texto directo del PDF sin reconocimiento.

    Genera dos archivos con nombres IDENTICOS al motor-OCR completo
    (para que md_parser de Alpamayo los encuentre igual):
    - {nombre}_metricas_{ts}.md
    - {nombre}_texto_{ts}.md

    Returns:
        (ruta_metricas, ruta_texto)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre = Path(doc.pdf_path).stem
    pages_ord = sorted(doc.pages, key=lambda p: p.page_number)

    ruta_metricas = os.path.join(output_dir, f"{nombre}_metricas_{ts}.md")
    ruta_texto = os.path.join(output_dir, f"{nombre}_texto_{ts}.md")

    _write_metricas(doc, pages_ord, ruta_metricas)
    _write_texto(doc, pages_ord, ruta_texto)

    logger.info(f"Markdown generado -> {ruta_metricas}")
    logger.info(f"Markdown generado -> {ruta_texto}")
    return ruta_metricas, ruta_texto


def _write_metricas(doc: DocumentResult, pages: List[PageResult], ruta: str) -> None:
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(f"# Metricas OCR — {Path(doc.pdf_path).name}\n\n")
        f.write(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**PDF:** `{doc.pdf_path}`  \n")
        f.write("**Engine:** pdfplumber (extraccion de texto digital)\n\n")

        f.write("## Resumen global\n\n")
        f.write("| Metrica | Valor |\n|---------|-------|\n")
        f.write(f"| Total paginas | {doc.total_pages} |\n")
        f.write(f"| Procesadas con pdfplumber | {doc.pages_pdfplumber} |\n")
        f.write(f"| Paginas con error | {doc.pages_error} |\n")
        f.write(f"| Confianza promedio | {doc.conf_promedio_documento:.4f} |\n")
        f.write(f"| Tiempo total | {doc.tiempo_total:.1f}s |\n\n")

        if doc.pages_error > 0:
            f.write(f"**Estado general:** 🔴 {doc.pages_error} paginas con error\n\n")
        else:
            f.write("**Estado general:** 🟢 Procesamiento correcto (texto nativo del PDF)\n\n")

        errores = [p for p in pages if p.is_error]
        if errores:
            f.write("## ⚠️ Paginas con error\n\n")
            f.write("| Pagina | Razon |\n|--------|-------|\n")
            for p in errores:
                f.write(f"| {p.page_number} | {p.fallback_reason or 'desconocido'} |\n")
            f.write("\n")

        f.write("## Resumen por pagina\n\n")
        f.write("| # | Engine | Lineas | Caracteres |\n")
        f.write("|---|--------|--------|------------|\n")
        for p in pages:
            if p.is_error:
                f.write(f"| {p.page_number} | 🔴 error | — | — |\n")
                continue
            chars = len(p.text)
            n_lines = len([l for l in p.lines if l.strip()])
            f.write(
                f"| {p.page_number} | 🟣 pdfplumber | {n_lines} | {chars} |\n"
            )


def _write_texto(doc: DocumentResult, pages: List[PageResult], ruta: str) -> None:
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(f"# Texto extraido — {Path(doc.pdf_path).name}\n\n")
        f.write(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**PDF:** `{doc.pdf_path}`  \n")
        f.write(
            f"**Paginas:** {doc.total_pages} | "
            f"pdfplumber: {doc.pages_pdfplumber} | "
            f"Error: {doc.pages_error}\n\n"
        )
        f.write("---\n\n")

        for p in pages:
            if p.is_error:
                f.write(f"## Pagina {p.page_number}  _🔴 error_\n\n")
                f.write(f"> ⚠️ Error: {p.fallback_reason}\n\n")
            else:
                f.write(f"## Pagina {p.page_number}  _🟣 pdfplumber_\n\n")
                if not p.text.strip():
                    f.write("> _(pagina en blanco o sin texto reconocido)_\n\n")
                else:
                    f.write("```\n")
                    f.write(p.text)
                    f.write("\n```\n\n")
            f.write("---\n\n")
