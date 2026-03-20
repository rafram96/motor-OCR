from __future__ import annotations
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from models.document_result import DocumentResult
from segmentation.models.separator_page import SeparatorPage
from segmentation.models.professional_section import ProfessionalSection

logger = logging.getLogger(__name__)


def write_segmentation_report(
    doc: DocumentResult,
    secciones: List[ProfessionalSection],
    candidatas_descartadas: List[SeparatorPage],
    output_dir: str,
) -> str:
    """
    Genera un reporte Markdown de la segmentación para revisión manual.

    Args:
        doc:                    DocumentResult original.
        secciones:              Secciones detectadas por el segmentador.
        candidatas_descartadas: Páginas evaluadas pero no aceptadas como separadoras.
        output_dir:             Directorio donde guardar el archivo.

    Returns:
        Ruta al archivo generado.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre = Path(doc.pdf_path).stem
    ruta   = os.path.join(output_dir, f"{nombre}_segmentacion_{ts}.md")

    with open(ruta, "w", encoding="utf-8") as f:
        _write_header(f, doc, secciones, candidatas_descartadas)
        _write_secciones(f, secciones)
        _write_descartadas(f, candidatas_descartadas)
        _write_texto_por_seccion(f, secciones)

    logger.info(f"Reporte de segmentación → {ruta}")
    return ruta


# ── Secciones del reporte ─────────────────────────────────────────────────────

def _write_header(f, doc, secciones, descartadas) -> None:
    total_candidatas = len(secciones) + len(descartadas)

    f.write(f"# Segmentación — {Path(doc.pdf_path).name}\n\n")
    f.write(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
    f.write(f"**PDF:** `{doc.pdf_path}`\n\n")

    f.write("## Resumen\n\n")
    f.write("| Métrica | Valor |\n|---------|-------|\n")
    f.write(f"| Total páginas | {doc.total_pages} |\n")
    f.write(f"| Candidatas evaluadas | {total_candidatas} |\n")
    f.write(f"| Profesionales detectados | {len(secciones)} |\n")
    f.write(f"| Candidatas descartadas | {len(descartadas)} |\n\n")

    if not secciones:
        f.write(
            "> ⚠️ **No se detectaron separadoras.** "
            "Revisar si el documento tiene la estructura esperada "
            "o ajustar `MAX_LINEAS_SEPARADORA`.\n\n"
        )


def _write_secciones(f, secciones: List[ProfessionalSection]) -> None:
    if not secciones:
        return

    f.write("## Secciones detectadas\n\n")
    f.write(
        "| # | Cargo | Cargo raw | Pág. inicio | Pág. fin | "
        "Total págs | Tablas | Método |\n"
    )
    f.write(
        "|---|-------|-----------|-------------|----------|"
        "-----------|--------|--------|\n"
    )

    for sec in secciones:
        pag_fin  = sec.pages[-1].page_number if sec.pages else "—"
        tablas   = "✓" if sec.has_tables else "—"
        # recuperar método desde la primera página (separator)
        metodo   = "—"
        f.write(
            f"| {sec.section_index} | {sec.cargo} | {sec.cargo_raw} | "
            f"{sec.separator_page} | {pag_fin} | "
            f"{sec.total_pages} | {tablas} | {metodo} |\n"
        )
    f.write("\n")


def _write_descartadas(f, descartadas: List[SeparatorPage]) -> None:
    if not descartadas:
        return

    f.write("## Candidatas descartadas\n\n")
    f.write("| Página | Líneas | Qwen conf | Razón | Texto (primeras 80 chars) |\n")
    f.write("|--------|--------|-----------|-------|---------------------------|\n")

    for sep in descartadas:
        texto_preview = sep.raw_text[:80].replace("\n", " ").replace("|", "\\|")
        f.write(
            f"| {sep.page_number} | {sep.line_count} | "
            f"{sep.confianza_qwen} | {sep.metodo} | {texto_preview} |\n"
        )
    f.write("\n")


def _write_texto_por_seccion(f, secciones: List[ProfessionalSection]) -> None:
    if not secciones:
        return

    f.write("## Texto por sección\n\n")
    f.write(
        "> Vista rápida del texto extraído por profesional. "
        "Útil para verificar que la segmentación es correcta.\n\n"
    )

    for sec in secciones:
        f.write(f"### {sec.section_index}. {sec.cargo}\n\n")
        f.write(f"_Páginas {sec.separator_page}–{sec.pages[-1].page_number if sec.pages else '?'} · {sec.total_pages} páginas_\n\n")

        # Primeras 3 páginas como preview
        for page in sec.pages[:3]:
            if page.is_error:
                f.write(f"**Página {page.page_number}:** ⚠️ error\n\n")
                continue
            preview = page.text[:400].replace("\n", "  \n") if page.text else "_(sin texto)_"
            f.write(f"**Página {page.page_number}:**\n\n")
            f.write(f"```\n{preview}\n```\n\n")

        if sec.total_pages > 3:
            f.write(f"_... {sec.total_pages - 3} páginas más_\n\n")

        f.write("---\n\n")