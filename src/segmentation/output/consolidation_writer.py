from __future__ import annotations
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from models.document_result import DocumentResult
from segmentation.models.professional_section import ProfessionalSection

logger = logging.getLogger(__name__)


def write_consolidation_report(
    doc: DocumentResult,
    secciones: List[ProfessionalSection],
    output_dir: str,
) -> str:
    """
    Genera el reporte Markdown de profesionales consolidados.
    Este es el input directo al módulo de extracción.

    Args:
        doc:        DocumentResult original.
        secciones:  Lista consolidada de ProfessionalSection.
        output_dir: Directorio donde guardar el archivo.

    Returns:
        Ruta al archivo generado.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre = Path(doc.pdf_path).stem
    ruta   = os.path.join(output_dir, f"{nombre}_profesionales_{ts}.md")

    es_tipo_b = any(s.es_tipo_b for s in secciones)
    tipo_doc  = "Tipo B (bloques temáticos)" if es_tipo_b else "Tipo A (bloque único)"
    max_bloques = max((len(s.bloques_origen) for s in secciones), default=1)

    with open(ruta, "w", encoding="utf-8") as f:

        # ── Encabezado ────────────────────────────────────────────────────────
        f.write(f"# Profesionales — {Path(doc.pdf_path).name}\n\n")
        f.write(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**PDF:** `{doc.pdf_path}`\n\n")

        # ── Resumen ───────────────────────────────────────────────────────────
        f.write("## Resumen\n\n")
        f.write("| Métrica | Valor |\n|---------|-------|\n")
        f.write(f"| Total profesionales | {len(secciones)} |\n")
        f.write(f"| Tipo de documento | {tipo_doc} |\n")
        if es_tipo_b:
            f.write(f"| Bloques por profesional | {max_bloques} |\n")
        f.write(f"| Total páginas del documento | {doc.total_pages} |\n\n")

        # ── Tabla de profesionales ────────────────────────────────────────────
        f.write("## Lista de profesionales\n\n")

        if es_tipo_b:
            f.write("| # | Cargo | N° | Págs totales | Bloques | Pág. inicio |\n")
            f.write("|---|-------|----|-------------|---------|-------------|\n")
            for sec in secciones:
                bloques_str = " · ".join(str(b) for b in sec.bloques_origen)
                f.write(
                    f"| {sec.section_index} | {sec.cargo} | "
                    f"{sec.numero or '—'} | {sec.total_pages} | "
                    f"{bloques_str} | {sec.separator_page} |\n"
                )
        else:
            f.write("| # | Cargo | Págs | Pág. inicio | Pág. fin |\n")
            f.write("|---|-------|------|-------------|----------|\n")
            for sec in secciones:
                pag_fin = sec.pages[-1].page_number if sec.pages else "—"
                f.write(
                    f"| {sec.section_index} | {sec.cargo} | "
                    f"{sec.total_pages} | {sec.separator_page} | {pag_fin} |\n"
                )

        f.write("\n")

        # ── Detalle por profesional ───────────────────────────────────────────
        f.write("## Detalle por profesional\n\n")

        for sec in secciones:
            numero_str = f" N°{sec.numero}" if sec.numero else ""
            f.write(f"### {sec.section_index}. {sec.cargo}{numero_str}\n\n")

            if sec.es_tipo_b:
                f.write("**Bloques de origen:**\n\n")
                for i, bloque in enumerate(sec.bloques_origen, 1):
                    f.write(f"- Bloque {i}: páginas {bloque}\n")
                f.write("\n")

            f.write(f"**Total páginas:** {sec.total_pages}  \n")
            f.write(f"**Página separadora:** {sec.separator_page}\n\n")

            # Preview primeras 3 páginas de contenido (excluir separadora)
            paginas_contenido = [
                p for p in sec.pages
                if p.page_number != sec.separator_page
                and not p.is_error
                and p.text.strip()
            ]

            if paginas_contenido:
                f.write("**Preview (primeras 2 páginas de contenido):**\n\n")
                for page in paginas_contenido[:2]:
                    preview = page.text[:300].replace("\n", "  \n")
                    f.write(f"_Página {page.page_number}:_\n```\n{preview}\n```\n\n")

            f.write("---\n\n")

    logger.info(f"Reporte de profesionales → {ruta}")
    return ruta