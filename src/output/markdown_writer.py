import os
import logging
from datetime import datetime
from pathlib import Path

from models.document_result import DocumentResult
from models.page_result import PageResult
from config import UMBRAL_CONFIANZA_PROMEDIO, UMBRAL_TASA_DESCARTE, UMBRAL_CONFIANZA_LINEA

logger = logging.getLogger(__name__)


def write_document_report(doc: DocumentResult, output_dir: str) -> tuple[str, str]:
    """
    Genera dos archivos Markdown por documento procesado:
    - {nombre}_metricas_{ts}.md  → métricas y calidad por página
    - {nombre}_texto_{ts}.md     → texto extraído página a página

    Args:
        doc:        DocumentResult con todas las páginas procesadas.
        output_dir: Directorio donde guardar los archivos.

    Returns:
        (ruta_metricas, ruta_texto)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre    = Path(doc.pdf_path).stem
    pages_ord = sorted(doc.pages, key=lambda p: p.page_number)

    ruta_metricas = os.path.join(output_dir, f"{nombre}_metricas_{ts}.md")
    ruta_texto    = os.path.join(output_dir, f"{nombre}_texto_{ts}.md")

    _write_metricas(doc, pages_ord, ruta_metricas)
    _write_texto(doc, pages_ord, ruta_texto)

    logger.info(f"Markdown generado → {ruta_metricas}")
    logger.info(f"Markdown generado → {ruta_texto}")

    return ruta_metricas, ruta_texto


# ── Reporte de métricas ───────────────────────────────────────────────────────

def _write_metricas(doc: DocumentResult, pages: list[PageResult], ruta: str) -> None:
    with open(ruta, "w", encoding="utf-8") as f:

        # ── Encabezado ────────────────────────────────────────────────────────
        f.write(f"# Métricas OCR — {Path(doc.pdf_path).name}\n\n")
        f.write(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**PDF:** `{doc.pdf_path}`\n\n")

        # ── Resumen global ────────────────────────────────────────────────────
        f.write("## Resumen global\n\n")
        f.write("| Métrica | Valor |\n|---------|-------|\n")
        f.write(f"| Total páginas | {doc.total_pages} |\n")
        f.write(f"| Procesadas con Paddle | {doc.pages_paddle} |\n")
        f.write(f"| Procesadas con Qwen (fallback) | {doc.pages_qwen} |\n")
        f.write(f"| Páginas con error | {doc.pages_error} |\n")
        f.write(f"| Confianza promedio del documento | {doc.conf_promedio_documento:.4f} |\n")
        f.write(f"| Tiempo total | {doc.tiempo_total:.1f}s |\n\n")

        # Semáforo global
        if doc.pages_error > 0:
            estado_global = f"🔴 {doc.pages_error} páginas con error"
        elif doc.conf_promedio_documento < UMBRAL_CONFIANZA_PROMEDIO:
            estado_global = f"🟡 Confianza global baja ({doc.conf_promedio_documento:.3f})"
        else:
            estado_global = "🟢 Procesamiento correcto"
        f.write(f"**Estado general:** {estado_global}\n\n")

        # ── Páginas con error ─────────────────────────────────────────────────
        errores = [p for p in pages if p.is_error]
        if errores:
            f.write("## ⚠️ Páginas con error\n\n")
            f.write("| Página | Razón |\n|--------|-------|\n")
            for p in errores:
                f.write(f"| {p.page_number} | {p.fallback_reason or 'desconocido'} |\n")
            f.write("\n")

        # ── Páginas con fallback Qwen ─────────────────────────────────────────
        fallbacks = [p for p in pages if p.engine_used == "qwen"]
        if fallbacks:
            f.write("## Páginas procesadas con Qwen (fallback)\n\n")
            f.write("| Página | Razón del fallback |\n|--------|--------------------|\n")
            for p in fallbacks:
                f.write(f"| {p.page_number} | {p.fallback_reason or '-'} |\n")
            f.write("\n")

        # ── Métricas por página ───────────────────────────────────────────────
        f.write("## Métricas por página\n\n")
        f.write(
            "| # | Engine | Conf.Prom | Descarte | Ángulo | "
            "Líneas bajas | Estado |\n"
        )
        f.write(
            "|---|--------|-----------|----------|--------|"
            "-------------|--------|\n"
        )

        for p in pages:
            if p.is_error:
                f.write(
                    f"| {p.page_number} | error | — | — | — | — | "
                    f"🔴 {p.fallback_reason} |\n"
                )
                continue

            conf_str     = f"{p.conf_promedio:.4f}" if p.conf_promedio is not None else "—"
            descarte_str = f"{p.tasa_descarte*100:.1f}%"
            engine_icon  = "🔵" if p.engine_used == "paddle" else "🟠"

            if p.engine_used == "paddle":
                if p.conf_promedio is not None and p.conf_promedio < UMBRAL_CONFIANZA_PROMEDIO:
                    estado = "🟡 conf baja"
                elif p.tasa_descarte > UMBRAL_TASA_DESCARTE:
                    estado = "🟡 descarte alto"
                elif p.lineas_baja_confianza > 0:
                    estado = f"🟡 {p.lineas_baja_confianza} líneas bajas"
                else:
                    estado = "🟢"
            else:
                estado = "🟠 qwen"

            f.write(
                f"| {p.page_number} | {engine_icon} {p.engine_used} | "
                f"{conf_str} | {descarte_str} | {p.angle_detected}° | "
                f"{p.lineas_baja_confianza} | {estado} |\n"
            )

        f.write("\n")

        # ── Detalle de líneas bajas (solo paddle, solo las que tienen) ────────
        paginas_con_lineas_bajas = [
            p for p in pages
            if p.engine_used == "paddle" and p.lineas_baja_confianza > 0
        ]
        if paginas_con_lineas_bajas:
            f.write("## Líneas con baja confianza por página\n\n")
            for p in paginas_con_lineas_bajas:
                f.write(f"### Página {p.page_number}\n\n")
                f.write("| # | Score | Texto |\n|---|-------|-------|\n")
                # Necesitamos reconstruir scores por línea — solo tenemos el agregado
                # Se deja este bloque como placeholder para cuando se exponga
                # rec_scores junto a lines en PageResult (mejora futura)
                f.write(
                    f"| — | — | "
                    f"({p.lineas_baja_confianza} líneas bajo {UMBRAL_CONFIANZA_LINEA}) |\n"
                )
                f.write("\n")

        # ── Configuración usada ───────────────────────────────────────────────
        f.write("## Configuración\n\n")
        f.write("| Parámetro | Valor |\n|-----------|-------|\n")
        f.write(f"| UMBRAL_CONFIANZA_PROMEDIO | {UMBRAL_CONFIANZA_PROMEDIO} |\n")
        f.write(f"| UMBRAL_TASA_DESCARTE | {UMBRAL_TASA_DESCARTE} |\n")
        f.write(f"| UMBRAL_CONFIANZA_LINEA | {UMBRAL_CONFIANZA_LINEA} |\n")


# ── Reporte de texto ──────────────────────────────────────────────────────────

def _write_texto(doc: DocumentResult, pages: list[PageResult], ruta: str) -> None:
    with open(ruta, "w", encoding="utf-8") as f:

        f.write(f"# Texto extraído — {Path(doc.pdf_path).name}\n\n")
        f.write(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**PDF:** `{doc.pdf_path}`  \n")
        f.write(
            f"**Páginas:** {doc.total_pages} | "
            f"Paddle: {doc.pages_paddle} | "
            f"Qwen: {doc.pages_qwen} | "
            f"Error: {doc.pages_error}\n\n"
        )
        f.write("---\n\n")

        for p in pages:
            # ── Encabezado de página ──────────────────────────────────────────
            engine_tag = {
                "paddle": "🔵 paddle",
                "qwen":   "🟠 qwen",
                "error":  "🔴 error",
            }.get(p.engine_used, p.engine_used)

            conf_tag = (
                f" · conf {p.conf_promedio:.3f}"
                if p.conf_promedio is not None
                else ""
            )
            angle_tag = (
                f" · {p.angle_detected}°"
                if p.angle_detected != 0
                else ""
            )

            f.write(f"## Página {p.page_number}  ")
            f.write(f"_{engine_tag}{conf_tag}{angle_tag}_\n\n")

            if p.is_error:
                f.write(f"> ⚠️ Error: {p.fallback_reason}\n\n")
            elif not p.text.strip():
                f.write("> _(página en blanco o sin texto reconocido)_\n\n")
            else:
                f.write("```\n")
                f.write(p.text)
                f.write("\n```\n\n")

            f.write("---\n\n")