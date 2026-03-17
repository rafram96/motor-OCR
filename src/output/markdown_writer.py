from __future__ import annotations

from pathlib import Path

from models.document_result import DocumentResult


def write_markdown(result: DocumentResult, output_path: str | Path) -> Path:
    """Exporta métricas y texto a un archivo Markdown."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    title = result.source or output_path.stem

    lines: list[str] = []
    lines.append(f"# OCR — {title}")
    lines.append("")

    lines.append("## Métricas")
    lines.append("")
    lines.append(f"- Páginas: {len(result.pages)}")
    lines.append("")

    lines.append("## Texto")
    lines.append("")

    for page in result.pages:
        lines.append(f"### Página {page.page_number}")
        lines.append("")
        lines.append(f"- Motor: {page.engine}")
        if page.confidence is not None:
            lines.append(f"- Confianza: {page.confidence:.3f}")
        if page.metrics:
            for k, v in page.metrics.items():
                lines.append(f"- {k}: {v}")
        lines.append("")

        page_text = (page.text or "").strip()
        lines.append(page_text if page_text else "_(sin texto)_")
        lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path
