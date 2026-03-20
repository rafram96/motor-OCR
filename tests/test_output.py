from pathlib import Path

from models.document_result import DocumentResult
from models.page_result import PageResult
from output.markdown_writer import write_document_report


def make_page(
    page_number,
    engine_used="paddle",
    fallback_reason=None,
    text="Texto",
    lines=None,
    conf_promedio=0.9,
    conf_mediana=0.9,
    conf_min=0.8,
    conf_max=0.95,
    conf_std=0.05,
    lineas_baja_confianza=0,
    det_count=3,
    rec_count=3,
    tasa_descarte=0.0,
    angle_detected=0,
    tiene_tabla=False,
    tiempo_paddle=1.0,
    tiempo_qwen=None,
    tiempo_total=1.5,
):
    return PageResult(
        page_number=page_number,
        image_path=f"/tmp/page_{page_number:04d}.png",
        engine_used=engine_used,
        fallback_reason=fallback_reason,
        text=text,
        lines=lines if lines is not None else [text] if text else [],
        conf_promedio=conf_promedio,
        conf_mediana=conf_mediana,
        conf_min=conf_min,
        conf_max=conf_max,
        conf_std=conf_std,
        lineas_baja_confianza=lineas_baja_confianza,
        det_count=det_count,
        rec_count=rec_count,
        tasa_descarte=tasa_descarte,
        angle_detected=angle_detected,
        tiene_tabla=tiene_tabla,
        tiempo_paddle=tiempo_paddle,
        tiempo_qwen=tiempo_qwen,
        tiempo_total=tiempo_total,
    )


def test_write_document_report_generates_metric_and_text_files(tmp_path):
    page_2 = make_page(
        2,
        engine_used="qwen",
        fallback_reason="confianza baja",
        text="texto fallback",
        lines=["texto fallback"],
        conf_promedio=None,
        conf_mediana=None,
        conf_min=None,
        conf_max=None,
        conf_std=None,
        tiempo_paddle=0.8,
        tiempo_qwen=1.2,
        tiempo_total=2.0,
    )
    page_1 = make_page(1, text="texto paddle", lines=["texto paddle"], conf_promedio=0.91)
    page_3 = PageResult.error_placeholder(3, "/tmp/page_0003.png", "worker error")

    doc = DocumentResult(
        pdf_path="/tmp/documento.pdf",
        total_pages=3,
        pages=[page_2, page_1, page_3],
        tiempo_total=7.5,
    )
    doc.compute_summary()

    ruta_metricas, ruta_texto = write_document_report(doc, str(tmp_path))

    assert Path(ruta_metricas).exists()
    assert Path(ruta_texto).exists()

    metricas = Path(ruta_metricas).read_text(encoding="utf-8")
    texto = Path(ruta_texto).read_text(encoding="utf-8")

    assert "# Métricas OCR" in metricas
    assert "## Resumen global" in metricas
    assert "Procesadas con Paddle" in metricas
    assert "Páginas procesadas con Qwen (fallback)" in metricas
    assert "Páginas con error" in metricas
    assert "## Métricas por página" in metricas
    assert "worker error" in metricas

    assert "# Texto extraído" in texto
    assert "## Página 1" in texto
    assert "_🔵 paddle" in texto
    assert "_🟠 qwen" in texto
    assert "> ⚠️ Error: worker error" in texto
    assert texto.index("## Página 1") < texto.index("## Página 2") < texto.index("## Página 3")