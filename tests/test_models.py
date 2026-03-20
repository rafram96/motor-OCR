import sys
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.document_result import DocumentResult
from models.page_result import PageResult


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


def test_error_placeholder_creates_error_page():
    page = PageResult.error_placeholder(4, "/tmp/page_0004.png", "boom")

    assert page.is_error is True
    assert page.engine_used == "error"
    assert page.fallback_reason == "boom"
    assert page.text == ""
    assert page.lines == []
    assert page.tasa_descarte == 0.0
    assert page.tiempo_total == 0.0


def test_line_count_ignores_blank_lines():
    page = make_page(1, lines=["uno", "", "  ", "dos"], text="uno\ndos")

    assert page.line_count == 2


def test_document_result_properties_and_summary():
    page_2 = make_page(
        2,
        engine_used="qwen",
        fallback_reason="confianza baja",
        text="dos",
        lines=["dos"],
        conf_promedio=None,
        conf_mediana=None,
        conf_min=None,
        conf_max=None,
        conf_std=None,
        tiempo_paddle=0.8,
        tiempo_qwen=1.2,
        tiempo_total=2.0,
    )
    page_1 = make_page(1, text="uno", lines=["uno"], conf_promedio=0.8, conf_mediana=0.8)
    page_3 = PageResult.error_placeholder(3, "/tmp/page_0003.png", "worker error")

    doc = DocumentResult(
        pdf_path="/tmp/doc.pdf",
        total_pages=3,
        pages=[page_2, page_1, page_3],
        tiempo_total=12.3,
    )
    doc.compute_summary()

    assert doc.full_text == "uno\n\ndos"
    assert doc.text_by_page == {2: "dos", 1: "uno", 3: ""}
    assert [p.page_number for p in doc.error_pages] == [3]
    assert [p.page_number for p in doc.fallback_pages] == [2]
    assert doc.pages_paddle == 1
    assert doc.pages_qwen == 1
    assert doc.pages_error == 1
    assert doc.conf_promedio_documento == pytest.approx(0.8)