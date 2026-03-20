import sys
from pathlib import Path

SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.document_result import DocumentResult
from models.page_result import PageResult
import segmentation.segmenter as segmenter
from segmentation.models.separator_page import SeparatorPage


def make_page(page_number: int, text: str, tiene_tabla: bool = False) -> PageResult:
    return PageResult(
        page_number=page_number,
        image_path=f"/tmp/page_{page_number:04d}.png",
        engine_used="paddle",
        fallback_reason=None,
        text=text,
        lines=[line for line in text.splitlines() if line.strip()],
        conf_promedio=0.9,
        conf_mediana=0.9,
        conf_min=0.8,
        conf_max=0.95,
        conf_std=0.05,
        lineas_baja_confianza=0,
        det_count=2,
        rec_count=2,
        tasa_descarte=0.0,
        angle_detected=0,
        tiene_tabla=tiene_tabla,
        tiempo_paddle=0.2,
        tiempo_qwen=None,
        tiempo_total=0.2,
    )


def sep_result(page: PageResult, es_separadora: bool, cargo: str = "", metodo: str = "descartada") -> SeparatorPage:
    return SeparatorPage(
        page_number=page.page_number,
        image_path=page.image_path,
        line_count=page.line_count,
        raw_text=page.text,
        es_separadora=es_separadora,
        cargo_detectado=cargo,
        cargo_normalizado=cargo,
        confianza_qwen="media" if es_separadora else "baja",
        metodo=metodo,
        tiempo_deteccion=0.1,
    )


def test_segment_document_returns_empty_when_no_separators(monkeypatch):
    pages = [make_page(1, "a"), make_page(2, "b"), make_page(3, "c")]
    doc = DocumentResult(pdf_path="/tmp/doc.pdf", total_pages=3, pages=pages)

    monkeypatch.setattr(segmenter, "es_candidata_separadora", lambda p: True)
    monkeypatch.setattr(segmenter, "evaluar_separadora", lambda p: sep_result(p, False))

    secciones = segmenter.segment_document(doc)

    assert secciones == []


def test_segment_document_groups_pages_between_separators(monkeypatch):
    pages = [
        make_page(3, "contenido 3"),
        make_page(1, "gerente"),
        make_page(6, "contenido 6"),
        make_page(5, "contenido 5", tiene_tabla=True),
        make_page(2, "contenido 2"),
        make_page(4, "supervisor"),
    ]
    doc = DocumentResult(pdf_path="/tmp/doc.pdf", total_pages=6, pages=pages)

    monkeypatch.setattr(segmenter, "es_candidata_separadora", lambda p: p.page_number in {1, 4})

    def fake_evaluar(page):
        if page.page_number == 1:
            return sep_result(page, True, cargo="Gerente de Contrato", metodo="qwen")
        if page.page_number == 4:
            return sep_result(page, True, cargo="Supervisor de Obra", metodo="fuzzy_fallback")
        return sep_result(page, False)

    monkeypatch.setattr(segmenter, "evaluar_separadora", fake_evaluar)

    secciones = segmenter.segment_document(doc)

    assert len(secciones) == 2

    assert secciones[0].section_index == 1
    assert secciones[0].cargo == "Gerente de Contrato"
    assert secciones[0].page_numbers == [1, 2, 3]
    assert secciones[0].total_pages == 3
    assert secciones[0].has_tables is False

    assert secciones[1].section_index == 2
    assert secciones[1].cargo == "Supervisor de Obra"
    assert secciones[1].page_numbers == [4, 5, 6]
    assert secciones[1].total_pages == 3
    assert secciones[1].has_tables is True