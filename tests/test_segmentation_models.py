import sys
from pathlib import Path

SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.page_result import PageResult
from segmentation.models.professional_section import ProfessionalSection
from segmentation.models.separator_page import SeparatorPage


def make_page(page_number: int, text: str = "texto", tiene_tabla: bool = False) -> PageResult:
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
        tiempo_paddle=0.1,
        tiempo_qwen=None,
        tiempo_total=0.1,
    )


def test_professional_section_full_text_skips_errors_and_blanks():
    p1 = make_page(1, "uno")
    p2 = PageResult.error_placeholder(2, "/tmp/page_0002.png", "error")
    p3 = make_page(3, "   ")
    p4 = make_page(4, "dos")

    section = ProfessionalSection(
        section_index=1,
        cargo="Gerente de Contrato",
        cargo_raw="GERENTE DE CONTRATO",
        separator_page=1,
        pages=[p1, p2, p3, p4],
        total_pages=4,
        has_tables=False,
    )

    assert section.full_text == "uno\n\ndos"


def test_professional_section_page_numbers_reflect_input_order():
    section = ProfessionalSection(
        section_index=2,
        cargo="Supervisor de Obra",
        cargo_raw="SUPERVISOR DE OBRA",
        separator_page=5,
        pages=[make_page(5), make_page(6), make_page(7)],
        total_pages=3,
        has_tables=True,
    )

    assert section.page_numbers == [5, 6, 7]


def test_separator_page_dataclass_fields():
    sep = SeparatorPage(
        page_number=10,
        image_path="/tmp/page_0010.png",
        line_count=2,
        raw_text="Jefe de supervicion",
        es_separadora=True,
        cargo_detectado="Jefe de supervicion",
        cargo_normalizado="Jefe De Supervisión",
        confianza_qwen="media",
        metodo="qwen",
        tiempo_deteccion=0.45,
    )

    assert sep.es_separadora is True
    assert sep.page_number == 10
    assert sep.metodo == "qwen"