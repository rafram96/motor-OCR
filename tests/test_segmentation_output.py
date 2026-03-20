import sys
from pathlib import Path

SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.document_result import DocumentResult
from models.page_result import PageResult
from segmentation.models.professional_section import ProfessionalSection
from segmentation.models.separator_page import SeparatorPage
from segmentation.output.markdown_writer import write_segmentation_report


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
        tiempo_paddle=0.1,
        tiempo_qwen=None,
        tiempo_total=0.1,
    )


def make_section(index: int, cargo: str, pages: list[PageResult]) -> ProfessionalSection:
    return ProfessionalSection(
        section_index=index,
        cargo=cargo,
        cargo_raw=cargo.upper(),
        separator_page=pages[0].page_number,
        pages=pages,
        total_pages=len(pages),
        has_tables=any(p.tiene_tabla for p in pages),
    )


def test_write_segmentation_report_generates_markdown(tmp_path):
    pages = [
        make_page(1, "gerente"),
        make_page(2, "contenido 2"),
        make_page(3, "supervisor"),
        make_page(4, "contenido 4", tiene_tabla=True),
    ]
    doc = DocumentResult(pdf_path="/tmp/expediente.pdf", total_pages=4, pages=pages)

    secciones = [
        make_section(1, "Gerente de Contrato", pages[:2]),
        make_section(2, "Supervisor de Obra", pages[2:]),
    ]
    descartadas = [
        SeparatorPage(
            page_number=5,
            image_path="/tmp/page_0005.png",
            line_count=4,
            raw_text="contenido no separador",
            es_separadora=False,
            cargo_detectado="",
            cargo_normalizado="",
            confianza_qwen="baja",
            metodo="descartada",
            tiempo_deteccion=0.2,
        )
    ]

    ruta = write_segmentation_report(doc, secciones, descartadas, str(tmp_path))

    contenido = Path(ruta).read_text(encoding="utf-8")

    assert Path(ruta).exists()
    assert "# Segmentación" in contenido
    assert "## Resumen" in contenido
    assert "## Secciones detectadas" in contenido
    assert "## Candidatas descartadas" in contenido
    assert "## Texto por sección" in contenido
    assert "Gerente de Contrato" in contenido
    assert "Supervisor de Obra" in contenido


def test_write_segmentation_report_without_sections_adds_warning(tmp_path):
    pages = [make_page(1, "contenido 1"), make_page(2, "contenido 2")]
    doc = DocumentResult(pdf_path="/tmp/expediente.pdf", total_pages=2, pages=pages)

    ruta = write_segmentation_report(doc, [], [], str(tmp_path))
    contenido = Path(ruta).read_text(encoding="utf-8")

    assert "No se detectaron separadoras" in contenido