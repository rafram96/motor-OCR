import sys
from pathlib import Path

SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import UMBRAL_CONFIANZA_PROMEDIO, UMBRAL_TASA_DESCARTE
from models.page_result import PageResult
from pipeline.decision import debe_usar_qwen


def make_page(conf_promedio, tasa_descarte, is_error=False):
    if is_error:
        return PageResult.error_placeholder(1, "/tmp/page.png", "error")

    return PageResult(
        page_number=1,
        image_path="/tmp/page.png",
        engine_used="paddle",
        fallback_reason=None,
        text="texto",
        lines=["texto"],
        conf_promedio=conf_promedio,
        conf_mediana=conf_promedio,
        conf_min=conf_promedio,
        conf_max=conf_promedio,
        conf_std=0.0,
        lineas_baja_confianza=0,
        det_count=1,
        rec_count=1,
        tasa_descarte=tasa_descarte,
        angle_detected=0,
        tiene_tabla=False,
        tiempo_paddle=0.1,
        tiempo_qwen=None,
        tiempo_total=0.1,
    )


def test_debe_usar_qwen_for_low_confidence():
    page = make_page(UMBRAL_CONFIANZA_PROMEDIO - 0.01, 0.0)

    usar_qwen, razon = debe_usar_qwen(page)

    assert usar_qwen is True
    assert "confianza promedio baja" in razon


def test_debe_usar_qwen_for_high_discard_rate():
    page = make_page(0.95, UMBRAL_TASA_DESCARTE + 0.1)

    usar_qwen, razon = debe_usar_qwen(page)

    assert usar_qwen is True
    assert "tasa de descarte alta" in razon


def test_debe_not_use_qwen_for_blank_page():
    page = make_page(None, 0.0)

    usar_qwen, razon = debe_usar_qwen(page)

    assert usar_qwen is False
    assert razon == ""


def test_debe_not_use_qwen_for_error_page():
    page = make_page(0.9, 0.0, is_error=True)

    usar_qwen, razon = debe_usar_qwen(page)

    assert usar_qwen is False
    assert razon == ""