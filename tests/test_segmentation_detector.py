import builtins
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.page_result import PageResult
import segmentation.detector as detector
from segmentation.config import MAX_LINEAS_SEPARADORA


class DummyClient:
    def __init__(self, content: str = "", should_fail: bool = False):
        self.content = content
        self.should_fail = should_fail
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        if self.should_fail:
            raise RuntimeError("qwen down")

        message = SimpleNamespace(content=self.content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


@pytest.fixture(autouse=True)
def reset_detector_singleton():
    detector._client = None
    yield
    detector._client = None


def make_page(page_number: int, lines: list[str], text: str | None = None) -> PageResult:
    text_value = text if text is not None else "\n".join(lines)
    return PageResult(
        page_number=page_number,
        image_path=f"/tmp/page_{page_number:04d}.png",
        engine_used="paddle",
        fallback_reason=None,
        text=text_value,
        lines=lines,
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
        tiene_tabla=False,
        tiempo_paddle=0.1,
        tiempo_qwen=None,
        tiempo_total=0.1,
    )


def test_es_candidata_separadora_bounds():
    assert detector.es_candidata_separadora(make_page(1, [])) is False
    assert detector.es_candidata_separadora(make_page(2, ["solo una"])) is True
    assert detector.es_candidata_separadora(make_page(3, ["x"] * MAX_LINEAS_SEPARADORA)) is True
    assert detector.es_candidata_separadora(make_page(4, ["x"] * (MAX_LINEAS_SEPARADORA + 1))) is False


def test_normalizar_cargo_applies_known_replacements():
    normalizado = detector.normalizar_cargo("  jefe de supervicion y varolizaciones ")

    assert "Supervisión" in normalizado
    assert "Valorizaciones" in normalizado
    assert normalizado.startswith("Jefe")


def test_fuzzy_detect_cargo_returns_false_when_rapidfuzz_missing(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rapidfuzz" or name.startswith("rapidfuzz"):
            raise ImportError("rapidfuzz missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    encontrado, cargo = detector.fuzzy_detect_cargo("Gerente de contrato")

    assert encontrado is False
    assert cargo == ""


def test_fuzzy_detect_cargo_returns_match_with_stub(monkeypatch):
    rapidfuzz_stub = types.ModuleType("rapidfuzz")
    rapidfuzz_stub.fuzz = SimpleNamespace(partial_ratio=lambda a, b: 90)
    rapidfuzz_stub.process = SimpleNamespace(
        extractOne=lambda texto, choices, scorer: ("gerente de contrato", 90, 0)
    )
    monkeypatch.setitem(sys.modules, "rapidfuzz", rapidfuzz_stub)

    encontrado, cargo = detector.fuzzy_detect_cargo("gerente contrato")

    assert encontrado is True
    assert cargo == "Gerente de Contrato"


def test_confirmar_con_qwen_returns_error_on_image_read_failure():
    page = make_page(10, ["titulo"])
    page.image_path = "/tmp/does_not_exist.png"

    es_sep, cargo, confianza = detector._confirmar_con_qwen(page)

    assert (es_sep, cargo, confianza) == (False, "", "error")


def test_confirmar_con_qwen_returns_error_on_api_failure(monkeypatch, tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"img")
    page = make_page(11, ["titulo"])
    page.image_path = str(image)

    monkeypatch.setattr(detector, "_get_client", lambda: DummyClient(should_fail=True))

    es_sep, cargo, confianza = detector._confirmar_con_qwen(page)

    assert (es_sep, cargo, confianza) == (False, "", "error")


def test_confirmar_con_qwen_parses_json_response(monkeypatch, tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"img")
    page = make_page(12, ["titulo"])
    page.image_path = str(image)

    content = (
        "<think>analizando</think>\n"
        "```json\n"
        '{"es_separadora": true, "cargo": "Jefe de supervicion", "confianza": "Alta"}\n'
        "```"
    )
    monkeypatch.setattr(detector, "_get_client", lambda: DummyClient(content=content))

    es_sep, cargo, confianza = detector._confirmar_con_qwen(page)

    assert es_sep is True
    assert cargo == "Jefe de supervicion"
    assert confianza == "alta"


def test_confirmar_con_qwen_handles_invalid_json(monkeypatch, tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"img")
    page = make_page(13, ["titulo"])
    page.image_path = str(image)

    monkeypatch.setattr(detector, "_get_client", lambda: DummyClient(content="no-json"))

    es_sep, cargo, confianza = detector._confirmar_con_qwen(page)

    assert (es_sep, cargo, confianza) == (False, "", "error")


def test_evaluar_separadora_accepts_qwen_medium_confidence(monkeypatch):
    page = make_page(21, ["Jefe de supervicion"], text="Jefe de supervicion")
    monkeypatch.setattr(detector, "_confirmar_con_qwen", lambda p: (True, "Jefe de supervicion", "media"))

    times = iter([10.0, 10.25])
    monkeypatch.setattr(detector.time, "time", lambda: next(times))

    sep = detector.evaluar_separadora(page)

    assert sep.es_separadora is True
    assert sep.metodo == "qwen"
    assert sep.cargo_normalizado == "Jefe De Supervisión"
    assert sep.tiempo_deteccion == pytest.approx(0.25)


def test_evaluar_separadora_uses_fuzzy_fallback(monkeypatch):
    page = make_page(22, ["gerente contrato"], text="gerente contrato")
    monkeypatch.setattr(detector, "_confirmar_con_qwen", lambda p: (False, "", "baja"))
    monkeypatch.setattr(detector, "fuzzy_detect_cargo", lambda texto: (True, "Gerente de Contrato"))

    times = iter([20.0, 20.5])
    monkeypatch.setattr(detector.time, "time", lambda: next(times))

    sep = detector.evaluar_separadora(page)

    assert sep.es_separadora is True
    assert sep.metodo == "fuzzy_fallback"
    assert sep.confianza_qwen == "fuzzy"
    assert sep.cargo_normalizado == "Gerente De Contrato"


def test_evaluar_separadora_discards_when_qwen_and_fuzzy_fail(monkeypatch):
    page = make_page(23, ["contenido extenso"], text="contenido extenso")
    monkeypatch.setattr(detector, "_confirmar_con_qwen", lambda p: (False, "", "error"))
    monkeypatch.setattr(detector, "fuzzy_detect_cargo", lambda texto: (False, ""))

    times = iter([30.0, 30.4])
    monkeypatch.setattr(detector.time, "time", lambda: next(times))

    sep = detector.evaluar_separadora(page)

    assert sep.es_separadora is False
    assert sep.metodo == "descartada"
    assert sep.confianza_qwen == "error"