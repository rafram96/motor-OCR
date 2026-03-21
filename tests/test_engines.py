import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import engines.paddle_engine as paddle_engine
import engines.qwen_engine as qwen_engine


class DummyOCR:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def predict(self, image_path):
        self.calls.append(image_path)
        return self.response


class DummyClient:
    def __init__(self, content="", fail=False):
        self.content = content
        self.fail = fail
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("api down")

        message = SimpleNamespace(content=self.content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


def test_get_ocr_singleton(monkeypatch):
    class TrackingPaddleOCR:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            TrackingPaddleOCR.instances.append(self)

    monkeypatch.setattr(paddle_engine, "PaddleOCR", TrackingPaddleOCR)
    paddle_engine._ocr_instance = None

    first = paddle_engine.get_ocr()
    second = paddle_engine.get_ocr()

    assert first is second
    assert len(TrackingPaddleOCR.instances) == 1


def test_predict_parses_metrics_and_text(monkeypatch):
    dummy_ocr = DummyOCR(
        [
            {
                "rec_texts": ["Linea 1", " ", "Linea 3"],
                "rec_scores": [0.9, 0.7, 0.8],
                "dt_polys": [1, 2, 3],
                "doc_preprocessor_res": {"angle": 90},
            }
        ]
    )
    monkeypatch.setattr(paddle_engine, "get_ocr", lambda: dummy_ocr)
    times = iter([100.0, 100.5, 101.25])
    monkeypatch.setattr(paddle_engine.time, "time", lambda: next(times))

    result = paddle_engine.predict("/tmp/page.png", 7)

    assert dummy_ocr.calls == ["/tmp/page.png"]
    assert result.engine_used == "paddle"
    assert result.text == "Linea 1\nLinea 3"
    assert result.lines == ["Linea 1", "Linea 3"]
    assert result.det_count == 3
    assert result.rec_count == 3
    assert result.tasa_descarte == 0.0
    assert result.conf_promedio == pytest.approx(0.85)
    assert result.conf_mediana == pytest.approx(0.85)
    assert result.conf_min == pytest.approx(0.8)
    assert result.conf_max == pytest.approx(0.9)
    assert result.lineas_baja_confianza == 1
    assert result.angle_detected == 90
    assert result.tiempo_paddle == pytest.approx(0.5)
    assert result.tiempo_total == pytest.approx(1.25)


def test_predict_empty_result_returns_placeholder(monkeypatch):
    dummy_ocr = DummyOCR([])
    monkeypatch.setattr(paddle_engine, "get_ocr", lambda: dummy_ocr)
    # Logger llama time.time() internamente — dar valores extra
    times = iter([200.0] + [200.2] * 20)
    monkeypatch.setattr(paddle_engine.time, "time", lambda: next(times))

    result = paddle_engine.predict("/tmp/page.png", 3)

    assert result.is_error is True
    assert result.fallback_reason == "paddle_empty_result"


def test_predict_exception_returns_placeholder(monkeypatch):
    class FailingOCR:
        def predict(self, image_path):
            raise RuntimeError("ocr boom")

    monkeypatch.setattr(paddle_engine, "get_ocr", lambda: FailingOCR())
    times = iter([300.0] + [300.3] * 20)
    monkeypatch.setattr(paddle_engine.time, "time", lambda: next(times))

    result = paddle_engine.predict("/tmp/page.png", 5)

    assert result.is_error is True
    assert result.fallback_reason.startswith("paddle_exception:")


def test_extract_text_strips_think_and_builds_result(monkeypatch):
    client = DummyClient(content="<think>probando</think>\nLinea 1\nLinea 2\n")
    monkeypatch.setattr(qwen_engine, "get_client", lambda: client)
    monkeypatch.setattr(qwen_engine, "_encode_image", lambda image_path, max_size=2048: "encoded")
    times = iter([400.0, 401.0, 403.0])
    monkeypatch.setattr(qwen_engine.time, "time", lambda: next(times))

    result = qwen_engine.extract_text(
        image_path="/tmp/page.png",
        page_number=9,
        fallback_reason="confianza baja",
        tiempo_paddle=1.5,
    )

    assert client.calls
    assert result.engine_used == "qwen"
    assert result.fallback_reason == "confianza baja"
    assert result.text == "Linea 1\nLinea 2"
    assert result.lines == ["Linea 1", "Linea 2"]
    assert result.conf_promedio is None
    assert result.tiempo_paddle == 1.5
    assert result.tiempo_qwen == pytest.approx(1.0)
    assert result.tiempo_total == pytest.approx(3.0)


def test_extract_text_image_read_error_returns_placeholder(monkeypatch):
    def raise_image_error(image_path, max_size=2048):
        raise OSError("bad image")

    monkeypatch.setattr(qwen_engine, "_encode_image", raise_image_error)
    times = iter([500.0] + [500.4] * 50)
    monkeypatch.setattr(qwen_engine.time, "time", lambda: next(times))

    result = qwen_engine.extract_text(
        image_path="/tmp/page.png",
        page_number=2,
        fallback_reason="confianza baja",
    )

    assert result.is_error is True
    assert result.fallback_reason.startswith("qwen_image_read_error:")


def test_extract_text_api_error_returns_placeholder(monkeypatch):
    client = DummyClient(content="", fail=True)
    monkeypatch.setattr(qwen_engine, "get_client", lambda: client)
    monkeypatch.setattr(qwen_engine, "_encode_image", lambda image_path, max_size=2048: "encoded")
    monkeypatch.setattr(qwen_engine.time, "sleep", lambda _: None)
    times = iter([600.0] + [600.8] * 20)
    monkeypatch.setattr(qwen_engine.time, "time", lambda: next(times))

    result = qwen_engine.extract_text(
        image_path="/tmp/page.png",
        page_number=6,
        fallback_reason="descarte alto",
    )

    assert result.is_error is True
    assert result.fallback_reason.startswith("qwen_api_error:")