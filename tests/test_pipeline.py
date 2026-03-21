import sys
import json
import pickle
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import engines.paddle_engine as paddle_engine
import engines.qwen_engine as qwen_engine
import main as main_module
import output.markdown_writer as markdown_writer_module
import pipeline.decision as decision_module
import pipeline.page_processor as page_processor_module
import pipeline.pdf_to_images as pdf_to_images_module
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


class DummyImage:
    def __init__(self, label):
        self.label = label

    def save(self, path, image_format):
        Path(path).write_text(self.label, encoding="utf-8")


class FakeFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class FakeExecutor:
    def __init__(self, max_workers=None, initializer=None):
        self.max_workers = max_workers
        self.initializer = initializer

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, args):
        return FakeFuture(fn(args))


def test_pdf_to_images_specific_pages(monkeypatch, tmp_path):
    pdf_file = tmp_path / "entrada.pdf"
    pdf_file.write_bytes(b"pdf")
    calls = []

    def fake_convert_from_path(pdf_path, dpi, first_page=None, last_page=None, **kwargs):
        calls.append((pdf_path, dpi, first_page, last_page))
        return [DummyImage(f"page-{first_page}")]

    monkeypatch.setattr(pdf_to_images_module, "convert_from_path", fake_convert_from_path)

    paths = pdf_to_images_module.pdf_to_images(
        str(pdf_file),
        str(tmp_path / "salida"),
        dpi=123,
        pages=[3, 1],
    )

    assert calls == [
        (str(pdf_file), 123, 1, 1),
        (str(pdf_file), 123, 3, 3),
    ]
    assert [Path(path).name for path in paths] == ["pagina_0001.png", "pagina_0003.png"]
    assert all(Path(path).exists() for path in paths)


def test_pdf_to_images_all_pages(monkeypatch, tmp_path):
    pdf_file = tmp_path / "entrada.pdf"
    pdf_file.write_bytes(b"pdf")
    calls = []

    def fake_convert_from_path(pdf_path, dpi, **kwargs):
        calls.append((pdf_path, dpi))
        return [DummyImage("one"), DummyImage("two")]

    monkeypatch.setattr(pdf_to_images_module, "convert_from_path", fake_convert_from_path)

    paths = pdf_to_images_module.pdf_to_images(
        str(pdf_file),
        str(tmp_path / "salida"),
        dpi=200,
        pages=None,
    )

    assert calls == [(str(pdf_file), 200)]
    assert [Path(path).name for path in paths] == ["pagina_0001.png", "pagina_0002.png"]


def test_pdf_to_images_missing_pdf_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        pdf_to_images_module.pdf_to_images(
            str(tmp_path / "no_existe.pdf"),
            str(tmp_path / "salida"),
        )


def test_process_page_returns_paddle_when_confident(monkeypatch):
    page = make_page(1, text="uno", lines=["uno"], conf_promedio=0.95)
    monkeypatch.setattr(paddle_engine, "predict", lambda image_path, page_number: page)
    monkeypatch.setattr(decision_module, "debe_usar_qwen", lambda page_result: (False, ""))

    qwen_calls = []

    def fail_qwen(*args, **kwargs):
        qwen_calls.append((args, kwargs))
        raise AssertionError("qwen should not be called")

    monkeypatch.setattr(qwen_engine, "extract_text", fail_qwen)

    result = page_processor_module.process_page("/tmp/page.png", 1)

    assert result is page
    assert qwen_calls == []


def test_process_page_uses_qwen_when_needed(monkeypatch):
    paddle_result = make_page(2, text="dos", lines=["dos"], conf_promedio=0.7, tiempo_paddle=1.23)
    qwen_result = make_page(
        2,
        engine_used="qwen",
        fallback_reason="confianza baja",
        text="dos mejorado",
        lines=["dos mejorado"],
        conf_promedio=None,
        conf_mediana=None,
        conf_min=None,
        conf_max=None,
        conf_std=None,
        tiempo_paddle=1.23,
        tiempo_qwen=1.8,
        tiempo_total=3.0,
    )

    monkeypatch.setattr(paddle_engine, "predict", lambda image_path, page_number: paddle_result)
    monkeypatch.setattr(decision_module, "debe_usar_qwen", lambda page_result: (True, "confianza baja"))

    captured = {}

    def fake_extract_text(*, image_path, page_number, fallback_reason, tiempo_paddle=None):
        captured["image_path"] = image_path
        captured["page_number"] = page_number
        captured["fallback_reason"] = fallback_reason
        captured["tiempo_paddle"] = tiempo_paddle
        return qwen_result

    monkeypatch.setattr(qwen_engine, "extract_text", fake_extract_text)

    result = page_processor_module.process_page("/tmp/page.png", 2)

    assert result is qwen_result
    assert captured == {
        "image_path": "/tmp/page.png",
        "page_number": 2,
        "fallback_reason": "confianza baja",
        "tiempo_paddle": 1.23,
    }


def test_process_page_returns_error_on_unexpected_exception(monkeypatch):
    def fail_predict(image_path, page_number):
        raise RuntimeError("boom")

    monkeypatch.setattr(paddle_engine, "predict", fail_predict)

    result = page_processor_module.process_page("/tmp/page.png", 3)

    assert result.is_error is True
    assert result.fallback_reason.startswith("unexpected:")


def test_process_document_orchestrates_workers_and_summary(monkeypatch, tmp_path):
    pdf_file = tmp_path / "entrada.pdf"
    pdf_file.write_bytes(b"pdf")

    page_1 = make_page(1, text="uno", lines=["uno"], conf_promedio=0.9, tiempo_total=1.0)
    page_2_paddle = make_page(
        2,
        text="dos paddle",
        lines=["dos paddle"],
        conf_promedio=0.7,
        tasa_descarte=0.6,
        tiempo_total=1.0,
    )
    page_2_qwen = make_page(
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
        tiempo_qwen=1.4,
        tiempo_total=2.2,
    )

    def fake_pdf_to_images(pdf_path, work_dir, pages=None):
        pages_dir = Path(work_dir) / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        page_paths = [
            pages_dir / "pagina_0002.png",
            pages_dir / "pagina_0001.png",
        ]
        for path in page_paths:
            path.write_text("image", encoding="utf-8")
        return [str(path) for path in page_paths]

    monkeypatch.setattr(main_module, "pdf_to_images", fake_pdf_to_images)

    def fake_subprocess_run(cmd, check):
        args_json = cmd[3]
        output_pkl = cmd[4]
        parsed = json.loads(args_json)
        results = []
        for _, page_num in parsed:
            results.append({1: page_1, 2: page_2_paddle}[page_num])
        with open(output_pkl, "wb") as f:
            pickle.dump(results, f)

        class _Done:
            returncode = 0

        return _Done()

    monkeypatch.setattr(main_module.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(decision_module, "debe_usar_qwen", lambda p: (p.page_number == 2, "confianza baja" if p.page_number == 2 else ""))
    monkeypatch.setattr(qwen_engine, "extract_text", lambda **kwargs: page_2_qwen)
    monkeypatch.setattr(main_module, "SAVE_MARKDOWN", True)
    monkeypatch.setattr(main_module, "MAX_WORKERS", 2)

    write_calls = []

    def fake_write_document_report(doc, work_dir):
        write_calls.append((doc, work_dir))
        return ("metricas.md", "texto.md")

    monkeypatch.setattr(markdown_writer_module, "write_document_report", fake_write_document_report)

    removed_paths = []

    def fake_rmtree(path):
        removed_paths.append(path)

    monkeypatch.setattr(main_module.shutil, "rmtree", fake_rmtree)

    doc = main_module.process_document(
        str(pdf_file),
        output_dir=str(tmp_path / "salida"),
        keep_images=False,
    )

    assert doc.total_pages == 2
    assert [page.page_number for page in doc.pages] == [1, 2]
    assert doc.pages_paddle == 1
    assert doc.pages_qwen == 1
    assert doc.pages_error == 0
    assert doc.conf_promedio_documento == pytest.approx(0.9)
    assert write_calls and write_calls[0][1] == str(Path(tmp_path / "salida") / "entrada")
    assert removed_paths == [str(Path(tmp_path / "salida") / "entrada" / "pages")]