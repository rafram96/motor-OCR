import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class _DummyPaddleOCR:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def predict(self, image_path):
        return []


class _DummyOpenAI:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        message = types.SimpleNamespace(content=str(kwargs.get("content", "")))
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])


def _install_stub(module_name, **attributes):
    module = types.ModuleType(module_name)
    for name, value in attributes.items():
        setattr(module, name, value)
    sys.modules[module_name] = module


_install_stub("paddleocr", PaddleOCR=_DummyPaddleOCR)
_install_stub("openai", OpenAI=_DummyOpenAI)
_install_stub("pdf2image", convert_from_path=lambda *args, **kwargs: [])


@pytest.fixture(autouse=True)
def reset_singletons():
    import engines.paddle_engine as paddle_engine
    import engines.qwen_engine as qwen_engine

    paddle_engine._ocr_instance = None
    qwen_engine._client = None
    yield
    paddle_engine._ocr_instance = None
    qwen_engine._client = None