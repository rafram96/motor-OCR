from __future__ import annotations

from typing import Any

from engines.paddle_engine import PaddleOCREngine
from engines.qwen_engine import QwenVLEngine
from models.page_result import PageResult

from .decision import decide_engine


DEFAULT_QWEN_PROMPT = (
    "Extrae TODO el texto visible en la imagen. Devuelve únicamente texto plano, sin explicaciones."
)


def process_page(
    image: Any,
    *,
    page_number: int,
    paddle: PaddleOCREngine,
    qwen: QwenVLEngine,
    min_confidence: float = 0.75,
    qwen_prompt: str = DEFAULT_QWEN_PROMPT,
) -> PageResult:
    """Procesa una página completa y retorna un `PageResult`."""

    paddle_text, paddle_conf = paddle.extract_text(image)
    decision = decide_engine(paddle_confidence=paddle_conf, min_confidence=min_confidence)

    if decision.use_qwen:
        text = qwen.extract_text(image, prompt=qwen_prompt)
        engine = "qwen"
        confidence = None
    else:
        text = paddle_text
        engine = "paddle"
        confidence = paddle_conf

    metrics = {
        "decision_reason": decision.reason,
        "paddle_confidence": paddle_conf,
    }

    return PageResult(
        page_number=page_number,
        text=text,
        engine=engine,
        confidence=confidence,
        metrics=metrics,
    )
