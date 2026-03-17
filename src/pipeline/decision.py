from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineDecision:
    """Decisión simple de enrutamiento para una página."""

    use_qwen: bool
    reason: str


def decide_engine(*, paddle_confidence: float | None, min_confidence: float = 0.75) -> EngineDecision:
    """Decide si usar Qwen-VL como fallback.

    Regla mínima:
    - Si no hay confianza: fallback.
    - Si la confianza es menor al umbral: fallback.
    """

    if paddle_confidence is None:
        return EngineDecision(use_qwen=True, reason="PaddleOCR no devolvió confianza")

    if paddle_confidence < min_confidence:
        return EngineDecision(
            use_qwen=True,
            reason=f"Confianza baja: {paddle_confidence:.3f} < {min_confidence:.3f}",
        )

    return EngineDecision(
        use_qwen=False,
        reason=f"Confianza ok: {paddle_confidence:.3f} >= {min_confidence:.3f}",
    )
