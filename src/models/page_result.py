from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PageResult:
    """Resultado OCR por página."""

    page_number: int
    text: str
    engine: str
    confidence: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
