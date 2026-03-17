from __future__ import annotations

from dataclasses import dataclass

from .page_result import PageResult


@dataclass(frozen=True)
class DocumentResult:
    """Resultado OCR por documento."""

    pages: list[PageResult]
    source: str | None = None

    @property
    def text(self) -> str:
        return "\n\n".join((p.text or "").strip() for p in self.pages).strip()
