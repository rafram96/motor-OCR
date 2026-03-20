from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict
from .page_result import PageResult


@dataclass
class DocumentResult:
    # ── Identificación ─────────────────────────────────────────────────────────
    pdf_path: str
    total_pages: int
    pages: List[PageResult] = field(default_factory=list)

    # ── Resumen de engines ─────────────────────────────────────────────────────
    pages_paddle: int = 0
    pages_qwen: int = 0
    pages_error: int = 0

    # ── Métricas globales ──────────────────────────────────────────────────────
    conf_promedio_documento: float = 0.0

    # ── Tiempos ────────────────────────────────────────────────────────────────
    tiempo_total: float = 0.0

    # ── Propiedades ────────────────────────────────────────────────────────────

    @property
    def full_text(self) -> str:
        """Texto completo del documento, ordenado por número de página."""
        return "\n\n".join(
            p.text
            for p in sorted(self.pages, key=lambda x: x.page_number)
            if not p.is_error
        )

    @property
    def text_by_page(self) -> Dict[int, str]:
        """Diccionario {numero_pagina: texto}. Incluye páginas error con string vacío."""
        return {p.page_number: p.text for p in self.pages}

    @property
    def error_pages(self) -> List[PageResult]:
        """Páginas que fallaron durante el procesamiento."""
        return [p for p in self.pages if p.is_error]

    @property
    def fallback_pages(self) -> List[PageResult]:
        """Páginas que fueron a Qwen como fallback."""
        return [p for p in self.pages if p.engine_used == "qwen"]

    def compute_summary(self) -> None:
        """
        Calcula y actualiza los campos de resumen.
        Llamar después de que todas las páginas estén añadidas.
        """
        self.pages_paddle = sum(1 for p in self.pages if p.engine_used == "paddle")
        self.pages_qwen   = sum(1 for p in self.pages if p.engine_used == "qwen")
        self.pages_error  = sum(1 for p in self.pages if p.engine_used == "error")

        confianzas = [p.conf_promedio for p in self.pages if p.conf_promedio is not None]
        self.conf_promedio_documento = sum(confianzas) / len(confianzas) if confianzas else 0.0