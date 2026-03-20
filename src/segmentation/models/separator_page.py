from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SeparatorPage:
    """
    Representa una página identificada como separadora de profesional.
    Una separadora es la primera página de la sección de cada profesional
    — contiene únicamente el cargo en grande y poco más.
    """
    page_number: int
    image_path: str
    line_count: int            # líneas no vacías detectadas por paddle
    raw_text: str              # texto extraído por paddle

    # ── Resultado de la evaluación ────────────────────────────────────────────
    es_separadora: bool
    cargo_detectado: str       # cargo tal como lo devolvió Qwen o fuzzy
    cargo_normalizado: str     # cargo después de normalización tipográfica
    confianza_qwen: str        # "alta" | "media" | "baja" | "fuzzy" | "none"
    metodo: str                # "qwen" | "fuzzy_fallback" | "descartada"
    tiempo_deteccion: float