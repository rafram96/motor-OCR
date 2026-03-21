from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PageResult:
    # ── Identificación ─────────────────────────────────────────────────────────
    page_number: int
    image_path: str

    # ── Motor y texto ──────────────────────────────────────────────────────────
    engine_used: str                    # "paddle" | "qwen" | "error"
    fallback_reason: Optional[str]      # razón del fallback o del error
    text: str                           # texto completo de la página
    lines: List[str]                    # líneas individuales (rec_texts filtradas)

    # ── Métricas paddle ────────────────────────────────────────────────────────
    # Son None cuando engine_used == "qwen" o "error"
    conf_promedio: Optional[float]
    conf_mediana: Optional[float]
    conf_min: Optional[float]
    conf_max: Optional[float]
    conf_std: Optional[float]
    lineas_baja_confianza: int          # líneas con score < UMBRAL_CONFIANZA_LINEA

    # ── Detección vs reconocimiento ────────────────────────────────────────────
    det_count: int                      # regiones detectadas (dt_polys)
    rec_count: int                      # regiones reconocidas (rec_texts)
    tasa_descarte: float                # (det - rec) / det, 0.0 si det == 0

    # ── Preprocesamiento ───────────────────────────────────────────────────────
    angle_detected: int                 # ángulo corregido: 0, 90, 180, 270

    # ── Layout ─────────────────────────────────────────────────────────────────
    tiene_tabla: bool                   # detectado por qwen (False por defecto)

    # ── Tiempos ────────────────────────────────────────────────────────────────
    tiempo_paddle: Optional[float]
    tiempo_qwen: Optional[float]
    tiempo_total: float

    # ── Detalle por línea (para reportes) ─────────────────────────────────────
    # Scores alineados con `lines` (mismo índice). Vacío para qwen/error.
    line_scores: List[float] = field(default_factory=list)

    @classmethod
    def error_placeholder(cls, page_number: int, image_path: str, reason: str) -> "PageResult":
        """
        Crea un PageResult vacío para páginas que fallaron.
        Permite que el DocumentResult tenga siempre N páginas aunque alguna crashee.
        """
        return cls(
            page_number=page_number,
            image_path=image_path,
            engine_used="error",
            fallback_reason=reason,
            text="",
            lines=[],
            conf_promedio=None,
            conf_mediana=None,
            conf_min=None,
            conf_max=None,
            conf_std=None,
            lineas_baja_confianza=0,
            det_count=0,
            rec_count=0,
            tasa_descarte=0.0,
            angle_detected=0,
            tiene_tabla=False,
            tiempo_paddle=None,
            tiempo_qwen=None,
            tiempo_total=0.0,
            line_scores=[],
        )

    @property
    def is_error(self) -> bool:
        return self.engine_used == "error"

    @property
    def line_count(self) -> int:
        """Número de líneas no vacías. Usado por el segmentador."""
        return len([l for l in self.lines if l.strip()])