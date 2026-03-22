from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from models.page_result import PageResult


@dataclass
class PageRange:
    """Rango de páginas de un bloque temático original."""
    start: int
    end: int
    separator_page: int

    def __str__(self) -> str:
        return f"{self.start}–{self.end}"


@dataclass
class ProfessionalSection:
    """
    Agrupa todas las páginas que pertenecen a un profesional.
    Puede contener páginas de múltiples bloques temáticos (Tipo B)
    o de un solo bloque (Tipo A).
    """
    section_index: int              # orden en el resultado final (1, 2, 3...)
    cargo: str                      # cargo normalizado
    cargo_raw: str                  # cargo exacto antes de normalizar
    numero: Optional[str]           # "1", "2", None — para cargos con N°
    separator_page: int             # página separadora del primer bloque
    pages: List[PageResult] = field(default_factory=list)
    total_pages: int = 0
    has_tables: bool = False
    bloques_origen: List[PageRange] = field(default_factory=list)  # rangos por bloque

    @property
    def full_text(self) -> str:
        """Texto completo de la sección, página a página."""
        return "\n\n".join(
            p.text for p in self.pages if not p.is_error and p.text.strip()
        )

    @property
    def page_numbers(self) -> List[int]:
        return [p.page_number for p in self.pages]

    @property
    def es_tipo_b(self) -> bool:
        """True si el profesional tiene múltiples bloques temáticos."""
        return len(self.bloques_origen) > 1