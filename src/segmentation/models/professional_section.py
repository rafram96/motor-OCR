from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

from models.page_result import PageResult


@dataclass
class ProfessionalSection:
    """
    Agrupa todas las páginas que pertenecen a un profesional.
    La primera página es siempre la separadora (portada con el cargo).
    """
    section_index: int         # orden en el documento (1, 2, 3...)
    cargo: str                 # cargo normalizado
    cargo_raw: str             # cargo exacto antes de normalizar
    separator_page: int        # número de página separadora
    pages: List[PageResult] = field(default_factory=list)
    total_pages: int = 0
    has_tables: bool = False

    @property
    def full_text(self) -> str:
        """Texto completo de la sección, página a página."""
        return "\n\n".join(
            p.text for p in self.pages if not p.is_error and p.text.strip()
        )

    @property
    def page_numbers(self) -> List[int]:
        return [p.page_number for p in self.pages]