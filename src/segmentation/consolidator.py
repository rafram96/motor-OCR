from __future__ import annotations
import re
import logging
from collections import defaultdict
from typing import List, Optional

from segmentation.models.professional_section import ProfessionalSection, PageRange

logger = logging.getLogger(__name__)


def consolidar_secciones(secciones: List[ProfessionalSection]) -> List[ProfessionalSection]:
    """
    Agrupa bloques del mismo profesional en una sola ProfessionalSection.

    Tipo A (un solo bloque por profesional): retorna la lista sin cambios.
    Tipo B (múltiples bloques por profesional): fusiona páginas en orden,
    conservando los rangos de origen en bloques_origen.

    Args:
        secciones: Lista de secciones detectadas por segment_document().

    Returns:
        Lista consolidada — un elemento por profesional único.
    """
    if not secciones:
        return []

    # ── Agrupar por clave cargo + número ─────────────────────────────────────
    grupos: dict[str, list[ProfessionalSection]] = defaultdict(list)
    for sec in secciones:
        clave = _clave_agrupacion(sec.cargo)
        grupos[clave].append(sec)

    # ── Detectar si es Tipo B ─────────────────────────────────────────────────
    max_bloques = max(len(v) for v in grupos.values())
    es_tipo_b = max_bloques > 1

    if es_tipo_b:
        logger.info(
            f"Documento Tipo B detectado — "
            f"{max_bloques} bloques temáticos por profesional"
        )
    else:
        logger.info("Documento Tipo A — un bloque por profesional")

    # ── Consolidar ────────────────────────────────────────────────────────────
    resultado: List[ProfessionalSection] = []
    nuevo_index = 1

    # Ordenar grupos por primera aparición en el documento
    grupos_ordenados = sorted(
        grupos.items(),
        key=lambda kv: kv[1][0].separator_page,
    )

    for clave, bloques in grupos_ordenados:
        bloques_ord = sorted(bloques, key=lambda s: s.separator_page)

        if len(bloques_ord) == 1:
            # Tipo A o profesional sin repetición — pasar sin cambios
            sec = bloques_ord[0]
            sec.section_index = nuevo_index
            sec.bloques_origen = [
                PageRange(
                    start=sec.pages[0].page_number if sec.pages else sec.separator_page,
                    end=sec.pages[-1].page_number if sec.pages else sec.separator_page,
                    separator_page=sec.separator_page,
                )
            ]
            resultado.append(sec)
        else:
            # Tipo B — fusionar páginas de todos los bloques
            primer = bloques_ord[0]

            todas_las_paginas = sorted(
                [p for b in bloques_ord for p in b.pages],
                key=lambda p: p.page_number,
            )

            bloques_origen = [
                PageRange(
                    start=b.pages[0].page_number if b.pages else b.separator_page,
                    end=b.pages[-1].page_number if b.pages else b.separator_page,
                    separator_page=b.separator_page,
                )
                for b in bloques_ord
            ]

            consolidada = ProfessionalSection(
                section_index=nuevo_index,
                cargo=primer.cargo,
                cargo_raw=primer.cargo_raw,
                numero=_extraer_numero(primer.cargo),
                separator_page=primer.separator_page,
                pages=todas_las_paginas,
                total_pages=len(todas_las_paginas),
                has_tables=any(b.has_tables for b in bloques_ord),
                bloques_origen=bloques_origen,
            )
            resultado.append(consolidada)

            logger.debug(
                f"  Consolidado: '{primer.cargo}' — "
                f"{len(bloques_ord)} bloques → {len(todas_las_paginas)} páginas totales"
            )

        nuevo_index += 1

    logger.info(
        f"Consolidación completada: "
        f"{len(secciones)} bloques → {len(resultado)} profesionales"
    )
    return resultado


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clave_agrupacion(cargo: str) -> str:
    """
    Normaliza el cargo para usarlo como clave de agrupación.
    'Especialista En Estructuras N° 1' → 'especialista en estructuras n°1'
    'Jefe De Supervisión'              → 'jefe de supervisión'
    """
    cargo_lower = cargo.lower().strip()
    # Normalizar variaciones de número: "n° 1", "n°1", "nº 1", "n 1" → "n°1"
    cargo_lower = re.sub(r'n[°º]?\s*(\d+)', r'n°\1', cargo_lower)
    return cargo_lower


def _extraer_numero(cargo: str) -> Optional[str]:
    """
    Extrae el número del cargo si existe.
    'Especialista En Estructuras N° 1' → '1'
    'Jefe De Supervisión'              → None
    """
    match = re.search(r'n[°º]?\s*(\d+)', cargo.lower())
    return match.group(1) if match else None