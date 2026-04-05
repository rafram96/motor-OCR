from __future__ import annotations
import re
import logging
from collections import defaultdict
from typing import List, Optional

from segmentation.models.professional_section import ProfessionalSection, PageRange

logger = logging.getLogger(__name__)


def _asignar_numeros_implicitos(
    secciones: List[ProfessionalSection],
    delimitadores: List[int],
) -> None:
    """
    Asigna N° implícito a secciones sin número explícito dentro de cada
    bloque temático (entre delimitadores consecutivos).

    Si "Especialista Estructuras" aparece 2 veces en B.1 sin N°,
    les asigna N°1 y N°2 respectivamente. Esto permite que la
    consolidación las empareje correctamente con B.2's ocurrencias.

    Solo modifica secciones cuyo cargo aparece más de una vez dentro
    del mismo bloque temático y que no tienen N° explícito.
    """
    if not delimitadores:
        return

    # Crear límites de bloques temáticos: [(inicio, fin), ...]
    limites: List[tuple[int, int]] = []
    for i, d in enumerate(delimitadores):
        fin = delimitadores[i + 1] - 1 if i + 1 < len(delimitadores) else 999999
        limites.append((d, fin))

    for inicio_bloque, fin_bloque in limites:
        # Secciones dentro de este bloque temático
        secs_en_bloque = [
            s for s in secciones
            if inicio_bloque <= s.separator_page <= fin_bloque
        ]

        # Contar apariciones de cada cargo base (sin N°)
        conteo_cargo: dict[str, list[ProfessionalSection]] = defaultdict(list)
        for sec in secs_en_bloque:
            cargo_base = _clave_agrupacion(sec.cargo)
            # Si ya tiene N° explícito, no tocar
            if _extraer_numero(sec.cargo) is not None:
                continue
            conteo_cargo[cargo_base].append(sec)

        # Asignar N° implícito solo a cargos que aparecen más de 1 vez
        for cargo_base, secs in conteo_cargo.items():
            if len(secs) <= 1:
                continue
            # Ordenar por página separadora para asignar en orden
            secs_ord = sorted(secs, key=lambda s: s.separator_page)
            for idx, sec in enumerate(secs_ord, start=1):
                sec.cargo = f"{sec.cargo} N°{idx}"
                sec.numero = str(idx)
                logger.debug(
                    f"  N° implícito: '{sec.cargo}' (pág {sec.separator_page})"
                )


def consolidar_secciones(
    secciones: List[ProfessionalSection],
    delimitadores: Optional[List[int]] = None,
) -> List[ProfessionalSection]:
    """
    Agrupa bloques del mismo profesional en una sola ProfessionalSection.

    Tipo A (un solo bloque por profesional): retorna la lista sin cambios.
    Tipo B (múltiples bloques por profesional): fusiona páginas en orden,
    conservando los rangos de origen en bloques_origen.

    Args:
        secciones: Lista de secciones detectadas por segment_document().
        delimitadores: Páginas de delimitadores temáticos (B.1, B.2, etc.)
                       Usado para asignar N° implícito a cargos repetidos.

    Returns:
        Lista consolidada — un elemento por profesional único.
    """
    if not secciones:
        return []

    # ── Asignar N° implícito a cargos repetidos sin número explícito ─────────
    if delimitadores:
        _asignar_numeros_implicitos(secciones, delimitadores)

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