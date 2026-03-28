from __future__ import annotations
from collections import defaultdict
import logging
import re
import time
from typing import List, Tuple

from models.document_result import DocumentResult
from models.page_result import PageResult
from segmentation.consolidator import _extraer_numero
from segmentation.detector import es_candidata_separadora, es_delimitador_bloque, evaluar_separadora
from segmentation.models.separator_page import SeparatorPage
from segmentation.models.professional_section import ProfessionalSection

logger = logging.getLogger(__name__)


def _clave_agrupacion(cargo: str) -> str:
    """
    Normaliza el cargo para agrupar bloques del mismo profesional.

    Ejemplos:
    - "Especialista En Estructuras N° 1" -> "especialista en estructuras n°1"
    - "Jefe De Supervisión" -> "jefe de supervisión"
    """
    cargo_lower = cargo.lower().strip()
    return re.sub(r"n[°º]?\s*(\d+)", r"n°\1", cargo_lower)


def consolidar_secciones(secciones: List[ProfessionalSection]) -> List[ProfessionalSection]:
    """
    Consolida bloques repetidos del mismo profesional (cargo + número, si existe).

    Si cada cargo aparece una sola vez, retorna la lista sin cambios efectivos.
    Si hay múltiples bloques del mismo cargo, unifica sus páginas en una sección.
    """
    grupos = defaultdict(list)
    for sec in secciones:
        grupos[_clave_agrupacion(sec.cargo)].append(sec)

    resultado: List[ProfessionalSection] = []
    for bloques in grupos.values():
        if len(bloques) == 1:
            resultado.append(bloques[0])
            continue

        primer_bloque = min(bloques, key=lambda b: b.separator_page)
        todas_las_paginas = sorted(
            [p for b in bloques for p in b.pages],
            key=lambda p: p.page_number,
        )

        consolidada = ProfessionalSection(
            section_index=primer_bloque.section_index,
            cargo=primer_bloque.cargo,
            cargo_raw=primer_bloque.cargo_raw,
            separator_page=primer_bloque.separator_page,
            pages=todas_las_paginas,
            total_pages=len(todas_las_paginas),
            has_tables=any(b.has_tables for b in bloques),
        )
        resultado.append(consolidada)

    return sorted(resultado, key=lambda s: s.separator_page)


def _format_eta(segundos: float) -> str:
    if segundos <= 0:
        return "0s"
    s = int(segundos)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def segment_document(doc: DocumentResult) -> Tuple[List[ProfessionalSection], List[SeparatorPage]]:
    """
    Divide un DocumentResult en secciones por profesional.

    Flujo:
    1. Filtrar páginas candidatas a separadora (pocas líneas)
    2. Confirmar cada candidata con Qwen (+ fallback fuzzy)
    3. Agrupar páginas entre separadoras consecutivas
    4. Retornar lista de ProfessionalSection ordenada por aparición

    Args:
        doc: DocumentResult del motor OCR con todas las páginas.

    Returns:
        (secciones, candidatas_descartadas)
        secciones: Lista de ProfessionalSection.
        candidatas_descartadas: Lista de SeparatorPage con es_separadora=False.
    """
    pages_ord = sorted(doc.pages, key=lambda p: p.page_number)

    # ── 1. Identificar separadoras ────────────────────────────────────────────
    candidatas: List[PageResult] = [
        p for p in pages_ord if es_candidata_separadora(p)
    ]

    logger.info(
        f"Segmentación: {len(pages_ord)} páginas totales, "
        f"{len(candidatas)} candidatas a separadora"
    )

    separadoras: List[SeparatorPage] = []
    descartadas: List[SeparatorPage] = []
    total_candidatas = len(candidatas)
    progreso_cada = max(1, total_candidatas // 10) if total_candidatas else 1
    t_candidatas = time.time()

    for idx, page in enumerate(candidatas, start=1):
        sep = evaluar_separadora(page)
        if sep.es_separadora:
            separadoras.append(sep)
        else:
            descartadas.append(sep)

        if idx == 1 or idx % progreso_cada == 0 or idx == total_candidatas:
            elapsed = time.time() - t_candidatas
            promedio = elapsed / idx
            restante = max(0.0, promedio * (total_candidatas - idx))
            pct = (idx / total_candidatas) * 100 if total_candidatas else 100.0
            logger.info(
                f"Segmentación progreso (confirmación): {idx}/{total_candidatas} "
                f"({pct:.1f}%), ETA {_format_eta(restante)}"
            )

    logger.info(f"Separadoras confirmadas: {len(separadoras)}")

    if not separadoras:
        logger.warning(
            "No se detectaron separadoras — el documento puede no tener "
            "la estructura esperada o todas las candidatas fueron descartadas."
        )
        return [], descartadas

    # ── 2. Agrupar páginas entre separadoras ──────────────────────────────────
    secciones: List[ProfessionalSection] = []

    for i, sep in enumerate(separadoras):
        inicio = sep.page_number
        fin = (
            separadoras[i + 1].page_number - 1
            if i + 1 < len(separadoras)
            else pages_ord[-1].page_number
        )

        paginas_seccion = [
            p for p in pages_ord
            if inicio <= p.page_number <= fin
        ]

        seccion = ProfessionalSection(
            section_index=i + 1,
            cargo=sep.cargo_normalizado,
            cargo_raw=sep.cargo_detectado,
            numero=_extraer_numero(sep.cargo_normalizado),
            separator_page=inicio,
            pages=paginas_seccion,
            total_pages=len(paginas_seccion),
            has_tables=any(p.tiene_tabla for p in paginas_seccion),
        )
        secciones.append(seccion)

        logger.debug(
            f"  Sección {i+1}: '{sep.cargo_normalizado}' "
            f"págs {inicio}–{fin} ({len(paginas_seccion)} páginas) "
            f"[{sep.metodo}]"
        )

    # ── 3. Recortar secciones usando delimitadores de bloque ────────────────────
    # Recopilar puntos de corte de dos fuentes:
    # a) Candidatas descartadas (páginas con pocas líneas que no son separadoras)
    # b) Delimitadores de bloque (ej: "B.2 EXPERIENCIA DEL PERSONAL CLAVE")
    #    que pueden tener muchas líneas de ruido y no pasan el filtro de densidad
    pags_corte: set[int] = set()

    # (a) Descartadas
    for d in descartadas:
        pags_corte.add(d.page_number)

    # (b) Delimitadores de bloque (escanea TODAS las páginas)
    for p in pages_ord:
        if p.page_number not in pags_corte and es_delimitador_bloque(p):
            pags_corte.add(p.page_number)

    if pags_corte:
        pags_corte_ord = sorted(pags_corte)
        logger.info(f"Puntos de corte detectados: {pags_corte_ord}")

        for seccion in secciones:
            if not seccion.pages:
                continue

            pag_inicio = seccion.pages[0].page_number
            pag_fin = seccion.pages[-1].page_number

            # Buscar el primer punto de corte dentro del rango de esta sección
            # (debe ser posterior a la separadora, no la separadora misma)
            corte = None
            for pc in pags_corte_ord:
                if pag_inicio < pc <= pag_fin:
                    corte = pc
                    break

            if corte is not None:
                paginas_antes = [
                    p for p in seccion.pages if p.page_number < corte
                ]
                recortadas = len(seccion.pages) - len(paginas_antes)
                seccion.pages = paginas_antes
                seccion.total_pages = len(paginas_antes)
                seccion.has_tables = any(p.tiene_tabla for p in paginas_antes)

                logger.info(
                    f"  Recorte sección '{seccion.cargo}': "
                    f"pág {corte} es delimitador, "
                    f"eliminadas {recortadas} págs → quedan {seccion.total_pages}"
                )

    logger.info(
        f"Segmentación completada: {len(secciones)} profesionales detectados"
    )
    return secciones, descartadas