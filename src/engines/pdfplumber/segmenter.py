from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from typing import List, Tuple

from models.document_result import DocumentResult
from models.page_result import PageResult
from segmentation.consolidator import _extraer_numero
from segmentation.detector import es_candidata_separadora, es_delimitador_bloque
from segmentation.models.separator_page import SeparatorPage
from segmentation.models.professional_section import ProfessionalSection

from .detector import evaluar_separadora_textonly

logger = logging.getLogger(__name__)

SEG_WORKERS = 3


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


def segment_textonly(
    doc: DocumentResult,
) -> Tuple[List[ProfessionalSection], List[SeparatorPage], List[int]]:
    """
    Segmentacion para DocumentResult generado por pdfplumber (sin imagen).
    Clon funcional de segment_document pero usando evaluar_separadora_textonly.

    Returns:
        (secciones, candidatas_descartadas, delimitadores)
    """
    pages_ord = sorted(doc.pages, key=lambda p: p.page_number)

    candidatas: List[PageResult] = [
        p for p in pages_ord if es_candidata_separadora(p)
    ]

    logger.info(
        f"Segmentacion (textonly): {len(pages_ord)} paginas totales, "
        f"{len(candidatas)} candidatas a separadora"
    )

    resultados_sep: List[SeparatorPage] = []
    total_candidatas = len(candidatas)
    t_candidatas = time.time()

    with ThreadPoolExecutor(max_workers=SEG_WORKERS) as pool:
        futuros = {
            pool.submit(evaluar_separadora_textonly, page): page.page_number
            for page in candidatas
        }
        done_count = 0
        progreso_cada = max(1, total_candidatas // 10) if total_candidatas else 1

        for futuro in as_completed(futuros):
            done_count += 1
            try:
                sep = futuro.result()
            except Exception as e:
                pn = futuros[futuro]
                logger.error(f"Error evaluando candidata pag {pn}: {e}")
                continue
            resultados_sep.append(sep)

            if (
                done_count == 1
                or done_count % progreso_cada == 0
                or done_count == total_candidatas
            ):
                elapsed = time.time() - t_candidatas
                promedio = elapsed / done_count
                restante = max(0.0, promedio * (total_candidatas - done_count))
                pct = (done_count / total_candidatas) * 100 if total_candidatas else 100.0
                logger.info(
                    f"Segmentacion progreso (textonly): {done_count}/{total_candidatas} "
                    f"({pct:.1f}%), ETA {_format_eta(restante)}"
                )

    separadoras: List[SeparatorPage] = []
    descartadas: List[SeparatorPage] = []
    for sep in sorted(resultados_sep, key=lambda s: s.page_number):
        if sep.es_separadora:
            separadoras.append(sep)
        else:
            descartadas.append(sep)

    logger.info(f"Separadoras confirmadas: {len(separadoras)}")

    if not separadoras:
        logger.warning(
            "No se detectaron separadoras - el fast-path retornara lista vacia, "
            "Alpamayo deberia hacer fallback a motor-OCR completo."
        )
        return [], descartadas, []

    # ── Agrupar paginas entre separadoras ────────────────────────────────────
    secciones: List[ProfessionalSection] = []

    for i, sep in enumerate(separadoras):
        inicio = sep.page_number
        fin = (
            separadoras[i + 1].page_number - 1
            if i + 1 < len(separadoras)
            else pages_ord[-1].page_number
        )

        paginas_seccion = [
            p for p in pages_ord if inicio <= p.page_number <= fin
        ]

        secciones.append(
            ProfessionalSection(
                section_index=i + 1,
                cargo=sep.cargo_normalizado,
                cargo_raw=sep.cargo_detectado,
                numero=_extraer_numero(sep.cargo_normalizado),
                separator_page=inicio,
                pages=paginas_seccion,
                total_pages=len(paginas_seccion),
                has_tables=any(p.tiene_tabla for p in paginas_seccion),
            )
        )

    # ── Recorte con descartadas + delimitadores tematicos ────────────────────
    pags_corte: set[int] = set()
    for d in descartadas:
        pags_corte.add(d.page_number)

    delimitadores_tematicos: List[int] = []
    for p in pages_ord:
        if p.page_number not in pags_corte and es_delimitador_bloque(p):
            pags_corte.add(p.page_number)
            delimitadores_tematicos.append(p.page_number)
    delimitadores_tematicos.sort()

    if pags_corte:
        pags_corte_ord = sorted(pags_corte)
        logger.info(f"Puntos de corte detectados: {pags_corte_ord}")

        for seccion in secciones:
            if not seccion.pages:
                continue

            pag_inicio = seccion.pages[0].page_number
            pag_fin = seccion.pages[-1].page_number

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
                    f"Recorte seccion '{seccion.cargo}': "
                    f"pag {corte} es delimitador, "
                    f"eliminadas {recortadas} pags -> quedan {seccion.total_pages}"
                )

    logger.info(
        f"Segmentacion textonly completada: {len(secciones)} profesionales detectados"
    )
    return secciones, descartadas, delimitadores_tematicos
