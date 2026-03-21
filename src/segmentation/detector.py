from __future__ import annotations
import base64
import json
import logging
import time
from typing import Optional

from openai import OpenAI

from models.page_result import PageResult
from segmentation.config import (
    MAX_LINEAS_SEPARADORA,
    MIN_LINEAS_SEPARADORA,
    QWEN_MODEL,
    QWEN_OLLAMA_BASE_URL,
    QWEN_OLLAMA_API_KEY,
    QWEN_MAX_TOKENS,
    QWEN_TIMEOUT,
    FUZZY_SCORE_MINIMO,
    CARGOS_BASE,
    NORMALIZACIONES,
)
from segmentation.models.separator_page import SeparatorPage

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None

PROMPT_SEPARADORA = """
Analiza esta imagen de un expediente de licitación pública peruana.
Responde SOLO con JSON, sin explicaciones. /no_think
{
    "es_separadora": true/false,
    "cargo": "cargo exacto o null",
    "confianza": "alta" | "media" | "baja"
}

Una página separadora ES aquella cuyo contenido principal es el cargo
del profesional propuesto por el postor. Puede tener:
- Solo el cargo en grande y centrado (ej: ESPECIALISTA EN ESTRUCTURAS)
- El nombre del consorcio arriba y el cargo abajo
- El cargo entre líneas decorativas con un número
- Un sello o firma del representante del consorcio abajo

Cargos válidos típicos: GERENTE DE CONTRATO, JEFE DE SUPERVISIÓN,
SUPERVISOR DE OBRA, ESPECIALISTA EN ESTRUCTURAS, ESPECIALISTA EN
ARQUITECTURA, ESPECIALISTA EN INSTALACIONES SANITARIAS, ESPECIALISTA
EN INSTALACIONES MECÁNICAS, ESPECIALISTA BIM, ESPECIALISTA EN
EQUIPAMIENTO, ESPECIALISTA EN SEGURIDAD Y SALUD, ESPECIALISTA EN
METRADOS Y COSTOS, ESPECIALISTA EN INSTALACIONES SANITARIAS y similares.

Una página separadora NO ES:
- Un diploma universitario (tiene escudo, texto "A nombre de la Nación")
- Un certificado del CIP o de otro colegio profesional
- Un Documento de Identidad
- Una constancia o certificado de trabajo
- Una página donde el cargo principal es Rector, Decano, Secretario
    General, Director, Representante Legal u otra autoridad institucional
- Una página con solo sellos, firmas o páginas en blanco
- Una página del Anexo N°16 con tablas de experiencia
"""


# ── Singleton cliente Qwen ────────────────────────────────────────────────────

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=QWEN_OLLAMA_BASE_URL,
            api_key=QWEN_OLLAMA_API_KEY,
            timeout=QWEN_TIMEOUT,
        )
    return _client


# ── Filtro por densidad ───────────────────────────────────────────────────────

def es_candidata_separadora(page: PageResult) -> bool:
    """
    Pre-filtro rápido basado en número de líneas.
    Las separadoras tienen 1–6 líneas. Las de contenido tienen 20–80.
    Evita mandar páginas de contenido a Qwen innecesariamente.
    """
    lines = [l for l in page.lines if l.strip()]
    if not lines:
        return False

    # Descartar páginas con solo caracteres repetidos (!!!!!, -----, etc.)
    texto_limpio = " ".join(lines)
    chars_unicos = set(texto_limpio.replace(" ", ""))
    if len(chars_unicos) <= 2:
        return False

    return MIN_LINEAS_SEPARADORA <= len(lines) <= MAX_LINEAS_SEPARADORA


# ── Normalización de cargos ───────────────────────────────────────────────────

def normalizar_cargo(cargo_raw: str) -> str:
    """
    Corrige errores tipográficos de OCR y aplica title-case.
    Opera en lowercase para evitar problemas de case antes del reemplazo.
    """
    texto_lower = cargo_raw.strip().lower()
    for error, correcto in NORMALIZACIONES.items():
        if error in texto_lower:
            texto_lower = texto_lower.replace(error, correcto.lower())
    return texto_lower.title()


# ── Fallback fuzzy ────────────────────────────────────────────────────────────

def fuzzy_detect_cargo(texto: str) -> tuple[bool, str]:
    """
    Busca el cargo más similar en CARGOS_BASE usando fuzzy matching.
    Retorna (encontrado, cargo_normalizado).
    """
    try:
        from rapidfuzz import fuzz, process as rfprocess
    except ImportError:
        logger.warning("rapidfuzz no instalado — fallback fuzzy deshabilitado")
        return False, ""

    resultado = rfprocess.extractOne(
        texto.lower(),
        [c.lower() for c in CARGOS_BASE],
        scorer=fuzz.partial_ratio,
    )
    if resultado and resultado[1] >= FUZZY_SCORE_MINIMO:
        # Retornar el cargo original de la lista (con acentos correctos)
        idx = [c.lower() for c in CARGOS_BASE].index(resultado[0])
        return True, CARGOS_BASE[idx]
    return False, ""


# ── Confirmación con Qwen ─────────────────────────────────────────────────────

def _confirmar_con_qwen(page: PageResult) -> tuple[bool, str, str]:
    """
    Llama a Qwen-VL para confirmar si la página es separadora.
    Retorna (es_separadora, cargo, confianza).
    confianza puede ser "alta", "media", "baja".
    En caso de error retorna (False, "", "error").
    """
    try:
        with open(page.image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Página {page.page_number}: no se pudo leer imagen — {e}")
        return False, "", "error"

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": PROMPT_SEPARADORA},
                    ],
                }
            ],
            temperature=0,
            max_tokens=QWEN_MAX_TOKENS,
        )
    except Exception as e:
        logger.warning(f"Página {page.page_number}: Qwen falló — {e}")
        return False, "", "error"

    raw = response.choices[0].message.content.strip()
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    # Limpiar posibles bloques markdown
    raw = raw.strip("`").strip()
    if raw.startswith("json"):
        raw = raw[4:].strip()

    try:
        data = json.loads(raw)
        es_sep    = bool(data.get("es_separadora", False))
        cargo     = str(data.get("cargo") or "")
        confianza = str(data.get("confianza", "baja")).lower()
        return es_sep, cargo, confianza
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Página {page.page_number}: Qwen devolvió JSON inválido — {e} — raw: {raw!r}")
        return False, "", "error"


# ── Evaluación principal ──────────────────────────────────────────────────────

def evaluar_separadora(page: PageResult) -> SeparatorPage:
    """
    Evalúa si una página candidata es separadora.

    Lógica de aceptación:
    - Fuzzy matchea cargo conocido en texto OCR → aceptar como fuzzy_directo
    - Si fuzzy no matchea:
      - Qwen dice es_separadora=True con confianza "alta" o "media" → aceptar
      - Qwen falla / confianza "baja" / error → intentar fuzzy_fallback
      - Fuzzy no matchea → descartar

    Siempre retorna un SeparatorPage (es_separadora puede ser False).
    """
    t_start = time.time()

    # ── Pre-check: fuzzy directo sobre texto OCR ────────────────────────────
    encontrado, cargo_fuzzy = fuzzy_detect_cargo(page.text)
    if encontrado:
        cargo_norm = normalizar_cargo(cargo_fuzzy)
        logger.info(
            f"Página {page.page_number}: separadora detectada por fuzzy directo "
            f"(cargo='{cargo_norm}')"
        )
        return SeparatorPage(
            page_number=page.page_number,
            image_path=page.image_path,
            line_count=page.line_count,
            raw_text=page.text,
            es_separadora=True,
            cargo_detectado=cargo_fuzzy,
            cargo_normalizado=cargo_norm,
            confianza_qwen="fuzzy",
            metodo="fuzzy_directo",
            tiempo_deteccion=time.time() - t_start,
        )

    # ── Si fuzzy no matchea, confirmar con Qwen ─────────────────────────────
    es_sep, cargo_qwen, confianza = _confirmar_con_qwen(page)

    # ── Qwen confiable ────────────────────────────────────────────────────────
    if es_sep and confianza in ("alta", "media"):
        cargo_norm = normalizar_cargo(cargo_qwen)
        logger.info(
            f"Página {page.page_number}: separadora detectada por Qwen "
            f"(cargo='{cargo_norm}', conf={confianza})"
        )
        return SeparatorPage(
            page_number=page.page_number,
            image_path=page.image_path,
            line_count=page.line_count,
            raw_text=page.text,
            es_separadora=True,
            cargo_detectado=cargo_qwen,
            cargo_normalizado=cargo_norm,
            confianza_qwen=confianza,
            metodo="qwen",
            tiempo_deteccion=time.time() - t_start,
        )

    # ── Qwen no confiable o falló → fuzzy fallback ───────────────────────────
    encontrado, cargo_fuzzy = fuzzy_detect_cargo(page.text)
    if encontrado:
        cargo_norm = normalizar_cargo(cargo_fuzzy)
        logger.info(
            f"Página {page.page_number}: separadora detectada por fuzzy "
            f"(cargo='{cargo_norm}', qwen_conf={confianza})"
        )
        return SeparatorPage(
            page_number=page.page_number,
            image_path=page.image_path,
            line_count=page.line_count,
            raw_text=page.text,
            es_separadora=True,
            cargo_detectado=cargo_fuzzy,
            cargo_normalizado=cargo_norm,
            confianza_qwen="fuzzy",
            metodo="fuzzy_fallback",
            tiempo_deteccion=time.time() - t_start,
        )

    # ── Descartada ────────────────────────────────────────────────────────────
    logger.debug(
        f"Página {page.page_number}: descartada "
        f"(qwen_es_sep={es_sep}, conf={confianza}, fuzzy=False)"
    )
    return SeparatorPage(
        page_number=page.page_number,
        image_path=page.image_path,
        line_count=page.line_count,
        raw_text=page.text,
        es_separadora=False,
        cargo_detectado="",
        cargo_normalizado="",
        confianza_qwen=confianza,
        metodo="descartada",
        tiempo_deteccion=time.time() - t_start,
    )