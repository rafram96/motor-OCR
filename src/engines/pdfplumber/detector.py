from __future__ import annotations
import json
import logging
import time
from typing import Optional

from openai import OpenAI

from models.page_result import PageResult
from segmentation.config import (
    FUZZY_SCORE_DIRECTO,
    FUZZY_SCORE_MINIMO,
    QWEN_OLLAMA_BASE_URL,
    QWEN_OLLAMA_API_KEY,
    QWEN_TIMEOUT,
)
from segmentation.detector import (
    fuzzy_detect_cargo,
    normalizar_cargo,
    _es_frase_descarte,
)
from segmentation.models.separator_page import SeparatorPage

logger = logging.getLogger(__name__)

FUZZY_SCORE_BORDERLINE = 70
LLM_MODEL_TEXT = "qwen2.5:14b"
LLM_MAX_TOKENS = 128
_MAX_CHARS_PROMPT = 2000

_client: Optional[OpenAI] = None


def _get_text_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=QWEN_OLLAMA_BASE_URL,
            api_key=QWEN_OLLAMA_API_KEY,
            timeout=QWEN_TIMEOUT,
        )
    return _client


PROMPT_SEPARADORA_TEXTONLY = """Texto extraido de una pagina de expediente OSCE (licitacion publica peruana):
---
{texto}
---
Una pagina SEPARADORA de profesional es aquella cuyo contenido PRINCIPAL es solo el CARGO del profesional.
Ejemplos validos: "Especialista en Estructuras", "Gerente de Contrato N°1", "Jefe de Supervision", "Especialista BIM N°2".

NO son separadoras:
- Certificados, constancias, diplomas, DNI, carnet del CIP u otro colegio
- Indices o portadas de seccion ("B.1 CALIFICACIONES DEL PERSONAL CLAVE", "B.2 EXPERIENCIA DEL PERSONAL CLAVE")
- Paginas con tablas de experiencia o CVs completos
- Paginas con sellos, firmas o en blanco

Responde SOLO JSON sin explicaciones:
{{"es_separadora": true/false, "cargo": "cargo completo con N° si aparece", "confianza": "alta"|"media"|"baja"}}
/no_think"""


def _llamar_llm_textonly(texto: str) -> tuple[bool, str, str]:
    """Llama qwen2.5:14b con el texto de la pagina. Retorna (es_separadora, cargo, confianza)."""
    try:
        client = _get_text_client()
        response = client.chat.completions.create(
            model=LLM_MODEL_TEXT,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT_SEPARADORA_TEXTONLY.format(
                        texto=texto[:_MAX_CHARS_PROMPT]
                    ),
                }
            ],
            temperature=0,
            max_tokens=LLM_MAX_TOKENS,
        )
    except Exception as e:
        logger.warning(f"qwen2.5:14b texto-only fallo - {e}")
        return False, "", "error"

    raw = response.choices[0].message.content.strip()
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    raw = raw.strip("`").strip()
    if raw.startswith("json"):
        raw = raw[4:].strip()

    try:
        data = json.loads(raw)
        return (
            bool(data.get("es_separadora", False)),
            str(data.get("cargo") or ""),
            str(data.get("confianza", "baja")).lower(),
        )
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"qwen2.5:14b JSON invalido - {e} - raw: {raw!r}")
        return False, "", "error"


def evaluar_separadora_textonly(page: PageResult) -> SeparatorPage:
    """
    Variante de evaluar_separadora sin imagen (solo texto).

    Logica:
    1. Fuzzy directo (score >= 90) -> aceptar, skip LLM
    2. Fuzzy borderline (70-89) -> consultar qwen2.5:14b texto-only
       - Si LLM dice separadora con confianza alta/media -> aceptar
    3. Fuzzy fallback (score >= 80) -> aceptar
    4. Descartar

    Siempre retorna un SeparatorPage (es_separadora puede ser False).
    """
    t_start = time.time()

    enc_fuzzy, cargo_fuzzy, score_fuzzy = fuzzy_detect_cargo(page.text)

    # Paso 1: fuzzy directo
    if (
        enc_fuzzy
        and score_fuzzy >= FUZZY_SCORE_DIRECTO
        and not _es_frase_descarte(cargo_fuzzy)
    ):
        cargo_norm = normalizar_cargo(cargo_fuzzy)
        logger.info(
            f"Pagina {page.page_number}: separadora por fuzzy directo "
            f"(cargo='{cargo_norm}', score={score_fuzzy})"
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

    # Paso 2: LLM texto-only para borderline
    es_sep_llm, cargo_llm, conf_llm = False, "", "none"
    if enc_fuzzy and score_fuzzy >= FUZZY_SCORE_BORDERLINE:
        es_sep_llm, cargo_llm, conf_llm = _llamar_llm_textonly(page.text)
        if (
            es_sep_llm
            and conf_llm in ("alta", "media")
            and cargo_llm.strip()
            and not _es_frase_descarte(cargo_llm)
        ):
            cargo_norm = normalizar_cargo(cargo_llm)
            logger.info(
                f"Pagina {page.page_number}: separadora por qwen texto-only "
                f"(cargo='{cargo_norm}', conf={conf_llm}, fuzzy_score={score_fuzzy})"
            )
            return SeparatorPage(
                page_number=page.page_number,
                image_path=page.image_path,
                line_count=page.line_count,
                raw_text=page.text,
                es_separadora=True,
                cargo_detectado=cargo_llm,
                cargo_normalizado=cargo_norm,
                confianza_qwen=conf_llm,
                metodo="qwen_textonly",
                tiempo_deteccion=time.time() - t_start,
            )

    # Paso 3: fuzzy fallback
    if (
        enc_fuzzy
        and score_fuzzy >= FUZZY_SCORE_MINIMO
        and not _es_frase_descarte(cargo_fuzzy)
    ):
        cargo_norm = normalizar_cargo(cargo_fuzzy)
        logger.info(
            f"Pagina {page.page_number}: separadora por fuzzy fallback "
            f"(cargo='{cargo_norm}', score={score_fuzzy}, llm={conf_llm})"
        )
        return SeparatorPage(
            page_number=page.page_number,
            image_path=page.image_path,
            line_count=page.line_count,
            raw_text=page.text,
            es_separadora=True,
            cargo_detectado=cargo_fuzzy,
            cargo_normalizado=cargo_norm,
            confianza_qwen=conf_llm,
            metodo="fuzzy_fallback",
            tiempo_deteccion=time.time() - t_start,
        )

    # Paso 4: descartar
    logger.debug(
        f"Pagina {page.page_number}: descartada "
        f"(fuzzy_score={score_fuzzy}, llm_es_sep={es_sep_llm}, conf={conf_llm})"
    )
    return SeparatorPage(
        page_number=page.page_number,
        image_path=page.image_path,
        line_count=page.line_count,
        raw_text=page.text,
        es_separadora=False,
        cargo_detectado="",
        cargo_normalizado="",
        confianza_qwen=conf_llm,
        metodo="descartada",
        tiempo_deteccion=time.time() - t_start,
    )
