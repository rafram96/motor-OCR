from __future__ import annotations
import base64
import json
import logging
import time
import unicodedata
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
    PATRONES_CARGO,
    PATRONES_DELIMITADOR,
    FRASES_DESCARTE,
)
from segmentation.models.separator_page import SeparatorPage

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None


def _strip_tildes(texto: str) -> str:
    """Elimina tildes/acentos para comparación robusta."""
    return "".join(
        c for c in unicodedata.normalize("NFD", texto.lower())
        if unicodedata.category(c) != "Mn"
    )


PROMPT_SEPARADORA = """
Analiza esta imagen de un expediente de licitación pública peruana.
Responde SOLO con JSON, sin explicaciones. /no_think
{
    "es_separadora": true/false,
    "cargo": "cargo COMPLETO incluyendo N°1, N°2, etc. si aparece",
    "confianza": "alta" | "media" | "baja"
}
IMPORTANTE: Si el cargo incluye un número como N°1, N°2, N°3, inclúyelo
en el campo cargo. Ejemplo: "Especialista en Supervisión de Estructuras N°2"
Una página separadora ES aquella cuyo contenido PRINCIPAL es el cargo
del profesional. Puede tener ruido adicional como sellos, números de folio,
texto del consorcio o marcas de agua — eso NO la descalifica. Los formatos
válidos son:
- Cargo en grande y centrado (estilo B.1)
- Cargo entre líneas punteadas horizontales, texto más pequeño,
  con número N°1 o N°2 debajo (estilo B.2/B.3)
- El nombre del consorcio arriba y el cargo abajo
- Un sello o firma del representante del consorcio abajo
Cargos válidos típicos: GERENTE DE SUPERVISIÓN, GERENTE DE CONTRATO,
JEFE DE SUPERVISIÓN, SUPERVISOR DE OBRA, ESPECIALISTA EN ESTRUCTURAS,
ESPECIALISTA EN ARQUITECTURA, ESPECIALISTA EN INSTALACIONES SANITARIAS,
ESPECIALISTA EN INSTALACIONES MECÁNICAS, ESPECIALISTA BIM, ESPECIALISTA
EN EQUIPAMIENTO, ESPECIALISTA EN SEGURIDAD Y SALUD, ESPECIALISTA EN
METRADOS Y COSTOS y similares. También incluye cargos con N°1, N°2, N°3.
Una página separadora NO ES:
- Un diploma universitario (tiene escudo, texto "A nombre de la Nación")
- Un certificado del CIP o de otro colegio profesional
- Un Documento de Identidad
- Una constancia o certificado de trabajo
- Una página de índice o portada de sección (ej: "B.1 CALIFICACIONES
  DEL PERSONAL CLAVE", "B.2 EXPERIENCIA DEL PERSONAL CLAVE",
  "B.3 EQUIPAMIENTO ESTRATÉGICO", "CAMIONETAS", "EQUIPO DE TOPOGRAFÍA",
  "DOCUMENTACIÓN DE PRESENTACIÓN FACULTATIVA",
  "EXPERIENCIA EN LA ESPECIALIDAD ADICIONAL DEL PERSONAL CLAVE",
  "SOSTENIBILIDAD AMBIENTAL", "INTEGRIDAD EN LA CONTRATACIÓN",
  "GESTIÓN DE CALIDAD")
- Una página donde el cargo principal es Rector, Decano, Secretario
  General, Director, Representante Legal u otra autoridad institucional
- Una página con solo sellos, firmas o páginas en blanco
- Una página del Anexo N°16 con tablas de experiencia
- Una empresa o consorcio (ej: "CHINA GEZHOU GROUP COMPANY LIMITED")
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
    Las separadoras suelen tener pocas líneas. Las de contenido tienen 20–80.
    Evita mandar páginas de contenido a Qwen innecesariamente.
    """
    lines = [l.strip() for l in page.lines if l.strip()]

    # Ignorar líneas de ruido: números de folio, tokens muy cortos, etc.
    lines_limpias = [
        l for l in lines
        if len(l) > 2 and not l.isdigit()
    ]

    if not lines_limpias:
        return False

    # Texto muy corto o solo ruido numérico → no es separadora
    texto_junto = " ".join(lines_limpias)
    if len(texto_junto.strip()) < 10:
        return False

    texto_norm = _strip_tildes(texto_junto)

    # Lista blanca: si no contiene ningún patrón de cargo → descartar
    if not any(_strip_tildes(p) in texto_norm for p in PATRONES_CARGO):
        logger.info(f"  DESCARTE PATRON pág {page.page_number}: sin patrón de cargo — '{texto_junto[:60]}'")
        return False

    # Lista negra: frases que nunca son separadoras aunque contengan cargo
    if any(_strip_tildes(frase) in texto_norm for frase in FRASES_DESCARTE):
        logger.info(f"  DESCARTE FRASE pág {page.page_number}: {texto_junto[:60]}")
        return False

    chars_unicos = set(texto_norm.replace(" ", ""))
    if len(chars_unicos) <= 2:
        return False

    logger.info(
        f"  CANDIDATA? pág {page.page_number}: "
        f"lines={len(lines)}, limpias={len(lines_limpias)}, "
        f"MAX={MAX_LINEAS_SEPARADORA}, "
        f"pasa={MIN_LINEAS_SEPARADORA <= len(lines_limpias) <= MAX_LINEAS_SEPARADORA}, "
        f"texto='{texto_norm[:60]}'"
    )
    return MIN_LINEAS_SEPARADORA <= len(lines_limpias) <= MAX_LINEAS_SEPARADORA


def es_delimitador_bloque(page: PageResult) -> bool:
    """
    Detecta páginas que son headers de bloque temático (ej: "B.2 EXPERIENCIA
    DEL PERSONAL CLAVE"). Estas páginas no son separadoras de profesional pero
    sí marcan el fin del profesional anterior.

    A diferencia de es_candidata_separadora, no filtra por densidad de líneas
    porque estas páginas suelen tener muchas líneas de ruido (guiones, etc.).
    Solo mira el texto significativo.
    """
    texto_completo = _strip_tildes(" ".join(page.lines))
    if not texto_completo.strip():
        return False

    for patron in PATRONES_DELIMITADOR:
        if _strip_tildes(patron) in texto_completo:
            logger.debug(
                f"  DELIMITADOR pág {page.page_number}: "
                f"matchea '{patron}'"
            )
            return True
    return False


# ── Normalización de cargos ───────────────────────────────────────────────────

def normalizar_cargo(cargo_raw: str) -> str:
    """
    Corrige errores tipográficos de OCR y aplica title-case.
    Opera en lowercase para evitar problemas de case antes del reemplazo.
    """
    texto_norm = cargo_raw.strip().lower()
    for error, correcto in NORMALIZACIONES.items():
        if error in texto_norm:
            texto_norm = texto_norm.replace(error, correcto.lower())
    return texto_norm.title()


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

    lineas = [l.strip() for l in texto.splitlines() if l.strip()]

    # Une todas las líneas útiles para capturar cargos partidos por OCR.
    texto_limpio = " ".join(
        l for l in lineas
        if not l.isdigit()
        and not set(l.replace(" ", "")) <= {"-"}
        and not (len(l) <= 2 and l.isalpha())
    )

    # Generar candidatos: texto completo + líneas individuales + pares consecutivos
    pares = [
        f"{lineas[i]} {lineas[i+1]}"
        for i in range(len(lineas) - 1)
    ]
    # También triples para cargos largos
    triples = [
        f"{lineas[i]} {lineas[i+1]} {lineas[i+2]}"
        for i in range(len(lineas) - 2)
    ]

    candidatos = [texto, texto_limpio] + lineas + pares + triples

    cargos_lower = [c.lower() for c in CARGOS_BASE]

    for candidato in candidatos:
        resultado = rfprocess.extractOne(
            candidato.lower(),
            cargos_lower,
            scorer=fuzz.token_sort_ratio,
        )
        if resultado and resultado[1] >= FUZZY_SCORE_MINIMO:
            idx = cargos_lower.index(resultado[0])
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
        from PIL import Image
        import io

        with Image.open(page.image_path) as img:
            w, h = img.size
            img_small = img.resize((w // 2, h // 2), Image.LANCZOS)
            buf = io.BytesIO()
            img_small.save(buf, format="PNG", optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            logger.debug(
                f"Página {page.page_number}: imagen reducida "
                f"{w}x{h} → {w//2}x{h//2} para Qwen"
            )
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
    - Qwen dice es_separadora=True con confianza "alta" o "media" y cargo claro → aceptar
    - Si Qwen falla / confianza "baja" / sin cargo claro → intentar fuzzy_fallback
      - Fuzzy matchea → aceptar como fuzzy_fallback
      - Fuzzy no matchea → descartar

    Siempre retorna un SeparatorPage (es_separadora puede ser False).
    """
    t_start = time.time()

    # ── Paso 1: Qwen como árbitro principal ─────────────────────────────────
    es_sep, cargo_qwen, confianza = _confirmar_con_qwen(page)

    # ── Qwen confiable ────────────────────────────────────────────────────────
    if es_sep and confianza in ("alta", "media") and cargo_qwen.strip():
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

    # ── Paso 2: Fuzzy como segundo árbitro independiente ─────────────────────
    # Actúa siempre que Qwen no aceptó — sea por error técnico,
    # baja confianza, o decisión consciente de no es separadora.
    encontrado, cargo_fuzzy = fuzzy_detect_cargo(page.text)
    if encontrado:
        cargo_norm = normalizar_cargo(cargo_fuzzy)
        logger.info(
            f"Página {page.page_number}: separadora detectada por fuzzy "
            f"(qwen_conf={confianza}, cargo='{cargo_norm}')"
        )
        return SeparatorPage(
            page_number=page.page_number,
            image_path=page.image_path,
            line_count=page.line_count,
            raw_text=page.text,
            es_separadora=True,
            cargo_detectado=cargo_fuzzy,
            cargo_normalizado=cargo_norm,
            confianza_qwen=confianza,
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