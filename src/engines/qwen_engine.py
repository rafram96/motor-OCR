import base64
import io
import time
import logging
from typing import Optional

from openai import OpenAI
from PIL import Image

from config import (
    QWEN_MODEL,
    QWEN_OLLAMA_BASE_URL,
    QWEN_OLLAMA_API_KEY,
    QWEN_MAX_TOKENS_OCR,
    QWEN_TIMEOUT,
)
from models.page_result import PageResult

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """Retorna el cliente Ollama singleton."""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=QWEN_OLLAMA_BASE_URL,
            api_key=QWEN_OLLAMA_API_KEY,
            timeout=QWEN_TIMEOUT,
        )
    return _client


def _encode_image(image_path: str, max_size: int = 2048) -> str:
    """Redimensiona la imagen si es muy grande antes de encodear."""
    with Image.open(image_path) as img:
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG", optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_text(
    image_path: str,
    page_number: int,
    fallback_reason: str,
    tiempo_paddle: Optional[float] = None,
) -> PageResult:
    """
    OCR fallback con Qwen-VL. Se llama cuando paddle tiene baja confianza.

    Args:
        image_path:      Ruta a la imagen.
        page_number:     Número de página en el documento.
        fallback_reason: Razón por la que se activó el fallback (para logs y métricas).
        tiempo_paddle:   Tiempo que tardó paddle antes de activar el fallback.

    Returns:
        PageResult con engine_used='qwen'. Las métricas de confianza son None.
    """
    t_start = time.time()
    max_sizes = [2048, 1024, 768]
    max_retries = len(max_sizes)

    for attempt, max_size in enumerate(max_sizes, 1):
        try:
            b64 = _encode_image(image_path, max_size=max_size)
        except Exception as e:
            logger.error(f"Página {page_number}: no se pudo leer imagen — {e}")
            return PageResult.error_placeholder(
                page_number=page_number,
                image_path=image_path,
                reason=f"qwen_image_read_error: {e}",
            )

        try:
            client = get_client()
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
                            {
                                "type": "text",
                                "text": (
                                    "Extrae todo el texto de esta imagen de documento "
                                    "escaneado peruano. Devuelve solo el texto en el "
                                    "orden correcto de lectura, sin comentarios. /no_think"
                                ),
                            },
                        ],
                    }
                ],
                temperature=0,
                max_tokens=QWEN_MAX_TOKENS_OCR,
            )

            raw = response.choices[0].message.content.strip()

            # Limpiar bloque de thinking si Qwen3 lo incluye
            if "</think>" in raw:
                raw = raw.split("</think>")[-1].strip()

            lineas = [l for l in raw.splitlines() if l.strip()]
            texto = "\n".join(lineas)
            elapsed = time.time() - t_start

            logger.debug(
                f"Página {page_number}: qwen OK — "
                f"{len(lineas)} líneas, t={elapsed:.2f}s, razón='{fallback_reason}', "
                f"intento={attempt}/{max_retries}, max_size={max_size}px"
            )

            return PageResult(
                page_number=page_number,
                image_path=image_path,
                engine_used="qwen",
                fallback_reason=fallback_reason,
                text=texto,
                lines=lineas,
                # Métricas paddle no aplican
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
                tiempo_paddle=tiempo_paddle,
                tiempo_qwen=elapsed,
                tiempo_total=elapsed,
            )
        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    f"Página {page_number}: Qwen falló (intento {attempt}/{max_retries}), "
                    f"reintentando con imagen {max_size}px — {e}"
                )
                time.sleep(2)
                continue

            logger.error(f"Página {page_number}: Qwen falló tras {max_retries} intentos — {e}")
            return PageResult.error_placeholder(
                page_number=page_number,
                image_path=image_path,
                reason=f"qwen_api_error: {e}",
            )

    # Salvaguarda defensiva (no debería alcanzarse).
    return PageResult.error_placeholder(
        page_number=page_number,
        image_path=image_path,
        reason="qwen_api_error: unexpected_retry_flow",
    )