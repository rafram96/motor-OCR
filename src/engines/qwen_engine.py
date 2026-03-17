from __future__ import annotations

import base64
import io
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QwenVLEngineConfig:
    """Configuración para el wrapper Qwen-VL a través de Ollama."""

    model: str = "qwen2.5vl:7b"
    host: str = "http://localhost:11434"
    request_timeout_s: float = 180.0


class QwenVLEngine:
    """Wrapper de un modelo de visión (Qwen-VL) servido por Ollama.

    Implementa una llamada mínima al endpoint nativo de Ollama `/api/generate`
    para evitar depender de librerías externas.
    """

    def __init__(self, config: QwenVLEngineConfig | None = None) -> None:
        self.config = config or QwenVLEngineConfig()

    def extract_text(self, image: Any, *, prompt: str | None = None) -> str:
        """Extrae texto visible en una imagen usando Qwen-VL vía Ollama.

        Args:
            image: `bytes` (imagen codificada), `str|Path` (ruta) o `PIL.Image.Image`.
            prompt: Prompt opcional. Por defecto pide texto plano.
        """

        image_bytes = _to_image_bytes(image)
        prompt = (
            prompt
            or "Extrae TODO el texto visible en la imagen. Devuelve únicamente texto plano, sin explicaciones."
        )

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "images": [base64.b64encode(image_bytes).decode("ascii")],
            "stream": False,
        }

        url = f"{self.config.host.rstrip('/')}/api/generate"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.config.request_timeout_s) as resp:
                body = resp.read().decode("utf-8")
        except Exception as exc:
            raise RuntimeError(f"Error llamando a Ollama en {url}: {exc}") from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            snippet = body[:2000]
            raise RuntimeError(f"Respuesta no-JSON desde Ollama (primeros 2k): {snippet}") from exc

        response_text = data.get("response")
        if not isinstance(response_text, str):
            raise RuntimeError(f"Respuesta inesperada de Ollama: {data}")

        return response_text.strip()


def _to_image_bytes(image: Any) -> bytes:
    if isinstance(image, bytes):
        return image

    if isinstance(image, (str, Path)):
        return Path(image).read_bytes()

    try:
        from PIL import Image  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Para pasar una imagen en memoria (PIL.Image.Image), instala `Pillow`."
        ) from exc

    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    raise TypeError(
        "Tipo de imagen no soportado. Usa bytes, ruta (str/Path) o PIL.Image.Image."
    )
