from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass(frozen=True)
class PaddleOCREngineConfig:
    """Configuración mínima para inicializar PaddleOCR."""

    lang: str = "es"
    use_gpu: bool = True
    kwargs: dict[str, Any] = field(default_factory=dict)


class PaddleOCREngine:
    """Wrapper del motor PaddleOCR.

    La importación de dependencias se hace de forma lazy para evitar errores
    al importar el paquete si PaddleOCR no está instalado todavía.
    """

    def __init__(self, config: PaddleOCREngineConfig | None = None) -> None:
        self.config = config or PaddleOCREngineConfig()
        self._ocr: Any | None = None

    def _lazy_init(self) -> None:
        if self._ocr is not None:
            return

        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "No se pudo importar `paddleocr`. Instala `paddleocr` (y `paddlepaddle-gpu` si aplica)."
            ) from exc

        self._ocr = PaddleOCR(
            lang=self.config.lang,
            use_gpu=self.config.use_gpu,
            **self.config.kwargs,
        )

    def ocr(self, image: Any, *, cls: bool = True) -> Any:
        """Ejecuta OCR y devuelve el resultado 'raw' de PaddleOCR."""

        self._lazy_init()
        assert self._ocr is not None
        return self._ocr.ocr(image, cls=cls)

    def extract_text(self, image: Any, *, cls: bool = True) -> tuple[str, float | None]:
        """Extrae texto plano + confianza promedio (si existe)."""

        raw = self.ocr(image, cls=cls)
        lines: list[str] = []
        confidences: list[float] = []

        for item in _iter_paddle_items(raw):
            text, conf = _parse_paddle_item(item)
            if text:
                lines.append(text)
            if conf is not None:
                confidences.append(conf)

        avg_conf = (sum(confidences) / len(confidences)) if confidences else None
        return "\n".join(lines).strip(), avg_conf


def _iter_paddle_items(raw: Any) -> Iterator[Any]:
    if raw is None:
        return

    if _looks_like_paddle_item(raw):
        yield raw
        return

    if isinstance(raw, (list, tuple)):
        for element in raw:
            yield from _iter_paddle_items(element)


def _looks_like_paddle_item(obj: Any) -> bool:
    if not isinstance(obj, (list, tuple)) or len(obj) < 2:
        return False

    meta = obj[1]
    if not isinstance(meta, (list, tuple)) or len(meta) < 2:
        return False

    return isinstance(meta[0], str)


def _parse_paddle_item(obj: Any) -> tuple[str, float | None]:
    if not _looks_like_paddle_item(obj):
        return "", None

    meta = obj[1]
    text = meta[0] if isinstance(meta[0], str) else str(meta[0])

    conf_raw = meta[1]
    try:
        conf = float(conf_raw) if conf_raw is not None else None
    except (TypeError, ValueError):
        conf = None

    return text, conf
