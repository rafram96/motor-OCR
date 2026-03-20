import logging
from models.page_result import PageResult

logger = logging.getLogger(__name__)


def process_page(image_path: str, page_number: int) -> PageResult:
    """
    Flujo completo para una sola página:
    1. PaddleOCR extrae texto y métricas
    2. decision.debe_usar_qwen() evalúa si el resultado es confiable
    3. Si no → retorna el resultado de paddle directamente
    4. Si sí → qwen_engine.extract_text() como fallback

    Esta función es la unidad de paralelismo en ProcessPoolExecutor.
    Debe ser serializable: sin lambdas ni closures en el scope global.

    Args:
        image_path:  Ruta absoluta a la imagen PNG.
        page_number: Número de página en el documento (base 1).

    Returns:
        PageResult definitivo — nunca lanza excepción, errores quedan en el result.
    """
    # Imports dentro de la función para compatibilidad con multiprocessing
    # (cada worker tiene su propio estado de módulo)
    from engines import paddle_engine
    from engines import qwen_engine
    from pipeline.decision import debe_usar_qwen

    try:
        # ── Paso 1: paddle ────────────────────────────────────────────────────
        paddle_result = paddle_engine.predict(image_path, page_number)

        # ── Paso 2: decisión ──────────────────────────────────────────────────
        usar_qwen, razon = debe_usar_qwen(paddle_result)

        if not usar_qwen:
            return paddle_result

        # ── Paso 3: fallback qwen ─────────────────────────────────────────────
        logger.info(f"Página {page_number}: fallback → Qwen ({razon})")
        return qwen_engine.extract_text(
            image_path=image_path,
            page_number=page_number,
            fallback_reason=razon,
            tiempo_paddle=paddle_result.tiempo_paddle,
        )

    except Exception as e:
        # Captura de último recurso — no debería llegar aquí porque
        # paddle_engine y qwen_engine ya manejan sus propias excepciones
        logger.error(f"Página {page_number}: error inesperado en process_page — {e}")
        return PageResult.error_placeholder(
            page_number=page_number,
            image_path=image_path,
            reason=f"unexpected: {e}",
        )