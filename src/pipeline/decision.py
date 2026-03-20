from config import UMBRAL_CONFIANZA_PROMEDIO, UMBRAL_TASA_DESCARTE
from models.page_result import PageResult


def debe_usar_qwen(page_result: PageResult) -> tuple[bool, str]:
    """
    Decide si una página necesita ser reprocesada con Qwen-VL.

    Evalúa dos criterios independientes — cualquiera activa el fallback:
    - Confianza promedio de paddle por debajo del umbral
    - Tasa de descarte (regiones detectadas pero no reconocidas) por encima del umbral

    Args:
        page_result: Resultado de paddle para la página.

    Returns:
        (usar_qwen: bool, razon: str)
        razon es string vacío si no se necesita fallback.
    """
    # Página que ya falló en paddle — no reintentar con qwen, ya tiene placeholder
    if page_result.is_error:
        return False, ""

    # Confianza promedio baja
    if page_result.conf_promedio is not None:
        if page_result.conf_promedio < UMBRAL_CONFIANZA_PROMEDIO:
            return True, (
                f"confianza promedio baja "
                f"({page_result.conf_promedio:.3f} < {UMBRAL_CONFIANZA_PROMEDIO})"
            )
    else:
        # conf_promedio es None → página en blanco o ilegible
        # No mandamos a Qwen páginas en blanco — no vale la pena
        return False, ""

    # Tasa de descarte alta
    if page_result.tasa_descarte > UMBRAL_TASA_DESCARTE:
        return True, (
            f"tasa de descarte alta "
            f"({page_result.tasa_descarte*100:.1f}% > {UMBRAL_TASA_DESCARTE*100:.0f}%)"
        )

    return False, ""