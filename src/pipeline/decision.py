from config import UMBRAL_CONFIANZA_PROMEDIO, UMBRAL_TASA_DESCARTE
from models.page_result import PageResult
from segmentation.config import MAX_LINEAS_SEPARADORA


def debe_usar_qwen(page_result: PageResult) -> tuple[bool, str]:
    """
    Decide si una página necesita ser reprocesada con Qwen-VL.

    Evalúa dos criterios y exige ambos para activar fallback:
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

    # Página en blanco o ilegible — no vale la pena mandar a Qwen
    if page_result.conf_promedio is None:
        return False, ""

    # Candidata a separadora: el segmentador la evaluará con Qwen visual.
    if page_result.line_count <= MAX_LINEAS_SEPARADORA:
        return False, ""

    conf_baja = page_result.conf_promedio < UMBRAL_CONFIANZA_PROMEDIO
    descarte_alto = page_result.tasa_descarte > UMBRAL_TASA_DESCARTE

    # Requiere AMBAS condiciones simultáneas
    if conf_baja and descarte_alto:
        return True, (
            f"confianza baja ({page_result.conf_promedio:.3f}) "
            f"y descarte alto ({page_result.tasa_descarte*100:.1f}%)"
        )

    return False, ""