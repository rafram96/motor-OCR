# ── Filtro de candidatas ──────────────────────────────────────────────────────
MAX_LINEAS_SEPARADORA  = 5     # páginas con más líneas no son separadoras
MIN_LINEAS_SEPARADORA  = 1     # páginas vacías tampoco

# ── Qwen ─────────────────────────────────────────────────────────────────────
QWEN_MODEL             = "qwen2.5vl:7b"
QWEN_OLLAMA_BASE_URL   = "http://localhost:11434/v1"
QWEN_OLLAMA_API_KEY    = "ollama"
QWEN_MAX_TOKENS        = 128
QWEN_TIMEOUT           = 60.0

# ── Fallback fuzzy ────────────────────────────────────────────────────────────
FUZZY_SCORE_MINIMO     = 75

CARGOS_BASE = [
    "Gerente de Contrato",
    "Jefe de Supervisión",
    "Supervisor de Obra",
    "Especialista en Arquitectura",
    "Especialista en Estructuras",
    "Especialista en Instalaciones Eléctricas",
    "Especialista en Instalaciones Sanitarias",
    "Especialista en Instalaciones Mecánicas",
    "Especialista en Seguridad y Salud",
    "Especialista en Metrados y Costos",
    "Especialista en Calidad",
    "Especialista en BIM",
    "Especialista en Equipamiento",
    # lista viva — ampliar sin tocar código
]

# ── Normalización de errores OCR conocidos ────────────────────────────────────
NORMALIZACIONES = {
    # Supervisión
    "supersion": "Supervisión",
    "supervicion": "Supervisión",
    "supervision": "Supervisión",
    "superv ision": "Supervisión",
    "superv1sion": "Supervisión",

    # Instalaciones
    "instalaclones": "Instalaciones",
    "instaiaciones": "Instalaciones",
    "instalacionese": "Instalaciones",

    # Especialidad sanitaria/mecánica/eléctrica
    "sanltarias": "Sanitarias",
    "sanitarlos": "Sanitarios",
    "mecanicas": "Mecánicas",
    "mecamicas": "Mecánicas",
    "mecanlcas": "Mecánicas",
    "electricas": "Eléctricas",
    "electrlcas": "Eléctricas",
    "electncas": "Eléctricas",

    # Seguridad y salud
    "segundad": "Seguridad",
    "seguri dad": "Seguridad",
    "salu d": "Salud",
    "salod": "Salud",

    # Metrados / costos / valorizaciones
    "metradosv": "Metrados",
    "co stos": "Costos",
    "varolizaciones": "Valorizaciones",
    "valonzaciones": "Valorizaciones",
    "valorizaclones": "Valorizaciones",

    # Otras especialidades frecuentes
    "teconologicas": "Tecnológicas",
    "arqultectura": "Arquitectura",
    "arqultectonico": "Arquitectónico",
    "estructvras": "Estructuras",
    "equipamlento": "Equipamiento",
    "ca1idad": "Calidad",

    # ampliar conforme aparezcan nuevos errores en producción
}