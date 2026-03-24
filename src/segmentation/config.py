# ── Filtro de candidatas ──────────────────────────────────────────────────────
MAX_LINEAS_SEPARADORA  = 15     # páginas con más líneas no son separadoras
MIN_LINEAS_SEPARADORA  = 1     # páginas vacías tampoco

# ── Qwen ─────────────────────────────────────────────────────────────────────
QWEN_MODEL             = "qwen2.5vl:7b"
QWEN_OLLAMA_BASE_URL   = "http://localhost:11434/v1"
QWEN_OLLAMA_API_KEY    = "ollama"
QWEN_MAX_TOKENS        = 128
QWEN_TIMEOUT           = 60.0

# ── Fallback fuzzy ────────────────────────────────────────────────────────────
FUZZY_SCORE_MINIMO     = 80

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
    "Especialista en Seguridad Salud y Medio Ambiente",
    "Especialista en Metrados y Costos",
    "Especialista en Metrados Costos y Valorizaciones",
    "Especialista en Calidad",
    "Especialista en Control y Aseguramiento de Calidad",
    "Especialista BIM",
    "Especialista en Equipamiento",
    "Especialista en Equipamiento Hospitalario",
    "Especialista en Implementacion de Soluciones de Tecnologia de la Informacion",
    "Especialista en Configuraciones Tecnologicas de la Informacion y Comunicaciones",
    "Especialista en Instalaciones de Comunicaciones",
    "Especialista TIC",
    "Especialista de Pre Instalación de Equipamiento Hospitalario"
    "Especialista en Supervisión de Estructuras",
    "Especialista en Supervisión de Arquitectura",
    "Especialista en Supervisión del Medio Ambiente",
    "Especialista en Supervisión de Control de Calidad",
    "Supervisión de Coordinador BIM",
    "Especialista en Supervisión de Instalaciones Sanitarias",
    "Especialista en Supervisión de Instalaciones Mecánicas",
    "Especialista en Supervisión de Instalaciones Eléctricas",
    "Especialista en Supervisión de Instalaciones Comunicaciones",
    "Especialista en Supervisión de Geotecnia",
    "Especialista en Supervisión de Equipamiento Hospitalario",
    "Especialista en Supervisión de Costos Metrados y Valorizaciones",
    "Especialista en Supervisión de la Seguridad Salud en el Trabajo",
    "Especialista de Pre Instalación de Equipamiento Hospitalario",
    "Gerente de Supervisión",
]

# ── Lista blanca: palabras clave que DEBEN aparecer en una separadora ─────────
# Si el texto de la página no contiene ninguna de estas → no es separadora.
PATRONES_CARGO = [
    "gerente",
    "jefe",
    "supervisor",      # cubre "Supervisión" y "Supervisor"
    "especialista",
    "coordinador",
    "residente",
    "pre instalacion", # ← agregar
    "preinstalacion",  # ← variante sin espacio
]

# ── Lista negra: frases que nunca son separadoras aunque contengan cargo ──────
FRASES_DESCARTE = [
    "asimismo, manifiesto",
    "me comprometo a prestar",
    "a nombre de la nacion",
    "el rector de la universidad",
    "ha sido incorporado",
    "certificado de trabajo",
    "calificaciones del personal clave",
    "experiencia del personal clave",
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

    "bim": "BIM",
    "b1m": "BIM",
    "implementaclon": "Implementación",
    "implementacion": "Implementación",
    "conflguraciones": "Configuraciones",
    "configuraciones": "Configuraciones",
    "informaclon": "Información",
    "lnformacion": "Información",
    "comunlcaciones": "Comunicaciones",
    "comunlcacion": "Comunicación",
    
    # Control de calidad
    "aseguramien to": "Aseguramiento",
    "aseguramienio": "Aseguramiento",
    "calldad": "Calidad",

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