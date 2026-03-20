import os

# ── Entorno ────────────────────────────────────────────────────────────────────
# Debe setearse ANTES de cualquier import de paddle
os.environ["FLAGS_use_mkldnn"] = "0"
POPPLER_PATH = r"C:\poppler\Library\bin"

# ── PaddleOCR ──────────────────────────────────────────────────────────────────
PADDLE_LANG                 = "es"
PADDLE_USE_TEXTLINE_ORIENT  = True
PADDLE_USE_DOC_ORIENT       = True
PADDLE_USE_DOC_UNWARPING    = True
PADDLE_REC_SCORE_THRESH     = 0.5
PADDLE_DET_THRESH           = 0.3
PADDLE_DET_BOX_THRESH       = 0.3

# ── Umbrales de decisión ───────────────────────────────────────────────────────
UMBRAL_CONFIANZA_PROMEDIO   = 0.85   # conf_promedio < esto → fallback qwen
UMBRAL_TASA_DESCARTE        = 0.20   # tasa_descarte > esto → fallback qwen
UMBRAL_CONFIANZA_LINEA      = 0.85   # para marcar líneas individuales como bajas

# ── Qwen-VL (Ollama) ───────────────────────────────────────────────────────────
QWEN_MODEL                  = "qwen2.5vl:7b"
QWEN_OLLAMA_BASE_URL        = "http://localhost:11434/v1"
QWEN_OLLAMA_API_KEY         = "ollama"
QWEN_MAX_TOKENS_OCR         = 2048
QWEN_TIMEOUT                = 120.0

# ── PDF a imágenes ─────────────────────────────────────────────────────────────
PDF_DPI                     = 300
PDF_IMAGE_FORMAT            = "PNG"

# ── Paralelismo ────────────────────────────────────────────────────────────────
# CPU servidor (i9): 3-4 | GPU servidor (RTX 5000): 2
MAX_WORKERS                 = 4

# ── Salida ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR                  = r"D:\proyectos\infoobras\ocr_output"
SAVE_MARKDOWN               = True