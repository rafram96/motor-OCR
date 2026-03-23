# run.py
import os
import logging
import sys
import time
import warnings
from contextlib import redirect_stderr

t_start = time.time()

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["GLOG_minloglevel"] = "3"

warnings.filterwarnings("ignore")

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent / "src"))

from config import LOG_LEVEL

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.DEBUG),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from main import process_and_segment

PDF = r"D:\proyectos\motor-OCR\data\profesoinales motlima corpei.pdf"

# Redirige stderr temporalmente para silenciar logs C++ de Paddle/CUDA en desarrollo.
with open(os.devnull, "w") as devnull, redirect_stderr(devnull):
    doc, secciones = process_and_segment(
        pdf_path=PDF,
        pages=list(range(92, 106)),
        keep_images=True,
    )

print("\n" + "="*60)
print(f"RESULTADO: {doc.total_pages} páginas procesadas")
print(f"  Paddle: {doc.pages_paddle} | Qwen: {doc.pages_qwen} | Error: {doc.pages_error}")
print(f"  Confianza promedio: {doc.conf_promedio_documento:.3f}")
print(f"  Tiempo total: {doc.tiempo_total:.1f}s")
print("="*60)

print(f"\nPROFESIONALES DETECTADOS: {len(secciones)}\n")
for sec in secciones:
    print(f"  {sec.section_index:2d}. {sec.cargo:<60} ({sec.total_pages} págs, desde pág {sec.separator_page})")

print(f"\nTiempo real total: {time.time() - t_start:.1f}s")