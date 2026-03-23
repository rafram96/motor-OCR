# run.py
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import LOG_LEVEL

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.DEBUG),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

from main import process_and_segment

PDF = r"D:\proyectos\motor-OCR\data\Profesionales.pdf"

doc, secciones = process_and_segment(
    pdf_path=PDF,
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
    print(f"  {sec.section_index:2d}. {sec.cargo:<50} ({sec.total_pages} págs, desde pág {sec.separator_page})")