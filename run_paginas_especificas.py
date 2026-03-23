import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import LOG_LEVEL

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.DEBUG),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from main import process_document

PDF = r"D:\proyectos\motor-OCR\data\profesoinales motlima corpei.pdf"

# Edita esta lista in-code con las páginas que quieras procesar (base 1).
PAGINAS = [11, 17, 20, 50, 68, 74]


def main() -> None:
    doc = process_document(
        pdf_path=PDF,
        pages=PAGINAS,
        keep_images=True,
    )

    print("\n" + "=" * 70)
    print(f"Páginas solicitadas: {PAGINAS}")
    print(
        f"Procesadas: {doc.total_pages} | "
        f"Paddle: {doc.pages_paddle} | Qwen: {doc.pages_qwen} | Error: {doc.pages_error}"
    )
    print("=" * 70)

    for p in sorted(doc.pages, key=lambda x: x.page_number):
        estado = p.engine_used
        detalle = p.fallback_reason or ""
        line_count = len(p.lines) if p.lines else 0
        print(
            f"Pag {p.page_number:>3}: engine={estado:<6} "
            f"lineas={line_count:<4} "
            f"detalle={detalle}"
        )
        print(f"\n--- Página {p.page_number} ---")
        for i, line in enumerate(p.lines):
            print(f"  [{i}] '{line}'")


if __name__ == "__main__":
    main()
