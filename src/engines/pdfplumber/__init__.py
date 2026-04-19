"""Engine pdfplumber — extraccion + segmentacion para PDFs digitales.

Fast-path que se salta PaddleOCR/Qwen-VL cuando el PDF ya tiene capa de texto.
Reutiliza constantes, fuzzy matching y consolidacion del motor-OCR existente.
Solo reemplaza el input (pdfplumber en vez de PaddleOCR+Qwen) y el arbitro de
separadoras (qwen2.5:14b texto-only en vez de qwen2.5vl:7b sobre imagen).
"""

from .pipeline import process_with_pdfplumber

__all__ = ["process_with_pdfplumber"]
