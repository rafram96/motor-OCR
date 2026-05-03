"""Engine table_extract — extraccion de tablas como matrices via PP-Structure V3.

Subprocess-callable via mode 'table_extract' del subprocess_wrapper.py.

Pipeline: PDF -> imagenes (PDF_DPI=300) -> PP-Structure -> matrices [filas][cols].

A diferencia de paddle_engine.py (OCR linea-a-linea), este engine usa el
modulo de table recognition de PP-Structure que devuelve celdas estructuradas
respetando merged cells y bordes.

Output: lista de tablas con matrices, una por tabla detectada en las paginas.
"""

from .pipeline import extract_tables_from_pdf

__all__ = ["extract_tables_from_pdf"]
