# motor-OCR

## Que es este repo
Motor de extraccion y segmentacion de documentos PDF para el ecosistema InfoObras. Se invoca como subprocess desde Alpamayo-InfoObras (no expone API REST).

Entrega archivos Markdown estructurados que el backend de Alpamayo consume sin tocar este repo.

## Cajas negras frente al resto del sistema
- Alpamayo NO importa Python de aqui — solo invoca `subprocess_wrapper.py` pasandole un JSON de args.
- Las dependencias de ML (PaddleOCR, Qwen-VL via Ollama) son fragiles. Cambios en el pipeline OCR pueden romper el venv completo.
- Aditivos seguros: agregar un engine nuevo junto a los existentes en `src/engines/` sin tocar PaddleOCR ni Qwen.

## Wrapper: modes disponibles

`subprocess_wrapper.py` recibe `{args_file, results_file}` y despacha segun `mode` del args JSON:

| mode | Entry point (`src/main.py`) | Proposito | Output en results JSON |
|------|------|-----------|------------------------|
| `ocr_only` | `process_document` | PaddleOCR + fallback Qwen-VL. Solo texto, sin segmentar. | `{total_pages, pages_paddle, pages_qwen, pages_pdfplumber, pages_error, conf_promedio_documento, tiempo_total, full_text, engine}` |
| `segmentation` | `process_and_segment` | OCR completo + segmentacion por profesional. | `{mode, doc{...engine:"motor_ocr"}, secciones[...]}` |
| `pdfplumber_segmentation` | `engines.pdfplumber.process_with_pdfplumber` | Fast-path para PDFs digitales: pdfplumber en vez de PaddleOCR/Qwen-VL + fuzzy + qwen2.5:14b texto-only. | Mismo schema que `segmentation` pero con `doc.engine: "pdfplumber"`. |

El mode default si no se pasa es `segmentation`.

## Engines disponibles

`src/engines/` contiene los adaptadores:
- `paddle_engine.py` — PaddleOCR (default para la mayoria de paginas).
- `qwen_engine.py` — Qwen-VL via Ollama (`qwen2.5vl:7b`, fallback cuando Paddle tiene baja confianza).
- `pdfplumber/` — **nuevo**. Engine para PDFs con capa de texto nativa (digitales). No requiere GPU, no hace OCR.

### engines/pdfplumber/

Fast-path que se salta PaddleOCR + Qwen-VL cuando el PDF ya tiene texto extraible. Reutiliza toda la logica de segmentacion/consolidacion que ya existe (`segmentation/consolidator.py`, `segmentation/detector.py::fuzzy_detect_cargo`, `segmentation/output/consolidation_writer.py`).

- `reader.py::read_pdf_pages` — extrae texto por pagina, construye `List[PageResult]` con `engine_used="pdfplumber"`.
- `detector.py::evaluar_separadora_textonly` — fuzzy directo (≥90) → qwen2.5:14b texto-only (70-89) → fuzzy fallback (≥80). El LLM texto-only sustituye al Qwen-VL visual que usa el flujo normal, ya que pdfplumber no tiene imagen.
- `segmenter.py::segment_textonly` — clon funcional de `segment_document` pero llamando al detector textonly.
- `markdown_writer.py::write_document_report_simple` — genera `*_metricas_*.md` y `*_texto_*.md` simplificados (sin metricas de OCR que no aplican).
- `pipeline.py::process_with_pdfplumber` — orquesta todo, genera los mismos 3 `.md` que el flujo normal.

**La decision de usar este engine se toma en Alpamayo**, no aqui. Alpamayo muestrea las primeras 5 paginas con pdfplumber y si chars/pag >= umbral (default 200) invoca `mode=pdfplumber_segmentation`.

## Archivos `.md` generados

Por cada PDF procesado motor-OCR escribe en `{output_dir}/{pdf_stem}/`:
- `{nombre}_metricas_{ts}.md` — metricas globales y por pagina
- `{nombre}_texto_{ts}.md` — texto pagina a pagina
- `{nombre}_segmentacion_{ts}.md` — debug de separadores detectados (solo motor-OCR completo)
- `{nombre}_profesionales_{ts}.md` — secciones consolidadas por profesional (**input principal** para Alpamayo)

El formato del `*_profesionales_*.md` es el contrato con `Alpamayo/src/extraction/md_parser.py`. No romper sin sincronizar ambos repos.

## Constantes compartidas

`src/segmentation/config.py`:
- `CARGOS_BASE` — 54 cargos OSCE tipicos.
- `NORMALIZACIONES` — dict de errores OCR comunes (tildes, letras confusas).
- `PATRONES_CARGO`, `FRASES_DESCARTE`, `PATRONES_DELIMITADOR` — filtros de separadoras.
- `FUZZY_SCORE_DIRECTO=90`, `FUZZY_SCORE_MINIMO=80` — thresholds de RapidFuzz.

El engine `pdfplumber` reutiliza estas constantes directamente (no las duplica).

## Que NO hacer
- No importar PaddleOCR/Qwen en `engines/pdfplumber/` — debe quedar GPU-free.
- No cambiar firmas de `process_document` ni `process_and_segment` — Alpamayo depende del schema actual.
- No cambiar los templates de `*_profesionales_*.md` sin coordinar con Alpamayo.
- No tocar el venv de PaddleOCR — dependencias muy fragiles.

## Config global (`src/config.py`)
- `OUTPUT_DIR = r"D:\proyectos\infoobras\ocr_output"` — donde se guardan los `.md` por default.
- `SAVE_MARKDOWN = True` — si es False, motor-OCR retorna DocumentResult sin escribir archivos.
- `QWEN_MODEL = "qwen2.5vl:7b"` — para segmentador visual.
- El engine pdfplumber usa tambien `qwen2.5:14b` (texto-only) pero esta hardcoded en `engines/pdfplumber/detector.py::LLM_MODEL_TEXT`.

## Requirements clave
```
paddleocr==3.4.0 + paddlepaddle-gpu==3.2.0   ← PaddleOCR
openai                                         ← cliente Ollama
rapidfuzz                                      ← fuzzy de cargos
pdf2image + pypdfium2 + pillow                 ← PDF → imagenes
pdfplumber                                     ← engine pdfplumber (nuevo)
```
