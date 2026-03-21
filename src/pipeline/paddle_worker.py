"""
Se ejecuta como subproceso independiente.
Recibe rutas de imágenes, corre paddle, guarda resultados en JSON y termina.
Al terminar el proceso, el OS libera toda la VRAM automáticamente.
"""
import sys
import json
import os
import pickle
import time
import logging

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

src_path = sys.argv[1]
args_json_path = sys.argv[2]  # ruta a JSON con lista de (image_path, page_number)
output_pkl = sys.argv[3]      # ruta donde guardar los PageResult serializados

sys.path.insert(0, src_path)

from engines.paddle_engine import predict


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _format_eta(segundos: float) -> str:
    if segundos <= 0:
        return "0s"
    s = int(segundos)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"

with open(args_json_path, "r", encoding="utf-8") as f:
    args_list = json.load(f)
results = []
total = len(args_list)
progreso_cada = max(1, total // 10) if total else 1
t_start = time.time()

logger.info(f"Paddle worker iniciado: {total} páginas")

for idx, (path, page_num) in enumerate(args_list, start=1):
    try:
        result = predict(path, page_num)
    except Exception as e:
        from models.page_result import PageResult
        result = PageResult.error_placeholder(page_num, path, str(e))
    results.append(result)

    if idx == 1 or idx % progreso_cada == 0 or idx == total:
        elapsed = time.time() - t_start
        promedio = elapsed / idx
        restante = max(0.0, promedio * (total - idx))
        pct = (idx / total) * 100 if total else 100.0
        logger.info(
            f"Paddle progreso: {idx}/{total} ({pct:.1f}%), ETA {_format_eta(restante)}"
        )

with open(output_pkl, "wb") as f:
    pickle.dump(results, f)

logger.info("Paddle worker finalizado.")