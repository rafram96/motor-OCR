"""
Se ejecuta como subproceso independiente.
Recibe rutas de imágenes, corre paddle, guarda resultados en JSON y termina.
Al terminar el proceso, el OS libera toda la VRAM automáticamente.
"""
import sys
import json
import os
import pickle

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

src_path = sys.argv[1]
args_json = sys.argv[2]   # JSON con lista de (image_path, page_number)
output_pkl = sys.argv[3]  # ruta donde guardar los PageResult serializados

sys.path.insert(0, src_path)

from engines.paddle_engine import predict

args_list = json.loads(args_json)
results = []
for path, page_num in args_list:
    try:
        result = predict(path, page_num)
    except Exception as e:
        from models.page_result import PageResult
        result = PageResult.error_placeholder(page_num, path, str(e))
    results.append(result)

with open(output_pkl, "wb") as f:
    pickle.dump(results, f)