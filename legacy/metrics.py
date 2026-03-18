"""
metrics.py
"""
import os
import numpy as np
from paddleocr import PaddleOCR
from datetime import datetime

# Configurar entorno
os.environ["FLAGS_use_mkldnn"] = "0"

# Inicializar PaddleOCR
ocr = PaddleOCR(
    use_textline_orientation=True,
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    lang='es',
    text_rec_score_thresh=0.5,
    text_det_thresh=0.3,
    text_det_box_thresh=0.3,
)

# Configuración
ruta_imagen    = r"D:\proyectos\prueba\imagenes_extraidas\pagina_3.png"
umbral         = 0.85
nombre_archivo = os.path.splitext(os.path.basename(ruta_imagen))[0]
fecha_actual   = datetime.now().strftime("%Y%m%d_%H%M%S")

# Procesar imagen
resultado = ocr.predict(ruta_imagen)
res = resultado[0]

# ── Extraer datos ──────────────────────────────────────────────
angle        = res["doc_preprocessor_res"]["angle"]
det_count    = len(res["dt_polys"])
rec_count    = len(res["rec_texts"])
descartadas  = det_count - rec_count
tasa_desc    = descartadas / det_count if det_count > 0 else 0

textos       = res["rec_texts"]
scores       = res["rec_scores"]
polys        = res["rec_polys"]
boxes        = res["rec_boxes"]
angles_line  = res["textline_orientation_angles"]
det_polys    = res["dt_polys"]
model_sets   = res["model_settings"]
det_params   = res["text_det_params"]
text_type    = res["text_type"]

conf_array        = np.array(scores)
conf_promedio     = float(np.mean(conf_array))
conf_mediana      = float(np.median(conf_array))
conf_min          = float(np.min(conf_array))
conf_max          = float(np.max(conf_array))
conf_std          = float(np.std(conf_array))
cuartiles         = np.percentile(conf_array, [25, 50, 75])
lineas_bajas      = sum(1 for s in scores if s < umbral)

bins = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]

if conf_promedio < umbral or tasa_desc > 0.20:
    recomendacion = "🔴 Enviar a Qwen-VL"
elif lineas_bajas > 0:
    recomendacion = f"🟡 Evaluar — {lineas_bajas} líneas con baja confianza"
else:
    recomendacion = "🟢 Usar texto de PaddleOCR directamente"

# ── Print consola ──────────────────────────────────────────────
print(f"Rotación detectada:       {angle}°")
print(f"Regiones detectadas:      {det_count}")
print(f"Regiones reconocidas:     {rec_count}")
print(f"Descartadas:              {descartadas} ({tasa_desc*100:.1f}%)")
print(f"Confianza promedio:       {conf_promedio:.4f}")
print(f"Líneas baja confianza:    {lineas_bajas}")
print(f"Recomendación:            {recomendacion}")

# ── Generar Markdown ───────────────────────────────────────────
directorio_salida = os.path.join(os.path.dirname(ruta_imagen), "resultados_ocr")
os.makedirs(directorio_salida, exist_ok=True)
archivo_md = os.path.join(directorio_salida, f"{nombre_archivo}_full_{fecha_actual}.md")

with open(archivo_md, "w", encoding="utf-8") as f:

    f.write(f"# Análisis OCR completo — {nombre_archivo}\n\n")
    f.write(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
    f.write(f"**Imagen:** `{ruta_imagen}`\n\n")

    # ── 1. Configuración del pipeline
    f.write("## 1. Configuración del pipeline\n\n")
    f.write(f"| Parámetro | Valor |\n|-----------|-------|\n")
    f.write(f"| use_doc_orientation_classify | {model_sets.get('use_doc_orientation_classify')} |\n")
    f.write(f"| use_doc_unwarping | {model_sets.get('use_doc_unwarping')} |\n")
    f.write(f"| use_textline_orientation | {model_sets.get('use_textline_orientation', 'N/A')} |\n")
    f.write(f"| text_type | {text_type} |\n")
    f.write(f"| text_rec_score_thresh | {res['text_rec_score_thresh']} |\n\n")

    f.write("### Parámetros de detección\n\n")
    f.write(f"| Parámetro | Valor |\n|-----------|-------|\n")
    for k, v in det_params.items():
        f.write(f"| {k} | {v} |\n")
    f.write("\n")

    # ── 2. Preprocesamiento
    f.write("## 2. Preprocesamiento del documento\n\n")
    f.write(f"| Campo | Valor |\n|-------|-------|\n")
    f.write(f"| Ángulo de rotación detectado | {angle}° |\n")
    f.write(f"| Corrección aplicada | {'Sí' if angle != 0 else 'No'} |\n\n")

    # ── 3. Detección vs reconocimiento
    f.write("## 3. Detección vs reconocimiento\n\n")
    f.write(f"| Métrica | Valor |\n|---------|-------|\n")
    f.write(f"| Regiones detectadas (dt_polys) | {det_count} |\n")
    f.write(f"| Regiones reconocidas (rec_texts) | {rec_count} |\n")
    f.write(f"| Descartadas por bajo score | {descartadas} ({tasa_desc*100:.1f}%) |\n\n")

    # ── 4. Estadísticas de confianza
    f.write("## 4. Estadísticas de confianza\n\n")
    f.write(f"| Métrica | Valor |\n|---------|-------|\n")
    f.write(f"| Promedio | {conf_promedio:.4f} |\n")
    f.write(f"| Mediana | {conf_mediana:.4f} |\n")
    f.write(f"| Mínimo | {conf_min:.4f} |\n")
    f.write(f"| Máximo | {conf_max:.4f} |\n")
    f.write(f"| Desviación estándar | {conf_std:.4f} |\n")
    f.write(f"| Q1 (25%) | {cuartiles[0]:.4f} |\n")
    f.write(f"| Q2 (50%) | {cuartiles[1]:.4f} |\n")
    f.write(f"| Q3 (75%) | {cuartiles[2]:.4f} |\n")
    f.write(f"| Líneas bajo umbral ({umbral}) | {lineas_bajas} ({lineas_bajas/rec_count*100:.1f}%) |\n\n")

    f.write("### Distribución de confianza\n\n")
    f.write("| Rango | Líneas | Porcentaje |\n|-------|--------|------------|\n")
    for i in range(len(bins) - 1):
        count = int(np.sum((conf_array >= bins[i]) & (conf_array < bins[i+1])))
        if count > 0:
            f.write(f"| {bins[i]:.2f}–{bins[i+1]:.2f} | {count} | {count/rec_count*100:.1f}% |\n")
    f.write("\n")

    # ── 5. Orientación de líneas
    f.write("## 5. Orientación de líneas detectadas\n\n")
    angulos_unicos = {}
    for a in angles_line:
        angulos_unicos[a] = angulos_unicos.get(a, 0) + 1
    f.write("| Ángulo | Cantidad |\n|--------|----------|\n")
    for a, cnt in sorted(angulos_unicos.items()):
        f.write(f"| {a}° | {cnt} |\n")
    f.write("\n")

    # ── 6. Texto reconocido completo
    f.write("## 6. Texto reconocido\n\n")
    f.write("```\n")
    for t in textos:
        f.write(f"{t}\n")
    f.write("```\n\n")

    # ── 7. Detalle línea por línea
    f.write("## 7. Detalle línea por línea\n\n")
    f.write("| # | Score | Estado | Ángulo | Texto |\n")
    f.write("|---|-------|--------|--------|-------|\n")
    for i, (texto, score) in enumerate(zip(textos, scores)):
        estado = "✓" if score >= umbral else "⚠️"
        ang    = angles_line[i] if i < len(angles_line) else "N/A"
        f.write(f"| {i+1} | {score:.4f} | {estado} | {ang}° | {texto} |\n")
    f.write("\n")

    # ── 8. Bounding boxes
    f.write("## 8. Bounding boxes (rec_boxes)\n\n")
    f.write("| # | x1 | y1 | x2 | y2 | Texto |\n")
    f.write("|---|----|----|----|----|-------|\n")
    for i, (box, texto) in enumerate(zip(boxes, textos)):
        b = box.flatten()
        f.write(f"| {i+1} | {b[0]} | {b[1]} | {b[2]} | {b[3]} | {texto} |\n")
    f.write("\n")

    # ── 9. Polígonos de reconocimiento
    f.write("## 9. Polígonos de reconocimiento (rec_polys)\n\n")
    f.write("| # | Puntos | Texto |\n|---|--------|-------|\n")
    for i, (poly, texto) in enumerate(zip(polys, textos)):
        pts = " → ".join([f"({p[0]},{p[1]})" for p in poly])
        f.write(f"| {i+1} | {pts} | {texto} |\n")
    f.write("\n")

    # ── 10. Polígonos de detección
    f.write("## 10. Polígonos de detección (dt_polys) — incluye descartadas\n\n")
    f.write("| # | Puntos |\n|---|--------|\n")
    for i, poly in enumerate(det_polys):
        pts = " → ".join([f"({p[0]},{p[1]})" for p in poly])
        f.write(f"| {i+1} | {pts} |\n")
    f.write("\n")

    # ── 11. Recomendación final
    f.write("## 11. Recomendación\n\n")
    f.write(f"**{recomendacion}**\n\n")
    f.write(f"| Criterio | Valor | Umbral | ¿Supera? |\n")
    f.write(f"|----------|-------|--------|----------|\n")
    f.write(f"| Confianza promedio | {conf_promedio:.4f} | {umbral} | {'No ⚠️' if conf_promedio < umbral else 'Sí ✓'} |\n")
    f.write(f"| Tasa de descarte | {tasa_desc*100:.1f}% | 20% | {'No ⚠️' if tasa_desc > 0.20 else 'Sí ✓'} |\n")

    if lineas_bajas > 0:
        f.write(f"\n### Líneas con baja confianza\n\n")
        f.write("| # | Score | Texto |\n|---|-------|-------|\n")
        for i, (texto, score) in enumerate(zip(textos, scores)):
            if score < umbral:
                f.write(f"| {i+1} | {score:.4f} | {texto} |\n")

print(f"\nGuardado en: {archivo_md}")