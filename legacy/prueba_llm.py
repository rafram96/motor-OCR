"""
qwen.py
"""

import time
import base64
from openai import OpenAI

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY  = "ollama"
MODEL           = "qwen2.5vl:7b"
IMAGE_PATH      = r"D:\proyectos\prueba\imagenes_extraidas\pagina_2.png"

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def classify_layout(image_path: str) -> dict:
    print(f"Imagen: {image_path}")
    print(f"Modelo: {MODEL}")
    print("-" * 50)

    b64 = encode_image(image_path)
    t_start = time.time()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}
                    },
                    {
                        "type": "text",
                        "text": (
                            "Analiza el layout de esta imagen de documento escaneado. "
                            "Responde SOLO con JSON, sin explicaciones:\n"
                            "{\n"
                            '  "tiene_tabla": true/false,\n'
                            '  "tipo": "texto_simple" | "tabla" | "mixta",\n'
                            '  "confianza": "alta" | "media" | "baja"\n'
                            "}"
                        )
                    }
                ]
            }
        ],
        temperature=0,
        max_tokens=60,
    )

    t_end = time.time()
    elapsed = t_end - t_start

    raw = response.choices[0].message.content.strip()
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    print(f"Respuesta raw: {raw}")
    print(f"Tiempo total:  {elapsed:.2f}s")

    usage = response.usage
    if usage:
        tps = usage.completion_tokens / elapsed if elapsed > 0 else 0
        print(f"Tokens — prompt: {usage.prompt_tokens} | completion: {usage.completion_tokens} | total: {usage.total_tokens}")
        print(f"Velocidad: {tps:.1f} tokens/seg")

    return {"raw": raw, "tiempo": elapsed}


if __name__ == "__main__":
    classify_layout(IMAGE_PATH)