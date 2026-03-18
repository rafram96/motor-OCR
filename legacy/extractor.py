"""
extractor.py
"""

import os
from pdf2image import convert_from_path

def extraer_paginas_pdf(pdf_path, output_dir, paginas=None, dpi=300):
    """
    Extrae páginas específicas de un PDF como imágenes
    
    Args:
        pdf_path: Ruta al archivo PDF
        output_dir: Directorio donde guardar las imágenes
        paginas: Lista de números de página a extraer (empieza en 1)
                Si es None, extrae todas las páginas
        dpi: Resolución de las imágenes (300 es buena calidad)
    """
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📄 PDF: {pdf_path}")
    
    if paginas:
        print(f"📑 Páginas a extraer: {paginas}")
        
        for pagina in paginas:
            try:
                # Extraer solo la página específica
                imagenes = convert_from_path(
                    pdf_path,
                    dpi=dpi,
                    first_page=pagina,
                    last_page=pagina
                )
                
                if imagenes:
                    # Guardar la imagen
                    nombre_imagen = f"pagina_{pagina}.png"
                    ruta_imagen = os.path.join(output_dir, nombre_imagen)
                    imagenes[0].save(ruta_imagen, "PNG")
                    print(f"  ✓ Página {pagina} guardada: {nombre_imagen}")
                else:
                    print(f"  ✗ No se pudo extraer página {pagina}")
                    
            except Exception as e:
                print(f"  ✗ Error en página {pagina}: {e}")
    else:
        # Extraer todas las páginas
        print("📑 Extrayendo todas las páginas...")
        imagenes = convert_from_path(pdf_path, dpi=dpi)
        
        for i, imagen in enumerate(imagenes, 1):
            nombre_imagen = f"pagina_{i}.png"
            ruta_imagen = os.path.join(output_dir, nombre_imagen)
            imagen.save(ruta_imagen, "PNG")
            print(f"  ✓ Página {i} guardada: {nombre_imagen}")
    
    print(f"\n✅ Imágenes guardadas en: {output_dir}")

# ===== USO =====
if __name__ == "__main__":
    # CONFIGURACIÓN - CAMBIA ESTO SEGÚN NECESITES
    PDF_PATH = r"D:\proyectos\prueba\utils\data\output.pdf"  # Ruta a tu PDF
    OUTPUT_DIR = "imagenes_extraidas"  # Carpeta donde guardar
    PAGINAS = [3,4]  # Páginas específicas a extraer (None para todas)
    
    # Ejecutar
    extraer_paginas_pdf(
        pdf_path=PDF_PATH,
        output_dir=OUTPUT_DIR,
        paginas=PAGINAS,
        dpi=300
    )