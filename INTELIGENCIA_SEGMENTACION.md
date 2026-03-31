# Inteligencia de Segmentación — Cómo se Identifican Profesionales

## Visión General

El sistema de segmentación identifica y agrupa profesionales en expedientes de licitación mediante **tres capas de filtrado + una lógica de consolidación**. No es simple búsqueda de texto — hay decisiones inteligentes en cada paso.

---

## Capa 1: Pre-filtro por Densidad (Candidatas)

**Función:** `es_candidata_separadora()`

### Problema que resuelve
Un documento tiene 390 páginas. ~350 son contenido denso (tablas, listas, formularios). Solo ~40 son **separadoras** (portadas con cargo del profesional). Mandar todas a Qwen (LLM de $) es ineficiente.

### Solución: Filtro de densidad
```
Las separadoras tienen pocas líneas (portadas limpias).
El contenido tiene muchas líneas (tablas, formularios).
```

### Criterios de filtro

**1. Rango de líneas limpias: 1-15 líneas**
- "Línea limpia" = >2 caracteres, no es puro número
- Filtra: 350 páginas de contenido → ~40 candidatas
- Ejemplo:
  ```
  ESPECIALISTA EN SUPERVISIÓN DE ESTRUCTURAS N°1
  (3 líneas de texto limpio → candidata ✓)

  vs

  TABLA CON 50 LÍNEAS DE DATOS
  (50 líneas limpias → NO es candidata, descartada ✗)
  ```

**2. Lista blanca de patrones de cargo: OBLIGATORIO**
- La página DEBE contener ≥1 patrón:
  - "gerente"
  - "jefe"
  - "supervisor" (cubre "Supervisión")
  - "especialista"
  - "coordinador"
  - "residente"
  - "pre instalacion" / "preinstalacion"
- Sin patrón → NO es candidata (evita falsas alarmas)
- Ejemplo:
  ```
  "B.2 EXPERIENCIA DEL PERSONAL CLAVE"
  → No contiene ningún patrón → Descartada en este paso
  ```

**3. Lista negra de descarte: EXCLUSIÓN**
- Si la página contiene CUALQUIERA de estas frases → No es separadora:
  - "a nombre de la nacion" (diplomas universitarios)
  - "el rector de la universidad" (autoridades, no profesionales)
  - "certificado de trabajo" (documentos de soporte)
  - "calificaciones del personal clave" (headers de bloque)
  - Más 4 frases específicas
- Ejemplo:
  ```
  Página con mucho texto "ESPECIALISTA" + "A NOMBRE DE LA NACIÓN"
  → Contiene patrón cargo ✓
  → Pero en lista negra ✗
  → Descartada
  ```

**Resultado de Capa 1:**
```
390 páginas → 40 candidatas → 30 evaluadas → 30 separadoras confirmadas
              ↓
         Reduce costo de Qwen en 90%
```

---

## Capa 2: Confirmación Visual (Qwen)

**Función:** `_confirmar_con_qwen()` + `fuzzy_detect_cargo()`

### Problema que resuelve
Una página puede tener pocas líneas Y el patrón "especialista" pero ser una página de error o una foto de baja calidad. Necesitamos confirmación visual.

### Solución: Dos confirmadores independientes

#### A. Confirmador 1: Qwen-VL (Visual)
```
Qwen ve la imagen de la página y decide:
{
  "es_separadora": true/false,
  "cargo": "Especialista en Supervisión de Estructuras N°2",
  "confianza": "alta" | "media" | "baja"
}
```

**Prompt inteligente:**
- ~140 líneas que instruyen a Qwen sobre:
  - Qué FORMA debe tener una separadora (portada limpia, cargo centrado, etc.)
  - Qué documentos NO son separadoras (diplomas, certificados, IDs)
  - Ejemplos de cargos válidos
  - Importancia de extraer números (N°1, N°2, etc.)

**Decisión:**
- Si `es_separadora=True` AND `confianza ∈ {alta, media}` AND `cargo` no vacío → **Aceptar**
- Si falla cualquier condición → Pasar al fallback

#### B. Confirmador 2: Fuzzy Matching (Fallback)
```
Si Qwen no confirmó con confianza, buscar similitud textual.
```

**Algoritmo:**
1. Extraer el texto OCR de la página
2. Generar candidatos de cargo:
   - Texto completo
   - Líneas individuales
   - Pares de líneas consecutivas
   - Triples de líneas consecutivas
3. Comparar cada candidato contra `CARGOS_BASE` (51 cargos conocidos)
4. Usar `rapidfuzz.token_sort_ratio` para medir similitud
5. Si score ≥ 80 → Aceptar

**Ejemplo:**
```
OCR lee: "Espialista en Supervición de Estruturas 1"
         (errores OCR: Espialista, Supervición, Estruturas)

Fuzzy compara contra "Especialista en Supervisión de Estructuras"
Score: 92 ✓ → Aceptar (match a pesar de errores OCR)
```

**Por qué funciona:**
- Qwen ve la imagen + es robusto a calidad baja
- Fuzzy es instantáneo (0.01s vs 15s Qwen) y maneja OCR errors
- Juntos = confiable + rápido

### Extracción de N° de Cargo
```
Detecta: "Especialista en Supervisión de Estructuras N°2"
Extrae: numero = "2"

Usa regex: r"n[°º]?\s*(\d+)"
Maneja variantes: "N°", "Nº", "N ", "n°", etc.
```

**Resultado de Capa 2:**
```
40 candidatas → Qwen + Fuzzy confirman → 30 separadoras reales
                        ↓
                  Cada una con:
                  - cargo (normalizado)
                  - numero (si existe)
                  - metodo (qwen | fuzzy_fallback)
```

---

## Capa 3: Delimitadores de Bloque

**Función:** `es_delimitador_bloque()`

### Problema que resuelve
Después de detectar separadoras en bloque B.1 (Calificaciones), hay páginas de **transición a B.2** (Experiencia):
```
Separadora: Pág 89  "Especialista En Pre Instalación... N°2"
Contenido:  Págs 90-91  (documentos del profesional)
Delimitador: Pág 92  "B.2 EXPERIENCIA DEL PERSONAL CLAVE" ← CORTE AQUÍ
Ruido:      Págs 93-97  (tablas normativas sueltas, no del profesional)
```

Sin este filtro, el profesional #30 tendría 37 páginas en lugar de 3.

### Solución: Detectar headers de bloque como tijeras
```
Si la página contiene "B.2 EXPERIENCIA DEL PERSONAL CLAVE"
→ Es un delimitador
→ Corta la sección anterior ahí
```

**Inteligencia:**
- No es una separadora (no tiene cargo profesional)
- Pero marca el fin de un bloque temático
- Páginas DESPUÉS de ella no pertenecen al profesional anterior

**Criterios:**
1. Líneas significativas ≤ 30 (filtro de densidad)
   - "Significativa" = >3 chars, no dígito, no solo dashes/puntos
   - Permite ruido visual (líneas de separación)
   - Rechaza páginas de contenido denso
2. Contiene patrón delimitador (lista de 8 patrones):
   - "calificaciones del personal clave"
   - "experiencia del personal clave"
   - "equipamiento estrategico"
   - "documentacion de presentacion"
   - "experiencia en la especialidad adicional"
   - "sostenibilidad ambiental"
   - "integridad en la contratacion"
   - "gestion de calidad"

**Resultado de Capa 3:**
```
Secciones sin recorte:
  Pro #30: págs 89-97 (37 págs infladas)

Con delimitadores:
  Pro #30: págs 89-91 (3 págs correctas) ← Recortada en pág 92
```

---

## Capa 4: Consolidación (Tipo A vs Tipo B)

**Función:** `consolidar_secciones()`

### Problema que resuelve
Documentos con **estructura de bloques temáticos** donde el mismo profesional aparece 3 veces:
```
B.1 CALIFICACIONES:
  Esp. Estructuras N°1 (págs 8-10)

B.2 EXPERIENCIA:
  Esp. Estructuras N°1 (págs 108-111)

B.3 EQUIPAMIENTO:
  Esp. Estructuras N°1 (págs 310-313)
```

Sin consolidación → 3 profesionales diferentes
Con consolidación → 1 profesional con 33 páginas de 3 bloques

### Solución: Agrupar por clave de cargo + número

**Clave de agrupación:**
```python
# Normaliza el cargo para comparación
"Especialista En Supervisión De Estructuras N° 1"
→ "especialista en supervisión de estructuras n°1"
→ (después de regex para variantes de N°)
```

**Lógica:**
1. Agrupar todas las separadoras por clave
2. Si un cargo aparece ≥2 veces → **Tipo B** (múltiples bloques)
3. Fusionar en una sola `ProfessionalSection`:
   - Páginas: unión de todos los bloques, ordenadas
   - `bloques_origen`: preservar rangos originales
   - `es_tipo_b`: boolean que indica estructura múltiple

**Ejemplo de resultado Tipo B:**
```
Especialista En Supervisión De Estructuras N°1

Bloques: 8–10 · 108–111 · 310–313
Total págs: 11
es_tipo_b: True

bloques_origen: [
  PageRange(start=8, end=10, separator_page=8),
  PageRange(start=108, end=111, separator_page=108),
  PageRange(start=310, end=313, separator_page=310),
]
```

---

## Inteligencias específicas

### 1. Manejo de errores OCR
```
OCR lee: "ESPECIALLISTA EN SUPERVISIÓN"
         (dos 'L')

Normalización automática:
- Aplica diccionario de 96 correcciones conocidas
- "especiallista" → "Especialista"
- "supervicion" → "Supervisión"
```

### 2. Variantes de número
```
Qwen puede devolver:
- "N°1"
- "N 1"
- "Nº1"
- "n°1"

Regex normaliza: r"n[°º]?\s*(\d+)"
Todos → numero="1"
```

### 3. Descarte de no-profesionales
```
Página con "ESPECIALISTA" pero:
+ "A NOMBRE DE LA NACIÓN" → diploma universitario
+ "RECTOR DE LA UNIVERSIDAD" → autoridad, no profesional
+ "CERTIFICADO DE TRABAJO" → documento de soporte

Lista negra rechaza automáticamente
```

### 4. Fuzzy fallback inteligente
```
Si Qwen falla o tiene confianza baja:
- Genera múltiples candidatos (líneas, pares, triples)
- Compara cada uno contra 51 cargos base
- Usa token_sort_ratio (ignora orden de palabras)

Ejemplo:
  "Supervisión Especialista De Estructuras"
  ≈ "Especialista En Supervisión De Estructuras"
  (token_sort_ratio = 92 ✓)
```

---

## Flujo integrado

```
PÁGINA 89 (ejemplo: Especialista Estructuras N°2)
    ↓
[1] Densidad OK (3 líneas) ✓
    Contiene "especialista" ✓
    No en lista negra ✓
    → Es candidata
    ↓
[2] Qwen ve imagen
    Confirma: es_separadora=True, cargo="Especialista...", conf=alta
    → Aceptada como separadora
    ↓
[3] Grupo de páginas 89-91 (hasta pág 92 que es delimitador)
    ↓
[4] Número extraído: "2"
    ↓
[5] Clave de agrupación: "especialista en supervisión de estructuras n°2"
    ↓
[6] Buscar otros bloques con misma clave
    Encuentra: págs 108-111, págs 310-313
    ↓
[7] Consolidar en ProfessionalSection (Tipo B)
    Resultado: 11 páginas de 3 bloques
```

---

## Por qué funciona

| Aspecto | Por qué es inteligente |
|--------|----------------------|
| **Pre-filtro densidad** | Reduce 90% de carga a Qwen, mantiene precisión |
| **Qwen + Fuzzy dual** | Robusto a OCR errors Y rápido (fuzzy fallback) |
| **Extracción de N°** | Diferencia "Especialista N°1" de "Especialista N°2" |
| **Delimitadores** | Detecta fin de bloque sin ser separadora |
| **Consolidación Tipo B** | Maneja estructura multi-bloque automáticamente |
| **Lista negra** | Evita falsos positivos (diplomas, certificados) |
| **Normalización OCR** | Corrige errores conocidos sin re-procesar |

---

## Resultados en la práctica

**Documento de prueba: 390 páginas, 30 profesionales**

```
Fase 1 (Densidad):       390 → 40 candidatas
Fase 2 (Qwen + Fuzzy):   40 → 30 separadoras confirmadas
Fase 3 (Delimitadores):  30 → 30 profesionales correctamente recortados
Fase 4 (Consolidación):  30 → 30 profesionales (10 Tipo B consolidados)

Resultado final:
  ✓ 30 profesionales identificados
  ✓ Cargos normalizados
  ✓ Números extraídos
  ✓ Bloques consolidados
  ✓ Páginas correctamente agrupadas
```

---

## Limitaciones conocidas

1. **Fuzzy match score ≥ 80:** A veces muy estricto. Si un cargo tiene mucho error OCR, puede no matchear.
   - **Solución futura:** Ajustar a score ≥ 75, o agregar variantes OCR a CARGOS_BASE

2. **CARGOS_BASE hardcoded:** Si aparece un nuevo cargo (ej: "Especialista en Sostenibilidad"), fuzzy no lo reconoce.
   - **Solución:** Qwen siempre lo aceptaría, pero fuzzy fallback no. Agregar manualmente a CARGOS_BASE.

3. **Delimitadores solo 8 patrones:** Si hay un bloque con nombre diferente, no se detecta.
   - **Solución:** Extender PATRONES_DELIMITADOR con nuevos encontrados.

---

## Conclusión

El sistema no es "buscar texto". Es una **máquina de decisión en capas**:
1. Filtra por heurística simple (densidad)
2. Confirma con IA visual (Qwen)
3. Fallback a fuzzy matching (OCR robust)
4. Detecta límites de estructura (delimitadores)
5. Consolida bloques (Tipo A/B)

Cada capa está diseñada para ser:
- **Rápida** (pre-filtros reducen carga)
- **Confiable** (múltiples confirmadores)
- **Tolerante** (maneja OCR errors, variantes)
- **Informativa** (extrae números, métodos, bloques)
