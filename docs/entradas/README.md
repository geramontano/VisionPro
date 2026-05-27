# Entradas del modelo

Este documento describe las entradas utilizadas por el bloque de modelos del sistema de reconocimiento de emociones.

El objetivo es dejar claro qué tipo de datos recibe el sistema, cómo se transforman y cuál es la entrada real que utilizan los modelos supervisados de clasificación.

---

## 1. Vista general de las entradas

La arquitectura distingue tres niveles principales de entrada:

1. Entrada cruda del sistema.
2. Entrada visual procesada.
3. Entrada numérica para los modelos de aprendizaje automático.

```mermaid
flowchart TD
    A["Entrada cruda<br/>imagen / frame / cámara"] --> B["Detección facial"]
    B --> C["Región facial recortada<br/>Face ROI"]
    C --> D["Preprocesamiento visual"]
    D --> E["Extracción de características"]

    E --> F1["HOG"]
    E --> F2["Textura"]
    E --> F3["Geometría facial"]
    E --> F4["Rasgos oculares / mirada"]

    F1 --> G["Vector de entrada del modelo<br/>X"]
    F2 --> G
    F3 --> G
    F4 --> G

    G --> H["Modelos supervisados"]
```

La entrada final de los modelos no es la imagen cruda, sino un vector numérico de características.

---

## 2. Niveles de entrada

| Nivel | Nombre | Descripción | Uso principal |
|---|---|---|---|
| Nivel 1 | Entrada cruda | Imagen, frame de video o captura desde cámara | Punto inicial del sistema |
| Nivel 2 | Entrada visual procesada | Rostro detectado, recortado y normalizado | Base para extraer características |
| Nivel 3 | Entrada del modelo | Vector numérico de características | Entrada directa de los clasificadores |
| Nivel 4 | Entrada supervisada | Pares `(X_i, y_i)` | Entrenamiento y evaluación |

---

## 3. Entrada cruda del sistema

La entrada cruda corresponde al dato visual original antes de cualquier procesamiento.

Puede provenir de:

```text
imagen individual
frame de video
captura en tiempo real desde cámara
```

```mermaid
flowchart LR
    A["Imagen individual"] --> D["Entrada cruda"]
    B["Frame de video"] --> D
    C["Cámara en tiempo real"] --> D

    D --> E["Detector facial"]
```

Esta entrada todavía no se utiliza directamente para entrenar los modelos. Primero debe pasar por detección facial y preprocesamiento.

---

## 4. Entrada visual procesada

Después de detectar el rostro, se genera una región facial de interés conocida como `Face ROI`.

```mermaid
flowchart TD
    A["Entrada cruda"] --> B["Detección facial"]
    B --> C["Coordenadas del rostro"]
    C --> D["Recorte facial"]
    D --> E["Face ROI"]
    E --> F["Preprocesamiento visual"]
```

La `Face ROI` concentra la información relevante del rostro y elimina regiones no necesarias, como fondo, ropa o elementos externos.

---

## 5. Preprocesamiento de la entrada visual

La entrada visual procesada se estandariza antes de extraer características.

```mermaid
flowchart TD
    A["Face ROI"] --> B["Conversión a escala de grises"]
    B --> C["Redimensionamiento"]
    C --> D["Normalización de intensidad"]
    D --> E["Imagen facial preparada"]
```

| Operación | Propósito |
|---|---|
| Conversión a escala de grises | Reducir complejidad visual |
| Redimensionamiento | Mantener tamaño uniforme |
| Normalización | Hacer comparables los valores de intensidad |
| Preparación facial | Generar una entrada estable para extracción de rasgos |

---

## 6. Entrada para extracción de características

La imagen facial preparada se usa como entrada para construir distintas familias de características.

```mermaid
flowchart TD
    A["Imagen facial preparada"] --> B1["Extractor HOG"]
    A --> B2["Extractor de textura"]
    A --> B3["Extractor de geometría facial"]
    A --> B4["Extractor de rasgos oculares"]

    B1 --> C1["hog_features"]
    B2 --> C2["texture_features"]
    B3 --> C3["geometry_features"]
    B4 --> C4["gaze_features"]
```

Cada extractor produce un vector numérico parcial. Estos vectores se concatenan para formar la entrada real de los modelos.

---

## 7. Entrada real de los modelos

Los modelos supervisados no reciben directamente la imagen completa.

La entrada real de los modelos es un vector de características:

```text
X_i = [HOG_i, texture_i, geometry_i, gaze_i]
```

donde:

| Componente | Tipo | Descripción |
|---|---|---|
| `HOG_i` | Vector numérico | Gradientes, bordes y estructura facial |
| `texture_i` | Vector numérico | Patrones locales de textura |
| `geometry_i` | Vector numérico | Relaciones espaciales del rostro |
| `gaze_i` | Vector numérico | Información ocular, mirada o atención visual |

```mermaid
flowchart LR
    A["hog_features"] --> E["X_i"]
    B["texture_features"] --> E
    C["geometry_features"] --> E
    D["gaze_features"] --> E

    E --> F["Entrada del clasificador"]
```

---

## 8. Contrato de entrada del modelo

El contrato de entrada define qué espera recibir el bloque de modelos.

| Elemento | Formato esperado | Obligatorio | Uso |
|---|---|---|---|
| `face_roi` | Imagen facial recortada | Sí | Base para extracción de características |
| `hog_features` | Vector numérico | Sí | Representación de bordes y gradientes |
| `texture_features` | Vector numérico | Según configuración | Complemento visual local |
| `geometry_features` | Vector numérico | Según configuración | Relaciones espaciales del rostro |
| `gaze_features` | Vector numérico | Según configuración | Rasgos oculares y mirada |
| `X_i` | Vector numérico concatenado | Sí | Entrada directa de los clasificadores |
| `y_i` | Etiqueta categórica | Solo entrenamiento | Emoción real asociada |

---

## 9. Entrada durante entrenamiento

Durante el entrenamiento, cada muestra se representa como un par supervisado:

```text
(X_i, y_i)
```

donde:

```text
X_i = vector de características de la muestra i
y_i = etiqueta emocional de la muestra i
```

```mermaid
flowchart TD
    A["Muestras visuales"] --> B["Extracción de características"]
    B --> C["Matriz X"]

    D["Etiquetas emocionales"] --> E["Vector y"]

    C --> F["Entrenamiento supervisado"]
    E --> F

    F --> G["Modelos candidatos"]
```

La matriz `X` contiene todas las muestras representadas como vectores numéricos.  
El vector `y` contiene las clases emocionales correspondientes.

---

## 10. Entrada durante inferencia

Durante la inferencia, el sistema recibe una nueva muestra visual y produce una emoción estimada.

```mermaid
flowchart TD
    A["Nueva imagen o frame"] --> B["Detección facial"]
    B --> C["Face ROI"]
    C --> D["Preprocesamiento"]
    D --> E["Extracción de características"]
    E --> F["X_new"]
    F --> G["Preprocesamiento guardado"]
    G --> H["Modelo seleccionado"]
    H --> I["Predicción emocional"]
```

La nueva muestra debe pasar por el mismo proceso usado durante el entrenamiento.  
Esto asegura que el modelo reciba datos en el mismo formato.

---

## 11. Relación con los modelos utilizados

Todos los modelos usados reciben la misma entrada `X`.

```mermaid
flowchart TD
    A["Vector de entrada X"] --> B["Escalado / normalización"]

    B --> C1["Logistic Regression"]
    B --> C2["SVM con kernel RBF"]
    B --> C3["Random Forest Classifier"]
    B --> C4["Extra Trees Classifier"]
    B --> C5["MLP Classifier"]

    C1 --> D["Predicción de emoción"]
    C2 --> D
    C3 --> D
    C4 --> D
    C5 --> D
```

Esto permite comparar los modelos de manera justa, ya que todos trabajan con la misma representación de entrada.

---

## 12. Resumen de entradas por componente

| Componente | Entrada | Salida |
|---|---|---|
| Detector facial | Imagen o frame | Coordenadas del rostro |
| Recorte facial | Coordenadas del rostro | Face ROI |
| Preprocesamiento | Face ROI | Imagen facial preparada |
| Extractores de características | Imagen facial preparada | HOG, textura, geometría y mirada |
| Concatenación | Vectores parciales | Vector `X_i` |
| Clasificadores | Vector `X_i` | Emoción predicha |
| Evaluación | Predicción y etiqueta real | Métricas de desempeño |

---

## 13. Resumen profesional

La entrada del bloque de modelos no debe entenderse únicamente como una imagen.  
En esta arquitectura, la imagen es la entrada inicial del sistema, pero la entrada real de los modelos supervisados es un vector de características construido a partir del rostro procesado.

El flujo correcto es:

```text
imagen o frame
-> rostro detectado
-> región facial procesada
-> extracción de características
-> vector X
-> clasificador supervisado
-> emoción predicha
```

Esta separación permite documentar de forma clara la diferencia entre entrada visual, entrada de extracción de rasgos y entrada directa de los modelos de aprendizaje automático.
