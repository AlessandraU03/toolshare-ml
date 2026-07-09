# ToolShare - Microservicio de Minería de Datos & Machine Learning

Este microservicio en Python FastAPI encapsula todos los algoritmos analíticos, modelos predictivos y validaciones de visión computacional de ToolShare. Su estructura está diseñada siguiendo los principios de la **Ingeniería de Software y MLOps (Machine Learning Operations)**.

---

## 🏗️ Arquitectura de Carpetas y Organización

La estructura del proyecto separa claramente las responsabilidades del sistema:

### 1. `api/` (Fase Online - Interfaz HTTP)
Contiene la capa de comunicación web del microservicio.
* **`api/main.py`:** Punto de entrada de FastAPI, levanta el servidor y registra los routers modulares.
* **`api/routes_tool.py`:** Enrutador de operaciones de herramientas (Desgaste CNN, Valuación Inteligente).
* **`api/routes_kyc.py`:** Enrutador del caso de uso de verificación de identidad KYC.

### 2. `data_mining/` (Fase Online - Algoritmos Activos)
Contiene el motor de inferencia analítica y procesamiento de datos en tiempo real:
* **`nlp/preprocessor.py`:** Tokenizador, eliminación de stop words en español y clasificación heurística de tipos de herramientas.
* **`valuation/pricing_engine.py`:** Lógica de devaluación y depreciación de herramientas, integración con regresores y bases de datos.
* **`kyc/`:** Detección de rostros con clasificadores Haar Cascades de OpenCV (`face_detector.py`) e interpretación OCR/INE (`ocr_scanner.py`).

### 3. `model_training/` (Fase Offline - Modelado e Investigación)
Contiene las canalizaciones (pipelines) y scripts independientes para entrenamiento de modelos y minería exploratoria de datos. **No se ejecutan en producción:**
* `entrenar.py`: Script para entrenamiento de MobileNetV3 (CNN de desgaste).
* `entrenar_random_forest.py`: Entrenamiento de estimadores específicos por categoría.
* `detectar_outliers_isolation_forest.py`: Remoción de ruido y valores atípicos en precios.
* `entrenar_semilla_kmeans.py`: Algoritmo de agrupamiento K-Means para clasificar el catálogo inicial.
* `split_dataset.py`, `verificar_dataset.py`, `evaluar_completo.py`: Herramientas de soporte analítico.

### 4. `models/` (Modelos Serializados)
Carpetas donde se almacenan los pesos entrenados (`.keras` y `.pkl`) listos para ser consumidos por `data_mining/` en producción.

---

## 🚀 Cómo Ejecutar el Proyecto

### 1. Levantar el Servidor de API (Producción/Inferencia)
Para iniciar el servidor local de FastAPI con recarga en vivo (reload), ejecuta en tu terminal:

```bash
uvicorn api.main:app --reload --port 8000
```

### 2. Ejecutar Pipelines de Entrenamiento (Offline)
Si deseas reentrenar alguno de los modelos o realizar minería exploratoria, ejecuta el script respectiva desde el directorio raíz usando el entorno virtual:

```bash
# Ejemplo: Entrenar el modelo de devaluación
python model_training/entrenar_devaluacion.py

# Ejemplo: Limpiar el dataset de precios usando Isolation Forest
python model_training/detectar_outliers_isolation_forest.py
```
