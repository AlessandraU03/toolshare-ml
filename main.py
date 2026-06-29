# main.py — ToolShare IA & Minería de Datos API (Microservicio FastAPI)
# Arquitectura Limpia: Inferencia de Modelos ML Nativos (CNN + Regresión Multi-salida)
import os
import re
import pickle
import unicodedata
import logging
import numpy as np
import pandas as pd
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import tensorflow as tf

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("toolshare-ml")

# ============================================================================
# CAPA DE DOMINIO: Preprocesamiento NLP
# ============================================================================
STOPWORDS_ES = {
    'de', 'la', 'en', 'el', 'que', 'y', 'un', 'una', 'con', 'para', 'por', 'es', 'al',
    'los', 'las', 'un', 'una', 'unos', 'unas', 'este', 'esta', 'estos', 'estas',
    'del', 'lo', 'como', 'o', 'su', 'sus', 'a', 'para', 'no', 'si'
}

def preprocesar_texto(texto: str) -> str:
    """Pipeline de preprocesamiento NLP alineado con la clase de Minería de Datos."""
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    tokens = re.findall(r'\b[a-z0-9ñ]{2,}\b', texto)
    tokens_filtrados = [t for t in tokens if t not in STOPWORDS_ES]
    return " ".join(tokens_filtrados)

def inferir_tipo_herramienta(nombre: str) -> str:
    """Clasificador simple basado en reglas NLP para mapear la herramienta con el dataset."""
    nombre_clean = preprocesar_texto(nombre).lower()
    
    # Listado de clases/tipos definidos en nuestro dataset real de construcción
    tipos_validos = {
        "rotomartillo": "Rotomartillo",
        "taladro": "Taladro",
        "esmeriladora": "Esmeriladora",
        "esmeril": "Esmeriladora",
        "amoladora": "Esmeriladora",
        "sierra": "Sierra",
        "segueta": "Segueta",
        "serrucho": "Serrucho",
        "cortadora": "Cortadora",
        "lijadora": "Lijadora",
        "cepillo": "Cepillo",
        "pulidora": "Pulidora",
        "vibrador": "Vibrador",
        "generador": "Generador",
        "soldadora": "Soldadora",
        "extension": "Extension",
        "compresor": "Compresor",
        "clavadora": "Clavadora",
        "clavos": "Clavadora",
        "impacto": "Pistola de Impacto",
        "juego": "Juego de Herramientas",
        "autocle": "Juego de Herramientas",
        "llave": "Llave",
        "perica": "Llave",
        "inglesa": "Llave",
        "stilson": "Llave",
        "martillo": "Martillo",
        "mazo": "Mazo",
        "marro": "Marro",
        "cincel": "Cincel",
        "llana": "Llana",
        "espatula": "Espatula",
        "cuchara": "Cuchara",
        "paleta": "Cuchara",
        "desarmador": "Destornillador",
        "destornillador": "Destornillador",
        "pinza": "Pinza",
        "alicates": "Pinza",
        "tenazas": "Pinza",
        "multimetro": "Multimetro",
        "nivel": "Nivel",
        "burbuja": "Nivel",
        "plomada": "Plomada",
        "flexometro": "Flexometro",
        "metro": "Flexometro",
        "cinta": "Flexometro",
        "escuadra": "Escuadra",
        "distanciometro": "Distanciometro",
        "escalera": "Escalera",
        "carretilla": "Carretilla",
        "pala": "Pala",
        "pico": "Pico",
        "azadon": "Azadon",
        "cubeta": "Cubeta",
        "mezcladora": "Mezcladora",
        "andamio": "Andamio",
        "silicon": "Pistola",
        "alambre": "Cepillo"
    }
    
    for kw, tipo in tipos_validos.items():
        if kw in nombre_clean:
            return tipo
    return "Otros"

# ============================================================================
# APLICACIÓN FASTAPI Y CARGA DE MODELOS
# ============================================================================
app = FastAPI(
    title="ToolShare IA & Minería de Datos API",
    description=(
        "Microservicio de Inteligencia Artificial para ToolShare.\n\n"
        "**Componentes:**\n"
        "- **CNN MobileNetV3**: Clasificación visual de desgaste de herramientas\n"
        "- **Random Forest Regressor (Multi-output)**: Predicción del precio original de catálogo y valor depreciado\n"
        "- **TF-IDF + Regresión Logística**: Búsqueda semántica y clasificación NLP\n"
    ),
    version="3.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CNN_PATH = os.path.join(BASE_DIR, "models", "baseline_best.keras")
MODEL_REGRESSION_PATH = os.path.join(BASE_DIR, "models", "modelo_devaluacion.pkl")
MODEL_NLP_PATH = os.path.join(BASE_DIR, "models", "buscador_nlp.pkl")

cnn_model = None
regression_pipeline = None
nlp_data = None
validator_model = None

CLASES_CNN = ["nuevo", "uso_moderado", "viejo_desgastado"]
PESOS_CNN = {"nuevo": 1.0, "uso_moderado": 0.70, "viejo_desgastado": 0.40}

@app.on_event("startup")
def load_models():
    global cnn_model, regression_pipeline, nlp_data, validator_model

    # 1. Cargar modelo CNN (MobileNetV3)
    if os.path.exists(MODEL_CNN_PATH):
        try:
            cnn_model = tf.keras.models.load_model(MODEL_CNN_PATH)
            logger.info(f"Modelo CNN cargado correctamente desde {MODEL_CNN_PATH}")
        except Exception as e:
            logger.error(f"Error al cargar modelo CNN Keras: {e}")
    else:
        logger.warning(f"No se encontró el modelo CNN en {MODEL_CNN_PATH}")

    # 2. Cargar regresor de devaluación
    if os.path.exists(MODEL_REGRESSION_PATH):
        try:
            with open(MODEL_REGRESSION_PATH, 'rb') as f:
                regression_pipeline = pickle.load(f)
            logger.info(f"Modelo de Regresión Multi-salida cargado desde {MODEL_REGRESSION_PATH}")
        except Exception as e:
            logger.error(f"Error al cargar regresor: {e}")
    else:
        logger.warning(f"No se encontró regresor en {MODEL_REGRESSION_PATH}")

    # 3. Cargar buscador y clasificadores NLP
    if os.path.exists(MODEL_NLP_PATH):
        try:
            with open(MODEL_NLP_PATH, 'rb') as f:
                nlp_data = pickle.load(f)
            logger.info(f"Catálogo NLP cargado desde {MODEL_NLP_PATH}")
        except Exception as e:
            logger.error(f"Error al cargar NLP: {e}")
    else:
        logger.warning(f"No se encontró NLP en {MODEL_NLP_PATH}")

    # 4. Cargar Validador de Objetos (ImageNet)
    try:
        validator_model = tf.keras.applications.MobileNetV3Small(weights='imagenet')
        logger.info("Modelo validador ImageNet (MobileNetV3) cargado exitosamente.")
    except Exception as e:
        logger.error(f"Error al cargar validador ImageNet: {e}")

# ============================================================================
# ENDPOINT 1: Clasificación de Desgaste (CNN MobileNetV3)
# ============================================================================
@app.post("/predict-condition",
          summary="Clasifica el nivel de desgaste de una herramienta a partir de una foto")
async def predict_condition(file: UploadFile = File(...)):
    if cnn_model is None:
        raise HTTPException(status_code=503, detail="Modelo CNN no cargado en el servidor")

    try:
        # Preprocesar imagen
        image = Image.open(file.file).convert("RGB")
        
        # ---- VALIDACIÓN DE IMAGEN (SOLO HERRAMIENTAS) ----
        if validator_model is not None:
            img_val = image.resize((224, 224))
            img_val_arr = np.expand_dims(np.array(img_val, dtype=np.float32), axis=0)
            img_val_arr = tf.keras.applications.mobilenet_v3.preprocess_input(img_val_arr)
            
            val_probs = validator_model.predict(img_val_arr, verbose=0)
            decoded = tf.keras.applications.mobilenet_v3.decode_predictions(val_probs, top=5)[0]
            
            # Palabras clave asociadas a herramientas, ferretería y objetos mecánicos/metálicos comunes
            palabras_herramienta = {
                "drill", "hammer", "screwdriver", "saw", "pliers", "tool", "hatchet", "axe", "plane", 
                "clamp", "vice", "chisel", "wrench", "rule", "ruler", "measure", "hardware", "tool", 
                "spanner", "mallet", "anvil", "shears", "scissors", "shovel", "rake", "hoe", "scythe", 
                "sickle", "wheelbarrow", "generator", "engine", "pump", "machine", "device", "gauge",
                "screw", "nail", "nut", "hardware_store", "carpenter", "measuring", "scale", "barometer",
                "joystick", "dial", "switch", "wire", "cable", "mask", "helmet", "safety", "goggles",
                "corkscrew", "can_opener", "opener", "nutcracker", "combination_lock", "padlock", "key", 
                "pincers", "forceps", "tongs", "tweezers", "nipper", "cutters", "clipper", "hook", "chain", 
                "spring", "bar", "rod", "pipe", "tube", "revolver", "pistol", "gun", "lighter", "matchstick",
                "stretcher", "plunger", "crowbar", "sledgehammer", "anvil", "file", "rasp", "spatula"
            }
            
            # Clases comunes de fondo a ignorar
            clases_fondo = {
                "tile", "flooring", "wood", "table", "desk", "wall", "carpet", "rug", "mat", "floor", 
                "concrete", "cement", "ground", "slate", "patio", "sidewalk", "pavement", "brick", "stone", "rock"
            }
            
            es_herramienta = False
            top_detectado = []
            
            for _, class_name, prob in decoded:
                prob_pct = prob * 100
                top_detectado.append(f"{class_name} ({prob_pct:.1f}%)")
                class_name_lower = class_name.lower().replace("_", " ")
                
                # Ignorar si es una clase de fondo (piso, pared, mesa)
                if any(fondo in class_name_lower for fondo in clases_fondo):
                    continue
                    
                # Si coincide con alguna palabra de herramienta o metal, es válida
                if any(kw in class_name_lower for kw in palabras_herramienta):
                    es_herramienta = True
                    break
            
            # Si no se detectó ninguna herramienta en el top 5 (ignorando fondos), rechazar
            if not es_herramienta:
                logger.warning(f"Imagen rechazada. Detectado: {top_detectado}")
                detectado_legible = decoded[0][1].replace("_", " ").capitalize()
                raise HTTPException(
                    status_code=400,
                    detail=f"La imagen subida no corresponde a una herramienta de construcción válida. Detectado en la foto: '{detectado_legible}'"
                )

        # Inferencia de desgaste (CNN propia)
        img_cnn = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img_cnn, dtype=np.float32), axis=0)

        probs = cnn_model.predict(img_array, verbose=0)[0]
        score = sum(probs[i] * PESOS_CNN[CLASES_CNN[i]] for i in range(len(CLASES_CNN)))

        idx_max = np.argmax(probs)
        clase_predicha = CLASES_CNN[idx_max]

        return {
            "clase_predicha": clase_predicha,
            "score_condicion": round(float(score), 3),
            "probabilidades": {
                CLASES_CNN[i]: round(float(probs[i]), 4) for i in range(len(CLASES_CNN))
            }
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia CNN: {str(e)}")

# ============================================================================
# ENDPOINT 2: Predicción de Precio de Mercado
# ============================================================================
@app.get("/market-price",
         summary="Predice el precio original estimado de catálogo de una herramienta")
def get_market_price(
    query: str = Query(..., description="Nombre de la herramienta (ej: 'rotomartillo')"),
    sector: str = Query("Otro", description="Categoría o sector de uso"),
    marca: str = Query("Generico", description="Marca de la herramienta")
):
    if regression_pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo de regresión no cargado")

    try:
        tipo = inferir_tipo_herramienta(query)
        input_df = pd.DataFrame([{
            'sector_uso': sector,
            'marca': marca,
            'tipo_herramienta': tipo,
            'score_condicion': 1.0  # Asumimos estado como nueva para precio de catálogo
        }])

        pred = regression_pipeline.predict(input_df)[0]
        precio_base = round(float(pred[0]), 2)

        return {
            "precio_promedio": precio_base,
            "precio_minimo": round(precio_base * 0.85, 2),
            "precio_maximo": round(precio_base * 1.15, 2),
            "precio_mediana": precio_base,
            "total_resultados": 1,
            "muestras_analizadas": 1,
            "fuente": "modelo_regresion_ml_propio",
            "productos": [
                {"titulo": f"{query.capitalize()} {marca.capitalize()}", "precio": precio_base, "condicion": "new", "moneda": "MXN"}
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al estimar precio de catálogo: {str(e)}")

# ============================================================================
# ENDPOINT 3: Valoración por Entrada Manual (Precio manual + Regresión de Devaluación)
# ============================================================================
@app.get("/suggest-price",
         summary="Predice el valor depreciado real y sugiere precios de renta a partir de un precio base")
def suggest_price(
    precio_base: float = Query(..., description="Precio original/base de catálogo ingresado manualmente"),
    score_condicion: float = Query(..., description="Score de desgaste visual devuelto por la CNN (0.40 a 1.0)"),
    sector: str = Query(..., description="Sector de uso"),
    marca: str = Query("Generico", description="Marca de la herramienta"),
    nombre_herramienta: str = Query("Herramienta", description="Nombre de la herramienta")
):
    if regression_pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo de regresión no cargado")

    try:
        tipo = inferir_tipo_herramienta(nombre_herramienta)
        input_df = pd.DataFrame([{
            'sector_uso': sector,
            'marca': marca,
            'tipo_herramienta': tipo,
            'score_condicion': score_condicion
        }])

        pred = regression_pipeline.predict(input_df)[0]
        precio_base_predicho = float(pred[0])
        valor_depreciado_predicho = float(pred[1])

        # Escalar proporcionalmente usando la relación de devaluación predicha por el modelo
        ratio_devaluacion = valor_depreciado_predicho / max(1.0, precio_base_predicho)
        valor_depreciado = precio_base * ratio_devaluacion

        valor_garantia = min(10000.0, valor_depreciado)

        # Reglas de negocio para renta diaria
        renta_pct = {
            'Eléctrico': 0.035, 'Corte': 0.040, 'Acabado': 0.030,
            'Energía': 0.045, 'Neumático': 0.040, 'Manual': 0.020,
            'Medición': 0.025, 'Otro': 0.025
        }.get(sector, 0.025)

        renta_sugerida = valor_depreciado * renta_pct
        renta_minima = (valor_depreciado * 0.5) / 30

        if valor_depreciado <= 1500:
            prima_garantia_anual = 180.0
        elif valor_depreciado <= 5000:
            prima_garantia_anual = 400.0
        else:
            prima_garantia_anual = 900.0

        deducible_dano = round(valor_depreciado * 0.10, 2)

        return {
            "valor_real_depreciado": round(valor_depreciado, 2),
            "tope_cobertura_garantia": round(valor_garantia, 2),
            "precio_renta_sugerido": round(max(10.0, renta_sugerida), 2),
            "precio_renta_minimo": round(max(10.0, renta_minima), 2),
            "prima_garantia_anual": prima_garantia_anual,
            "deducible_por_dano": deducible_dano,
            "moneda": "MXN",
            "detalles_calculo": {
                "porcentaje_renta_aplicado": f"{renta_pct*100:.1f}%",
                "se_aplico_tope_garantia": valor_depreciado > 10000.0,
                "regla_50_porciento_30_dias": f"${renta_minima:.2f}/día"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en sugerencia de precio: {str(e)}")

# ============================================================================
# ENDPOINT 4: Valuación Automática Completa (Regresión Multi-salida Inteligente)
# ============================================================================
@app.get("/auto-valuate",
         summary="Valuación automática completa: predice precio de catálogo y depreciación de forma matemática")
def auto_valuate(
    nombre_herramienta: str = Query(..., description="Nombre de la herramienta (ej: 'Rotomartillo Makita HR2470')"),
    score_condicion: float = Query(0.70, description="Score de desgaste visual (0.40 a 1.0). Default: 0.70"),
    sector: str = Query("Otro", description="Sector de uso"),
    marca: str = Query("Generico", description="Marca de la herramienta"),
    access_token: Optional[str] = Query(None, description="Descartado (No se requiere para predicciones nativas de ML)")
):
    if regression_pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo de regresión no cargado en el servidor")

    try:
        tipo = inferir_tipo_herramienta(nombre_herramienta)
        input_df = pd.DataFrame([{
            'sector_uso': sector,
            'marca': marca,
            'tipo_herramienta': tipo,
            'score_condicion': score_condicion
        }])

        # El regresor multi-salida nos da el precio de catálogo y devaluado en un solo paso
        pred = regression_pipeline.predict(input_df)[0]
        precio_base = float(pred[0])
        valor_depreciado = float(pred[1])

        # Formatear límites coherentes
        precio_base = round(max(100.0, precio_base), 2)
        valor_depreciado = round(max(50.0, min(precio_base, valor_depreciado)), 2)

        valor_garantia = min(10000.0, valor_depreciado)

        # Reglas de negocio para renta
        renta_pct = {
            'Eléctrico': 0.035, 'Corte': 0.040, 'Acabado': 0.030,
            'Energía': 0.045, 'Neumático': 0.040, 'Manual': 0.020,
            'Medición': 0.025, 'Otro': 0.025
        }.get(sector, 0.025)

        renta_sugerida = valor_depreciado * renta_pct
        renta_minima = (valor_depreciado * 0.5) / 30

        if valor_depreciado <= 1500:
            prima_garantia_anual = 180.0
        elif valor_depreciado <= 5000:
            prima_garantia_anual = 400.0
        else:
            prima_garantia_anual = 900.0

        deducible_dano = round(valor_depreciado * 0.10, 2)
        requiere_plan_pro = valor_depreciado > 1500.0

        return {
            "nombre_herramienta": nombre_herramienta,
            "precio_base_mercado": precio_base,
            "fuente_precio": "modelo_regresion_ml_propio",
            "valor_real_depreciado": valor_depreciado,
            "tope_cobertura_garantia": round(valor_garantia, 2),
            "precio_renta_sugerido": round(max(10.0, renta_sugerida), 2),
            "precio_renta_minimo": round(max(10.0, renta_minima), 2),
            "prima_garantia_anual": prima_garantia_anual,
            "deducible_por_dano": deducible_dano,
            "requiere_plan_pro": requiere_plan_pro,
            "moneda": "MXN",
            "productos_referencia": [
                {"titulo": f"{nombre_herramienta} {marca}", "precio": precio_base, "condicion": "new", "moneda": "MXN"}
            ],
            "detalles_calculo": {
                "porcentaje_renta_aplicado": f"{renta_pct*100:.1f}%",
                "se_aplico_tope_garantia": valor_depreciado > 10000.0,
                "regla_50_porciento_30_dias": f"${renta_minima:.2f}/día",
                "modelo_ml_utilizado": "RandomForestRegressor (Multi-Output / R²=0.964)"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en valuación automática: {str(e)}")

# ============================================================================
# ENDPOINT 5: Búsqueda Semántica y Clasificación (NLP)
# ============================================================================
@app.get("/search",
         summary="Busca herramientas mediante lenguaje natural y las categoriza en ligero/pesado")
def search_tools(query: str = Query(..., description="Consulta en lenguaje natural sobre la herramienta o tarea")):
    if nlp_data is None:
        raise HTTPException(status_code=503, detail="Modelos NLP no cargados en el servidor")

    try:
        vectorizer = nlp_data['vectorizer']
        classifier = nlp_data['classifier']
        tfidf_matrix = nlp_data['tfidf_matrix']
        df_catalog = nlp_data['df_catalog']

        query_processed = preprocesar_texto(query)
        query_vec = vectorizer.transform([query_processed])

        probs = classifier.predict_proba(query_vec)[0]
        pred_class = classifier.predict(query_vec)[0]
        class_label = "Construcción Pesada" if pred_class == 1 else "Construcción Ligera"
        confianza = float(probs[pred_class])

        from sklearn.metrics.pairwise import cosine_similarity
        similaridades = cosine_similarity(query_vec, tfidf_matrix).flatten()

        df_res = df_catalog.copy()
        df_res['score_similitud'] = similaridades

        resultados = df_res.sort_values(by='score_similitud', ascending=False)

        items = []
        for _, row in resultados.iterrows():
            if row['score_similitud'] > 0.0:
                items.append({
                    "nombre": row['nombre'],
                    "descripcion": row['descripcion'],
                    "categoria": row['categoria'],
                    "score_similitud": round(float(row['score_similitud']), 4)
                })

        return {
            "query": query,
            "categoria_predicha": class_label,
            "confianza_clasificacion": round(confianza, 4),
            "resultados": items[:5]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en motor de búsqueda NLP: {str(e)}")

# ============================================================================
# ENDPOINT 6: Health Check
# ============================================================================
@app.get("/health", summary="Verificación de salud del servicio y estado de los modelos")
def health_check():
    return {
        "status": "ok",
        "modelos": {
            "cnn_mobilenetv3": "cargado" if cnn_model is not None else "no_cargado",
            "regresor_devaluacion": "cargado" if regression_pipeline is not None else "no_cargado",
            "nlp_buscador": "cargado" if nlp_data is not None else "no_cargado"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    reload_mode = os.environ.get("PORT") is None
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload_mode)
