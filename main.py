# main.py — ToolShare IA & Minería de Datos API (Microservicio FastAPI)
# Arquitectura Limpia: Capa de Aplicación (FastAPI) → Capa de Dominio (Modelos ML) → Capa de Infraestructura (APIs externas)
import os
import re
import pickle
import unicodedata
import logging
import numpy as np
import pandas as pd
import requests
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

# ============================================================================
# CAPA DE INFRAESTRUCTURA: Consulta API Pública de Mercado Libre México
# ============================================================================
MERCADOLIBRE_API_BASE = "https://api.mercadolibre.com/sites/MLM/search"

def consultar_precio_mercadolibre(termino_busqueda: str, limite: int = 10, access_token: str = None) -> dict:
    """
    Consulta la API pública de Mercado Libre México (sitio MLM) para obtener
    precios reales de mercado de herramientas.
    """
    try:
        params = {
            "q": termino_busqueda,
            "limit": limite,
            "sort": "relevance"
        }
        
        headers = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        if access_token and access_token.strip():
            headers["Authorization"] = f"Bearer {access_token}"
            
        response = requests.get(
            MERCADOLIBRE_API_BASE,
            params=params,
            timeout=5,
            headers=headers
        )
        
        if response.status_code != 200:
            logger.warning(f"API de Mercado Libre respondió con status {response.status_code}")
            if response.status_code == 403:
                logger.warning(
                    "Mercado Libre retornó 403 Forbidden. Para habilitar consultas reales en desarrollo, "
                    "asegúrate de registrar un token válido configurando MP_ACCESS_TOKEN en tu archivo .env."
                )
            return None
        
        datos = response.json()
        resultados = datos.get('results', [])
        
        if not resultados:
            logger.info(f"Sin resultados en Mercado Libre para: '{termino_busqueda}'")
            return None
        
        # Extraer precios válidos (solo artículos con precio > 0 y en MXN)
        precios = []
        productos = []
        for item in resultados:
            precio = item.get('price', 0)
            moneda = item.get('currency_id', 'MXN')
            titulo = item.get('title', 'Sin título')
            condicion = item.get('condition', 'no_especificada')
            
            if precio and precio > 0 and moneda == 'MXN':
                precios.append(precio)
                productos.append({
                    "titulo": titulo,
                    "precio": precio,
                    "condicion": condicion,
                    "moneda": moneda
                })
        
        if not precios:
            return None
        
        return {
            "precio_promedio": round(sum(precios) / len(precios), 2),
            "precio_minimo": round(min(precios), 2),
            "precio_maximo": round(max(precios), 2),
            "precio_mediana": round(sorted(precios)[len(precios) // 2], 2),
            "total_resultados": datos.get('paging', {}).get('total', len(resultados)),
            "muestras_analizadas": len(precios),
            "fuente": "mercadolibre_api_publica",
            "productos": productos[:5]  # Top 5 para referencia
        }
        
    except requests.exceptions.Timeout:
        logger.warning("Timeout al consultar API de Mercado Libre")
        return None
    except requests.exceptions.ConnectionError:
        logger.warning("Error de conexión con API de Mercado Libre")
        return None
    except Exception as e:
        logger.error(f"Error inesperado consultando Mercado Libre: {e}")
        return None

# ============================================================================
# APLICACIÓN FASTAPI
# ============================================================================
app = FastAPI(
    title="ToolShare IA & Minería de Datos API",
    description=(
        "Microservicio de Inteligencia Artificial para ToolShare.\n\n"
        "**Componentes:**\n"
        "- **CNN MobileNetV3**: Clasificación visual de desgaste de herramientas\n"
        "- **Random Forest Regressor**: Predicción de valor depreciado y renta sugerida\n"
        "- **TF-IDF + Regresión Logística**: Búsqueda semántica y clasificación NLP\n"
        "- **API Mercado Libre**: Consulta de precios de mercado en tiempo real\n"
    ),
    version="2.0"
)

# ---- Configuración de Rutas de Modelos ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CNN_PATH = os.path.join(BASE_DIR, "models", "baseline_best.keras")
MODEL_REGRESSION_PATH = os.path.join(BASE_DIR, "models", "modelo_devaluacion.pkl")
MODEL_NLP_PATH = os.path.join(BASE_DIR, "models", "buscador_nlp.pkl")

# ---- Carga de Modelos ----
cnn_model = None
regression_pipeline = None
nlp_data = None

CLASES_CNN = ["nuevo", "uso_moderado", "viejo_desgastado"]
PESOS_CNN = {"nuevo": 1.0, "uso_moderado": 0.70, "viejo_desgastado": 0.40}

@app.on_event("startup")
def load_models():
    global cnn_model, regression_pipeline, nlp_data

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
            logger.info(f"Modelo de Regresión cargado desde {MODEL_REGRESSION_PATH}")
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


# ============================================================================
# ENDPOINT 1: Clasificación de Desgaste (CNN MobileNetV3)
# ============================================================================
@app.post("/predict-condition",
          summary="Clasifica el nivel de desgaste de una herramienta a partir de una foto")
async def predict_condition(file: UploadFile = File(...)):
    if cnn_model is None:
        raise HTTPException(status_code=503, detail="Modelo CNN no cargado en el servidor")

    try:
        image = Image.open(file.file).convert("RGB").resize((224, 224))
        img_array = np.expand_dims(np.array(image, dtype=np.float32), axis=0)

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia CNN: {str(e)}")


# ============================================================================
# ENDPOINT 2: Consulta de Precios de Mercado (API Mercado Libre)
# ============================================================================
@app.get("/market-price",
         summary="Consulta precios reales de mercado de herramientas vía API de Mercado Libre México")
def get_market_price(
    query: str = Query(..., description="Nombre o descripción de la herramienta (ej: 'taladro makita 12v')"),
    limite: int = Query(10, description="Cantidad de resultados a analizar (máx 50)", ge=1, le=50),
    access_token: Optional[str] = Query(None, description="Access Token de Mercado Libre / Mercado Pago")
):
    """
    Consulta la API pública de Mercado Libre México (sitio MLM) para obtener
    precios reales de mercado. Esto reemplaza el web scraping convencional
    que sufriría bloqueos por parte de Mercado Libre (Cloudflare, CAPTCHAs).
    
    **Ventajas sobre Web Scraping:**
    - No sufre bloqueos de IP ni CAPTCHAs
    - Datos estructurados en JSON (no hay que parsear HTML)
    - API oficial, legal y gratuita
    - Respuesta rápida (< 1 segundo)
    """
    resultado = consultar_precio_mercadolibre(query, limite, access_token)

    if resultado is None:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontraron resultados en Mercado Libre para: '{query}'"
        )

    return resultado


# ============================================================================
# ENDPOINT 3: Valoración Completa (Precio de Mercado + Regresión de Devaluación)
# ============================================================================
@app.get("/suggest-price",
         summary="Predice el valor depreciado real y sugiere precios de renta")
def suggest_price(
    precio_base: float = Query(..., description="Precio original/base de catálogo de la herramienta"),
    score_condicion: float = Query(..., description="Score de desgaste visual devuelto por la CNN (0.40 a 1.0)"),
    sector: str = Query(..., description="Sector de uso: Eléctrico, Corte, Acabado, Energía, Neumático, Manual, Medición, Otro"),
    marca: str = Query("Generico", description="Marca: Makita, DeWalt, Bosch, Truper, Milwaukee, Fluke, Honda, Stanley, Generico")
):
    if regression_pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo de regresión no cargado en el servidor")

    try:
        input_df = pd.DataFrame([{
            'precio_base_catalogo': precio_base,
            'score_condicion': score_condicion,
            'sector_uso': sector,
            'marca': marca
        }])

        valor_depreciado = float(regression_pipeline.predict(input_df)[0])
        valor_garantia = min(10000.0, valor_depreciado)

        renta_pct = {
            'Eléctrico': 0.035, 'Corte': 0.040, 'Acabado': 0.030,
            'Energía': 0.045, 'Neumático': 0.040, 'Manual': 0.020,
            'Medición': 0.025, 'Otro': 0.025
        }.get(sector, 0.025)

        renta_sugerida = valor_depreciado * renta_pct

        # Regla de negocio MVP: El precio mínimo de renta es el 50% del valor estimado en 30 días
        renta_minima = (valor_depreciado * 0.5) / 30

        # Calcular primas de garantía anual según el valor depreciado
        if valor_depreciado <= 1500:
            prima_garantia_anual = 180.0
        elif valor_depreciado <= 5000:
            prima_garantia_anual = 400.0
        else:
            prima_garantia_anual = 900.0

        # Deducible por daño: 10% del valor comercial
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
        raise HTTPException(status_code=500, detail=f"Error en predicción de precios: {str(e)}")


# ============================================================================
# ENDPOINT 4: Valuación Automática Completa (Mercado Libre + Regresión)
# Combina la consulta de precios reales con el modelo de devaluación
# ============================================================================
@app.get("/auto-valuate",
         summary="Valuación automática completa: consulta precios reales de Mercado Libre y aplica el modelo de devaluación")
def auto_valuate(
    nombre_herramienta: str = Query(..., description="Nombre de la herramienta (ej: 'Rotomartillo Makita HR2470')"),
    score_condicion: float = Query(0.70, description="Score de desgaste visual (0.40 a 1.0). Default: 0.70 (uso moderado)"),
    sector: str = Query("Otro", description="Sector de uso: Eléctrico, Corte, Acabado, Energía, Neumático, Manual, Medición, Otro"),
    marca: str = Query("Generico", description="Marca: Makita, DeWalt, Bosch, Truper, Milwaukee, Fluke, Honda, Stanley, Generico"),
    access_token: Optional[str] = Query(None, description="Access Token de Mercado Libre / Mercado Pago")
):
    """
    **Pipeline completo de valuación automatizada:**
    
    1. Consulta la API pública de Mercado Libre México para obtener el precio base real
    2. Alimenta el modelo Random Forest Regressor con el precio base, score de condición, sector y marca
    3. Retorna el valor depreciado, renta sugerida, prima de garantía y deducible por daño
    
    Este endpoint elimina la necesidad de que el usuario declare manualmente el valor de su herramienta.
    """
    if regression_pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo de regresión no cargado en el servidor")

    # PASO 1: Consultar precio base de mercado en Mercado Libre
    datos_mercado = consultar_precio_mercadolibre(nombre_herramienta, limite=10, access_token=access_token)

    precio_base = None
    fuente_precio = "no_disponible"
    productos_referencia = []

    if datos_mercado:
        precio_base = datos_mercado["precio_mediana"]  # Usamos mediana para mayor robustez
        fuente_precio = "mercadolibre_api_publica"
        productos_referencia = datos_mercado.get("productos", [])
        logger.info(f"Precio base obtenido de Mercado Libre: ${precio_base} MXN (mediana de {datos_mercado['muestras_analizadas']} resultados)")
    else:
        # Fallback: precios base por categoría si la API no responde
        precios_fallback = {
            'Eléctrico': 3000.0, 'Corte': 2800.0, 'Acabado': 1800.0,
            'Energía': 5000.0, 'Neumático': 3500.0, 'Manual': 800.0,
            'Medición': 1500.0, 'Otro': 1000.0
        }
        precio_base = precios_fallback.get(sector, 1500.0)
        fuente_precio = "fallback_local"
        logger.warning(f"API de Mercado Libre no disponible. Usando precio fallback: ${precio_base} MXN")

    # PASO 2: Ejecutar el regresor de devaluación
    try:
        input_df = pd.DataFrame([{
            'precio_base_catalogo': precio_base,
            'score_condicion': score_condicion,
            'sector_uso': sector,
            'marca': marca
        }])

        valor_depreciado = float(regression_pipeline.predict(input_df)[0])
        valor_garantia = min(10000.0, valor_depreciado)

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

        # Determinar si la herramienta califica para Plan Gratuito o requiere Plan Pro
        requiere_plan_pro = valor_depreciado > 1500.0

        return {
            "nombre_herramienta": nombre_herramienta,
            "precio_base_mercado": round(precio_base, 2),
            "fuente_precio": fuente_precio,
            "valor_real_depreciado": round(valor_depreciado, 2),
            "tope_cobertura_garantia": round(valor_garantia, 2),
            "precio_renta_sugerido": round(max(10.0, renta_sugerida), 2),
            "precio_renta_minimo": round(max(10.0, renta_minima), 2),
            "prima_garantia_anual": prima_garantia_anual,
            "deducible_por_dano": deducible_dano,
            "requiere_plan_pro": requiere_plan_pro,
            "moneda": "MXN",
            "productos_referencia": productos_referencia,
            "detalles_calculo": {
                "porcentaje_renta_aplicado": f"{renta_pct*100:.1f}%",
                "se_aplico_tope_garantia": valor_depreciado > 10000.0,
                "regla_50_porciento_30_dias": f"${renta_minima:.2f}/día",
                "modelo_ml_utilizado": "RandomForestRegressor (R²=0.9707)"
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
