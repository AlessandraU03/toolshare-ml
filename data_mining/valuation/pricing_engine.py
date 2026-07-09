import os
import pickle
import psycopg2
import logging
import pandas as pd
from typing import Optional
from data_mining.nlp.preprocessor import inferir_tipo_herramienta, TIPO_TO_SECTOR

logger = logging.getLogger("toolshare-ml")

POSTGRES_DSN = "postgres://postgres:lagartija@localhost:5432/tool_inventory?sslmode=disable"
MIN_TRANSACTIONS_THRESHOLD = 30

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_REGRESSION_PATH = os.path.join(MODELS_DIR, "modelo_devaluacion.pkl")

# Carga perezosa del regresor
regression_pipeline = None

def get_regression_pipeline():
    global regression_pipeline
    if regression_pipeline is None:
        if os.path.exists(MODEL_REGRESSION_PATH):
            try:
                with open(MODEL_REGRESSION_PATH, 'rb') as f:
                    regression_pipeline = pickle.load(f)
                logger.info(f"Modelo de Regresión Multi-salida cargado correctamente.")
            except Exception as e:
                logger.error(f"Error cargando regresor: {e}")
        else:
            logger.warning(f"No se encontró regresor en {MODEL_REGRESSION_PATH}")
    return regression_pipeline

def calcular_pricing_motor(
    nombre_herramienta: str,
    score_condicion: float,
    sector: str,
    marca: str,
    age_months: int = 12,
    precio_base_manual: Optional[float] = None
) -> dict:
    """Motor central de cálculo de devaluación y tarifas de renta sugerida."""
    tipo = inferir_tipo_herramienta(nombre_herramienta)
    sector_real = TIPO_TO_SECTOR.get(tipo, sector)
    if not sector_real or sector_real.strip() in ["", "Categoría", "Otro"]:
        sector_real = "Manual"

    n_transacciones = 0
    precio_base_semilla = None
    valor_nuevo_semilla = None
    
    try:
        conn = psycopg2.connect(POSTGRES_DSN)
        cursor = conn.cursor()
        
        cursor.execute("SELECT n_transacciones FROM v_conteo_transacciones_categoria WHERE category = %s", (tipo,))
        row = cursor.fetchone()
        if row:
            n_transacciones = row[0]
            
        cursor.execute("SELECT precio_base, valor_nuevo FROM catalogo_semilla WHERE category = %s", (tipo,))
        row_semilla = cursor.fetchone()
        if row_semilla:
            precio_base_semilla = float(row_semilla[0])
            valor_nuevo_semilla = float(row_semilla[1])
            
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error al consultar base de datos Postgres: {e}")

    modelo_utilizado = ""
    precio_base = 0.0
    valor_depreciado = 0.0
    fuente_precio = ""
    precio_renta_sugerido = 0.0

    if precio_base_manual is not None:
        precio_base = precio_base_manual
        fuente_precio = "entrada_manual_usuario"
    elif valor_nuevo_semilla is not None:
        precio_base = valor_nuevo_semilla
        fuente_precio = "catalogo_semilla"
    else:
        pipeline = get_regression_pipeline()
        if pipeline is not None:
            input_df = pd.DataFrame([{
                'sector_uso': sector_real,
                'marca': marca,
                'tipo_herramienta': tipo,
                'score_condicion': 1.0
            }])
            pred = pipeline.predict(input_df)[0]
            precio_base = float(pred[0])
            fuente_precio = "modelo_regresion_ml_propio"
        else:
            precio_base = 1000.0  # Fallback estático
            fuente_precio = "fallback_catalogo"

    model_filename = f"random_forest_{tipo.lower().replace(' ', '_')}.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)
    
    if n_transacciones >= MIN_TRANSACTIONS_THRESHOLD and os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                rf_model = pickle.load(f)
            
            input_df = pd.DataFrame([{
                'age_months': age_months,
                'condition_score': score_condicion,
                'estimated_value': precio_base
            }])
            pred_rate = rf_model.predict(input_df)[0]
            
            deprec_factor = max(0.15, 1.0 - (age_months * 0.015))
            valor_depreciado = precio_base * deprec_factor * score_condicion
            precio_renta_sugerido = float(pred_rate)
            
            modelo_utilizado = f"Random Forest ({tipo})"
        except Exception as e:
            logger.error(f"Error usando Random Forest específico: {e}")
            n_transacciones = 0 

    if n_transacciones < MIN_TRANSACTIONS_THRESHOLD or precio_renta_sugerido == 0.0:
        # Fallback Heurístico de Minería de Datos
        deprec_factor = max(0.15, 1.0 - (age_months * 0.015))
        valor_depreciado = precio_base * deprec_factor * score_condicion
        
        tasa = 0.015
        if sector_real in ["Eléctrico", "Neumático", "Energía"]:
            tasa = 0.025
            
        precio_renta_sugerido = valor_depreciado * tasa
        modelo_utilizado = "Formula Heuristica Devaluacion"

    precio_renta_sugerido = max(10.0, min(2500.0, precio_renta_sugerido))

    tope_garantia = round(precio_base, 2)
    deducible_sugerido = round(precio_base * 0.10, 2)

    return {
        "tipo_herramienta": tipo,
        "sector_uso": sector_real,
        "marca": marca,
        "fuente_precio_catalogo": fuente_precio,
        "valor_nuevo_estimado": round(precio_base, 2),
        "valor_depreciado_estimado": round(valor_depreciado, 2),
        "precio_renta_sugerido": round(precio_renta_sugerido, 2),
        "deducible_garantia_sugerido": deducible_sugerido,
        "tope_cooperativo_garantia": tope_garantia,
        "modelo_pricing_utilizado": modelo_utilizado
    }
