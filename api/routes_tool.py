import os
import io
import logging
from PIL import Image
import numpy as np
import tensorflow as tf
from fastapi import APIRouter, File, UploadFile, Query, HTTPException, status
from data_mining.valuation.pricing_engine import calcular_pricing_motor

logger = logging.getLogger("toolshare-ml")

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_CNN_PATH = os.path.join(MODELS_DIR, "modelo_desgaste.keras")

# Carga diferida de CNN
cnn_model = None

def get_cnn_model():
    global cnn_model
    if cnn_model is None:
        if os.path.exists(MODEL_CNN_PATH):
            try:
                cnn_model = tf.keras.models.load_model(MODEL_CNN_PATH)
                logger.info("Modelo CNN de Desgaste cargado exitosamente.")
            except Exception as e:
                logger.error(f"Error cargando modelo CNN: {e}")
        else:
            logger.warning(f"No se encontró modelo CNN en {MODEL_CNN_PATH}")
    return cnn_model

CLASS_NAMES = ["Muy desgastado", "Desgaste moderado", "Buen estado", "Excelente"]

@router.post("/predict-condition")
async def predict_condition(file: UploadFile = File(...)):
    """Predice el estado de desgaste físico de la herramienta mediante CNN (MobileNetV3)."""
    model = get_cnn_model()
    if model is None:
        return {
            "clase_predicha": "Buen estado",
            "score_condicion": 0.85,
            "probabilidades": {"Buen estado": 1.0}
        }

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]
        max_idx = int(np.argmax(predictions))
        
        scores_mapping = [0.45, 0.65, 0.85, 1.0]
        score_cond = scores_mapping[max_idx]

        probs = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}

        return {
            "clase_predicha": CLASS_NAMES[max_idx],
            "score_condicion": score_cond,
            "probabilidades": probs
        }
    except Exception as e:
        logger.error(f"Error en predict-condition: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando la imagen de la herramienta: {str(e)}"
        )

@router.get("/auto-valuate")
async def auto_valuate(
    nombre_herramienta: str = Query(..., description="Nombre comercial de la herramienta"),
    score_condicion: float = Query(..., description="Score de desgaste continuo [0.1 - 1.0]"),
    sector: str = Query("Manual", description="Categoría de uso"),
    marca: str = Query("Generica", description="Marca fabricante"),
    age_months: int = Query(12, description="Edad de la herramienta en meses"),
    precio_base_manual: float = Query(None, description="Precio de compra original opcional")
):
    """Calcula precios sugeridos, devaluación cooperativa y tope de garantías."""
    try:
        res = calcular_pricing_motor(
            nombre_herramienta=nombre_herramienta,
            score_condicion=score_condicion,
            sector=sector,
            marca=marca,
            age_months=age_months,
            precio_base_manual=precio_base_manual
        )
        return res
    except Exception as e:
        logger.error(f"Error en auto-valuate: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculando la valuación: {str(e)}"
        )
