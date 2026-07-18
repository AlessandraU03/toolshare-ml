import os
import io
import logging
from PIL import Image
import numpy as np
import tensorflow as tf
from fastapi import APIRouter, File, UploadFile, Query, HTTPException, status
from data_mining.valuation.pricing_engine import calcular_pricing_motor
from data_mining.valuation.ticket_ocr import extraer_precio_de_ticket

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

# Mismo orden de clases que model_training/entrenar.py (CLASES) — el índice
# de salida del modelo depende de ese orden, así que deben mantenerse idénticos.
CLASS_NAMES = ["nuevo", "uso_moderado", "viejo_desgastado"]
CLASS_LABELS_DISPLAY = {
    "nuevo": "Nuevo",
    "uso_moderado": "Uso moderado",
    "viejo_desgastado": "Viejo / desgastado",
}
# score_condicion en el rango 0.40 - 1.0 definido en la arquitectura del proyecto.
SCORE_MAPPING = {"nuevo": 1.0, "uso_moderado": 0.65, "viejo_desgastado": 0.40}

@router.post("/predict-condition")
async def predict_condition(file: UploadFile = File(...)):
    """Predice el estado de desgaste físico de la herramienta mediante CNN (MobileNetV3)."""
    model = get_cnn_model()
    if model is None:
        return {
            "clase_predicha": "Uso moderado",
            "score_condicion": 0.65,
            "probabilidades": {"uso_moderado": 1.0}
        }

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]
        max_idx = int(np.argmax(predictions))
        clase = CLASS_NAMES[max_idx]

        probs = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}

        return {
            "clase_predicha": CLASS_LABELS_DISPLAY[clase],
            "score_condicion": SCORE_MAPPING[clase],
            "probabilidades": probs
        }
    except Exception as e:
        logger.error(f"Error en predict-condition: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando la imagen de la herramienta: {str(e)}"
        )

@router.post("/extract-ticket-price")
async def extract_ticket_price(file: UploadFile = File(...)):
    """Lee un ticket/factura de compra por OCR y extrae el monto pagado.

    Es la fuente de precio más confiable del motor de pricing: el propietario
    sube un comprobante real en vez de declarar un valor a mano.
    """
    try:
        contents = await file.read()
        resultado = extraer_precio_de_ticket(contents)
        return resultado
    except Exception as e:
        logger.error(f"Error en extract-ticket-price: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando el ticket de compra: {str(e)}"
        )

@router.get("/auto-valuate")
async def auto_valuate(
    nombre_herramienta: str = Query(..., description="Nombre comercial de la herramienta"),
    score_condicion: float = Query(..., description="Score de desgaste continuo [0.40 - 1.0]"),
    sector: str = Query("Manual", description="Categoría de uso"),
    marca: str = Query("Generica", description="Marca fabricante"),
    age_months: int = Query(12, description="Edad de la herramienta en meses"),
    precio_base_manual: float = Query(None, description="Precio de compra original opcional"),
    ticket_validado: bool = Query(False, description="True si precio_base_manual viene de un ticket leído por OCR"),
):
    """Calcula precios sugeridos, devaluación cooperativa y tope de garantías.

    El shape de la respuesta está alineado con el struct que decodifica
    Api_Apptransacional/internal/tool/service/tool_service.go (AutoValuate).
    """
    try:
        res = calcular_pricing_motor(
            nombre_herramienta=nombre_herramienta,
            score_condicion=score_condicion,
            sector=sector,
            marca=marca,
            age_months=age_months,
            precio_base_manual=precio_base_manual,
            ticket_validado=ticket_validado,
        )
        return {
            "precio_base_mercado": res["valor_nuevo_estimado"],
            "precio_renta_sugerido": res["precio_renta_sugerido"],
            "precio_renta_minimo": res["precio_renta_minimo"],
            "valor_real_depreciado": res["valor_depreciado_estimado"],
            "tope_cobertura_garantia": res["tope_cooperativo_garantia"],
            "deducible_garantia_sugerido": res["deducible_garantia_sugerido"],
            "requiere_revision_manual": res["requiere_revision_manual"],
            "detalles_calculo": {
                "modelo_ml_utilizado": res["modelo_pricing_utilizado"],
                "fuente_precio_catalogo": res["fuente_precio_catalogo"],
                "tipo_herramienta": res["tipo_herramienta"],
                "sector_uso": res["sector_uso"],
            },
        }
    except Exception as e:
        logger.error(f"Error en auto-valuate: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculando la valuación: {str(e)}"
        )

@router.get("/suggest-price")
async def suggest_price(
    precio_base: float = Query(..., description="Valor estimado ya conocido de la herramienta"),
    score_condicion: float = Query(..., description="Score de desgaste continuo [0.40 - 1.0]"),
    sector: str = Query("Manual", description="Categoría de uso"),
    marca: str = Query("Generica", description="Marca fabricante"),
    nombre_herramienta: str = Query("", description="Nombre comercial de la herramienta"),
    age_months: int = Query(12, description="Edad de la herramienta en meses"),
):
    """Recalcula renta sugerida y garantía para un precio_base ya conocido.

    Lo usa Api_Apptransacional/internal/tool/service/tool_service.go
    (GetPricingSuggestion) cuando el valor estimado ya está definido y solo
    se necesita la tarifa de renta y el tope de garantía correspondientes.
    """
    try:
        res = calcular_pricing_motor(
            nombre_herramienta=nombre_herramienta or sector,
            score_condicion=score_condicion,
            sector=sector,
            marca=marca,
            age_months=age_months,
            precio_base_manual=precio_base,
            ticket_validado=True,
        )
        return {
            "valor_real_depreciado": res["valor_depreciado_estimado"],
            "tope_cobertura_garantia": res["tope_cooperativo_garantia"],
            "precio_renta_sugerido": res["precio_renta_sugerido"],
        }
    except Exception as e:
        logger.error(f"Error en suggest-price: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculando el precio sugerido: {str(e)}"
        )
