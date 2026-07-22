"""Motor de OCR compartido (KYC + tickets de compra).

Reemplaza a Tesseract: PaddleOCR es un motor de detección+reconocimiento
(no un LLM de visión) — pesa decenas de MB, corre en CPU sin problema y es
notablemente más robusto que Tesseract en fotos de celular con luz
despareja o texto chico, que es justo el caso de una credencial INE o un
ticket fotografiado a mano.

enable_mkldnn=False es obligatorio en este entorno: con la aceleración
oneDNN activada (default), esta versión de paddlepaddle truena con
NotImplementedError al inferir (bug de la combinación paddlepaddle 3.3.1 +
CPU con oneDNN, no del modelo ni de las imágenes).
"""
import logging
from PIL import Image

logger = logging.getLogger("toolshare-ml")

_ocr_engine = None


def _get_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR
        logger.info("Cargando motor PaddleOCR (primera vez, puede tardar)...")
        _ocr_engine = PaddleOCR(
            lang="es",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
        logger.info("PaddleOCR listo.")
    return _ocr_engine


def extraer_texto(imagen: Image.Image) -> str:
    """Corre OCR sobre una imagen PIL y devuelve el texto reconocido, una
    línea por renglón detectado (mismo formato que antes producía
    pytesseract.image_to_string, para no tener que tocar la lógica de
    extracción de montos/clave de elector que ya existe)."""
    import numpy as np

    engine = _get_engine()
    resultados = engine.predict(np.array(imagen.convert("RGB")))
    if not resultados:
        return ""
    textos = resultados[0].get("rec_texts", [])
    return "\n".join(textos)
