import io
import re
import logging
from typing import Optional
from PIL import Image

logger = logging.getLogger("toolshare-ml")

MONTO_REGEX = re.compile(r"\$?\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")
PALABRAS_TOTAL = ("total", "importe", "monto a pagar", "gran total")


def _texto_a_monto(texto: str) -> Optional[float]:
    limpio = texto.replace(",", "")
    try:
        valor = float(limpio)
    except ValueError:
        return None
    if valor <= 0:
        return None
    return valor


def extraer_precio_de_ticket(imagen_bytes: bytes) -> dict:
    """Extrae el precio de compra declarado en un ticket/factura mediante OCR.

    A diferencia del OCR de INE en kyc/ocr_scanner.py (que tiene un mecanismo
    de contingencia porque solo valida un formato de texto), aquí no se
    fabrica ningún precio si el OCR falla: un monto inventado afectaría
    directamente el tope de la garantía cooperativa, así que ante duda se
    reporta valid=False y se deja la decisión a las fuentes de precio del
    catálogo semilla.
    """
    try:
        import pytesseract
    except ImportError:
        logger.warning("pytesseract no está disponible en este entorno.")
        return {
            "valid": False,
            "error": "El servicio de lectura de tickets no está disponible en este momento."
        }

    try:
        imagen = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
        texto_completo = pytesseract.image_to_string(imagen)
    except Exception as e:
        logger.error(f"Error al procesar el ticket con OCR: {e}")
        return {
            "valid": False,
            "error": "No se pudo leer la imagen del ticket. Verifica el formato o la resolución."
        }

    montos_candidatos = []
    for linea in texto_completo.splitlines():
        linea_lower = linea.lower()
        if any(palabra in linea_lower for palabra in PALABRAS_TOTAL):
            for coincidencia in MONTO_REGEX.findall(linea):
                monto = _texto_a_monto(coincidencia)
                if monto is not None:
                    montos_candidatos.append((monto, "alta"))

    if not montos_candidatos:
        for coincidencia in MONTO_REGEX.findall(texto_completo):
            monto = _texto_a_monto(coincidencia)
            if monto is not None:
                montos_candidatos.append((monto, "baja"))

    if not montos_candidatos:
        return {
            "valid": False,
            "error": "No se detectó ningún monto legible en el ticket."
        }

    montos_candidatos.sort(key=lambda item: item[0], reverse=True)
    precio_detectado, confianza = montos_candidatos[0]

    return {
        "valid": True,
        "precio_detectado": round(precio_detectado, 2),
        "confianza": confianza,
        "texto_ocr": texto_completo.strip()[:500],
    }
