import io
import re
import logging
from typing import Optional
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("toolshare-ml")

# El primer patrón exige coma de miles ("1,392.00"); el segundo acepta
# cualquier corrida de dígitos sin coma ("1200.00"), muy común en tickets
# mexicanos. Antes de este fix, un monto de 4+ dígitos sin coma (ej. 1200.00)
# se truncaba a solo sus primeros 3 dígitos (120).
MONTO_REGEX = re.compile(r"(?:[\$\d\s]{1,3}(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})?)")
PALABRAS_TOTAL = (
    "total", "importe", "monto a pagar", "gran total", "neto", 
    "subtotal", "cargo", "pagar", "pago", "venta", "efectivo",
    "tarjeta", "cambio", "total mxn"
)


def _preprocesar_ticket_para_ocr(imagen: Image.Image) -> Image.Image:
    """Misma técnica usada para la credencial INE (kyc/ocr_scanner.py): un
    ticket fotografiado con celular rara vez tiene suficiente contraste/
    resolución para que Tesseract lea bien los montos sin ayuda."""
    gris = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2GRAY)
    alto, ancho = gris.shape[:2]
    escala = 2 if max(alto, ancho) < 1600 else 1
    if escala != 1:
        gris = cv2.resize(gris, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
    suavizada = cv2.bilateralFilter(gris, 9, 75, 75)
    binaria = cv2.adaptiveThreshold(
        suavizada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return Image.fromarray(binaria)


def _texto_a_monto(texto: str) -> Optional[float]:
    # Limpiamos símbolos comunes que Tesseract confunde con dinero
    limpio = re.sub(r"[^\d\.]", "", texto)
    try:
        valor = float(limpio)
    except ValueError:
        return None
    
    # Ignorar números que parezcan años (2020-2030), códigos postales de 5 dígitos (e.g. 30500)
    # o números excesivamente grandes o pequeños para una herramienta.
    if 2018 <= valor <= 2035:
        return None
    if 10000 <= valor <= 99999 and "." not in texto: # Códigos postales típicos
        return None
    if valor < 10 or valor > 150000: # Precios absurdos de herramientas
        return None
        
    return valor


def extraer_precio_de_ticket(imagen_bytes: bytes) -> dict:
    """Extrae el precio de compra declarado en un ticket/factura mediante OCR."""
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
        imagen_procesada = _preprocesar_ticket_para_ocr(imagen)
        texto_completo = pytesseract.image_to_string(imagen_procesada)
        logger.info(f"Ticket - Texto crudo del OCR ({len(texto_completo)} chars): {texto_completo!r}")
    except Exception as e:
        logger.error(f"Error al procesar el ticket con OCR: {e}")
        return {
            "valid": False,
            "error": "No se pudo leer la imagen del ticket. Verifica el formato o la resolución."
        }

    montos_candidatos = []
    lineas = texto_completo.splitlines()
    
    for i, linea in enumerate(lineas):
        linea_lower = linea.lower()
        if any(palabra in linea_lower for palabra in PALABRAS_TOTAL):
            # 1. Buscamos montos en la misma línea
            for coincidencia in MONTO_REGEX.findall(linea):
                monto = _texto_a_monto(coincidencia)
                if monto is not None:
                    montos_candidatos.append((monto, "alta"))
            
            # 2. Si no hay monto en la misma línea, escaneamos la línea inmediata de abajo
            # (Muy común en tickets de formato angosto donde el precio se imprime abajo de la etiqueta)
            if i + 1 < len(lineas):
                siguiente_linea = lineas[i + 1]
                for coincidencia in MONTO_REGEX.findall(siguiente_linea):
                    monto = _texto_a_monto(coincidencia)
                    if monto is not None:
                        montos_candidatos.append((monto, "alta"))

    # Fallback general si no encontramos palabras clave de total
    if not montos_candidatos:
        for coincidencia in MONTO_REGEX.findall(texto_completo):
            monto = _texto_a_monto(coincidencia)
            if monto is not None:
                montos_candidatos.append((monto, "baja"))

    if not montos_candidatos:
        logger.warning("Ticket - No se encontró ningún monto candidato en el texto del OCR.")
        return {
            "valid": False,
            "error": "No se detectó ningún monto legible en el ticket."
        }

    # Ordenamos priorizando candidatos de "alta" confianza, y luego por monto mayor lógico
    montos_candidatos.sort(key=lambda item: (0 if item[1] == "alta" else 1, -item[0]))
    precio_detectado, confianza = montos_candidatos[0]
    logger.info(f"Ticket - Candidatos: {montos_candidatos}, elegido: {precio_detectado} ({confianza})")

    return {
        "valid": True,
        "precio_detectado": round(precio_detectado, 2),
        "confianza": confianza,
        "texto_ocr": texto_completo.strip()[:500],
    }
