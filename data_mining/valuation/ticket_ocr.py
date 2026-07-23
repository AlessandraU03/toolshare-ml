import io
import re
import logging
from typing import Optional
from PIL import Image
from data_mining.ocr_engine import extraer_texto_con_confianza

logger = logging.getLogger("toolshare-ml")

# El primer patrón exige coma de miles ("1,392.00"); el segundo acepta
# cualquier corrida de dígitos sin coma ("1200.00"), muy común en tickets
# mexicanos. Antes de este fix, un monto de 4+ dígitos sin coma (ej. 1200.00)
# se truncaba a solo sus primeros 3 dígitos (120).
MONTO_REGEX = re.compile(r"(?:[\$\d\s]{1,3}(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})?)")

# Niveles de confianza por especificidad de la palabra clave. "total" es lo
# mas confiable; "tarjeta"/"pago"/"importe" tambien aparecen pegados al
# bloque de metadatos de pago con tarjeta (numero de cuenta, autorizacion,
# afiliacion), que esta lleno de numeros grandes y por eso van al fondo.
PALABRAS_ALTA = ("total", "gran total", "total mxn", "monto a pagar")
PALABRAS_MEDIA = ("neto", "importe")
PALABRAS_BAJA = ("cargo", "pagar", "pago", "venta", "efectivo", "tarjeta", "cambio")
_ORDEN_CONFIANZA = {"alta": 0, "media": 1, "baja": 2, "fallback": 3}

# PaddleOCR devuelve, por cada línea detectada, qué tan seguro está el
# modelo de haber leído bien ese texto. Texto impreso limpio de un ticket
# térmico normalmente da >0.9. Un monto tachado y reescrito a mano (u otra
# alteración física) el modelo lo reconoce con mucha menos confianza porque
# nunca fue entrenado para leer letra manuscrita como si fuera texto
# impreso -- por debajo de este umbral no confiamos en el monto ganador y
# lo tratamos como si no se hubiera encontrado un total válido.
UMBRAL_OCR_SOSPECHOSO = 0.80


def _confianza_de_linea(linea_lower: str) -> Optional[str]:
    # "subtotal" contiene la palabra "total" como substring, asi que se
    # revisa aparte antes que la lista de alta confianza para no confundirse
    # con el total real.
    if "subtotal" in linea_lower:
        return "media"
    if any(p in linea_lower for p in PALABRAS_ALTA):
        return "alta"
    if any(p in linea_lower for p in PALABRAS_MEDIA):
        return "media"
    if any(p in linea_lower for p in PALABRAS_BAJA):
        return "baja"
    return None


def _preprocesar_ticket_para_ocr(imagen: Image.Image) -> Image.Image:
    """A diferencia de Tesseract, PaddleOCR ya está entrenado sobre fotos a
    color reales y binarizar la imagen a mano (como se hacía antes) le quita
    información en vez de ayudarle. Lo único que sigue valiendo la pena es
    agrandar la imagen cuando el texto es muy chico."""
    ancho, alto = imagen.size
    if max(alto, ancho) < 1600:
        factor = 2
        imagen = imagen.resize((ancho * factor, alto * factor), Image.LANCZOS)
    return imagen


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
        imagen = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
        imagen_procesada = _preprocesar_ticket_para_ocr(imagen)
        pares = extraer_texto_con_confianza(imagen_procesada)
        lineas = [texto for texto, _ in pares]
        scores = [score for _, score in pares]
        texto_completo = "\n".join(lineas)
        logger.info(f"Ticket - Texto crudo del OCR ({len(texto_completo)} chars): {texto_completo!r}")
    except Exception as e:
        logger.error(f"Error al procesar el ticket con OCR: {e}")
        return {
            "valid": False,
            "error": "No se pudo leer la imagen del ticket. Verifica el formato o la resolución."
        }

    montos_candidatos = []  # (monto, confianza, orden_de_aparicion, score_ocr)

    for i, linea in enumerate(lineas):
        confianza = _confianza_de_linea(linea.lower())
        if confianza is None:
            continue

        # 1. Buscamos montos en la misma línea
        for coincidencia in MONTO_REGEX.findall(linea):
            monto = _texto_a_monto(coincidencia)
            if monto is not None:
                montos_candidatos.append((monto, confianza, i, scores[i]))

        # 2. Si no hay monto en la misma línea, escaneamos la línea inmediata de abajo
        # (Muy común en tickets de formato angosto donde el precio se imprime abajo de la etiqueta)
        if i + 1 < len(lineas):
            for coincidencia in MONTO_REGEX.findall(lineas[i + 1]):
                monto = _texto_a_monto(coincidencia)
                if monto is not None:
                    montos_candidatos.append((monto, confianza, i, scores[i + 1]))

    # Fallback general si no encontramos palabras clave de total
    if not montos_candidatos:
        for i, linea in enumerate(lineas):
            for coincidencia in MONTO_REGEX.findall(linea):
                monto = _texto_a_monto(coincidencia)
                if monto is not None:
                    montos_candidatos.append((monto, "fallback", i, scores[i]))

    if not montos_candidatos:
        logger.warning("Ticket - No se encontró ningún monto candidato en el texto del OCR.")
        return {
            "valid": False,
            "error": "No se detectó ningún monto legible en el ticket."
        }

    # Priorizamos la palabra clave mas especifica (TOTAL > SUBTOTAL/IMPORTE >
    # TARJETA/PAGO/CAMBIO) y, dentro del mismo nivel, el PRIMER monto que
    # aparece en el ticket -- no el mas grande. El total real siempre
    # aparece antes del bloque de metadatos de la tarjeta (numero de cuenta,
    # autorizacion, afiliacion), que esta lleno de numeros grandes que antes
    # ganaban solo por ser "el mayor".
    montos_candidatos.sort(key=lambda item: (_ORDEN_CONFIANZA[item[1]], item[2]))
    precio_detectado, confianza, _, score_ocr = montos_candidatos[0]
    logger.info(f"Ticket - Candidatos: {montos_candidatos}, elegido: {precio_detectado} ({confianza}, score={score_ocr:.3f})")

    if score_ocr < UMBRAL_OCR_SOSPECHOSO:
        logger.warning(
            f"Ticket - Monto {precio_detectado} descartado por baja confianza de OCR "
            f"({score_ocr:.3f} < {UMBRAL_OCR_SOSPECHOSO}), posible alteración física del ticket."
        )
        return {
            "valid": False,
            "error": "El monto del ticket no se pudo leer con suficiente claridad "
                     "(puede estar tachado, sobrescrito o dañado). Se usará el catálogo de "
                     "referencia y tu herramienta quedará marcada para revisión manual.",
        }

    return {
        "valid": True,
        "precio_detectado": round(precio_detectado, 2),
        "confianza": confianza,
        "texto_ocr": texto_completo.strip()[:500],
    }
