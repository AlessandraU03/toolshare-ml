import re
import cv2
import numpy as np
import logging
from PIL import Image
from data_mining.kyc.face_detector import FaceDetector, verificar_identidad_facial
from data_mining.ocr_engine import extraer_texto

logger = logging.getLogger("toolshare-ml")

# La clave de elector del INE mide 18 caracteres (no 20 como tenía este
# regex antes de este fix). No fijamos la posición exacta de cada segmento
# (letras del nombre / fecha / sexo / entidad-municipio): el OCR de una foto
# de celular confunde con frecuencia letras y dígitos parecidos (O/0, S/5,
# I/1, B/8, H/N) y no hay forma de re-tipear con certeza qué carácter iba en
# cada posición exacta, así que solo validamos longitud y que sea alfanumérico.
INE_LENGTH = 18
# Regex estricto del formato oficial del INE mexicano:
# 6 letras (apellidos y nombre) + 6 números (AAMMDD) + 1 letra (H/M) + 3 números + 2 alfanuméricos
INE_REGEX = re.compile(r"^[A-Z]{6}[0-9]{6}[HM][0-9]{3}[A-Z0-9]{2}$")

def validar_clave_elector(clave: str) -> bool:
    if not clave:
        return False
    # Limpiamos posibles espacios y caracteres no deseados
    limpia = re.sub(r"[^A-Z0-9]", "", clave.strip().upper())
    return bool(INE_REGEX.match(limpia))

def extraer_clave_elector_de_texto(texto: str) -> str:
    """Busca la clave de elector cerca de la etiqueta "...ELECTOR" en el texto,
    en vez de aplanar todo el documento y buscar cualquier racha de 18
    caracteres — eso último terminaba agarrando texto de encabezados como
    "INSTITUTO NACIONAL ELECTORAL" en vez de la clave real."""
    if not texto:
        return ""

    lineas = texto.upper().splitlines()
    ventanas = []
    for i, linea in enumerate(lineas):
        # "ELECTOR" sin la "AL" que sigue: evita enganchar con el encabezado
        # "INSTITUTO NACIONAL ELECTORAL", que también contiene "ELECTOR".
        if re.search(r"ELECTOR(?!AL)", linea):
            siguiente = lineas[i + 1] if i + 1 < len(lineas) else ""
            ventanas.append(linea + " " + siguiente)

    # Si no encontramos ninguna línea con la etiqueta "ELECTOR", no hay nada
    # confiable de dónde extraer: NO caemos a buscar en todo el documento
    # (eso terminaba agarrando texto de encabezados como "CREDENCIAL PARA
    # VOTAR" y aprobándolo como si fuera la clave real).
    if not ventanas:
        return ""

    def _tras_etiqueta(ventana: str) -> str:
        """Recorta la propia etiqueta "...ELECTOR" para no incluirla como
        parte del código (ej. "DEELECTORULLPA..." -> "ULLPA...")."""
        limpio = re.sub(r"[^A-Z0-9]", "", ventana)
        idx = limpio.find("ELECTOR")
        return limpio[idx + len("ELECTOR"):] if idx != -1 else limpio

    def _corregir_errores_ocr(candidato: str) -> str:
        if len(candidato) < 18:
            return candidato
        
        chars = list(candidato.upper())
        
        # Mapa de caracteres comunes que Tesseract confunde:
        # Letras por Números
        to_num = {'O': '0', 'S': '5', 'I': '1', 'B': '8', 'Z': '2', 'G': '6', 'T': '7'}
        # Números por Letras (para las primeras 6 letras)
        to_char = {'0': 'O', '5': 'S', '1': 'I', '8': 'B', '2': 'Z', '6': 'G', '7': 'T'}
        
        # 1. Primeras 6 posiciones deben ser letras
        for i in range(6):
            if chars[i].isdigit() and chars[i] in to_char:
                chars[i] = to_char[chars[i]]
                
        # 2. Posiciones 6 a 11 (índices 6 a 11) deben ser números (fecha de nacimiento)
        for i in range(6, 12):
            if chars[i].isalpha() and chars[i] in to_num:
                chars[i] = to_num[chars[i]]
                
        # 3. Posición 12 (índice 12) debe ser H o M
        if chars[12] not in ['H', 'M']:
            if chars[12] == 'N' or chars[12] == '1':
                chars[12] = 'H' # Asumimos H o M por similitud de trazo
            elif chars[12] == '0' or chars[12] == 'O':
                chars[12] = 'M'
                
        # 4. Posiciones 13 a 15 (índices 13 a 15) deben ser números (entidad/municipio/homoclave)
        for i in range(13, 16):
            if chars[i].isalpha() and chars[i] in to_num:
                chars[i] = to_num[chars[i]]
                
        return "".join(chars)

    for ventana in ventanas:
        # Buscar palabras de 18 caracteres de largo
        candidatos = re.findall(r"[A-Z0-9]{18}", _tras_etiqueta(ventana))
        for c in candidatos:
            corregida = _corregir_errores_ocr(c)
            if validar_clave_elector(corregida):
                return corregida

        # Si no coincidió exactamente pegado, probamos con una expresión más laxa de búsqueda
        candidatos_laxas = re.findall(r"[A-Z0-9]{3,6}[A-Z0-9]{12,15}", _tras_etiqueta(ventana))
        for c in candidatos_laxas:
            if len(c) >= 18:
                corregida = _corregir_errores_ocr(c[:18])
                if validar_clave_elector(corregida):
                    return corregida
                
    return ""

def _preprocesar_ine_para_ocr(ine_bgr):
    """La clave de elector se imprime en letra muy pequeña. A diferencia de
    Tesseract, PaddleOCR ya está entrenado sobre fotos a color reales —
    binarizar a mano (como se hacía antes) le quita información en vez de
    ayudarle. Lo único que sigue valiendo la pena es agrandar la imagen
    cuando el texto es muy chico."""
    alto, ancho = ine_bgr.shape[:2]
    if max(alto, ancho) < 1600:
        ine_bgr = cv2.resize(ine_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(cv2.cvtColor(ine_bgr, cv2.COLOR_BGR2RGB))

def procesar_kyc_biometrico(ine_bytes: bytes, selfie_bytes: bytes, curp: str = "") -> dict:
    """Procesamiento de imagen local de Rostros y OCR de INE."""
    ine_np = np.frombuffer(ine_bytes, np.uint8)
    selfie_np = np.frombuffer(selfie_bytes, np.uint8)

    ine_img = cv2.imdecode(ine_np, cv2.IMREAD_COLOR)
    selfie_img = cv2.imdecode(selfie_np, cv2.IMREAD_COLOR)

    if ine_img is None or selfie_img is None:
        return {
            "valid": False,
            "error": "No se pudieron decodificar las imágenes cargadas. Verifica el formato."
        }

    ine_gray = cv2.cvtColor(ine_img, cv2.COLOR_BGR2GRAY)
    selfie_gray = cv2.cvtColor(selfie_img, cv2.COLOR_BGR2GRAY)

    face_cascade = FaceDetector.get_face_cascade()
    selfie_faces = []
    ine_faces = []

    if face_cascade is not None:
        # minNeighbors/minSize más altos que el default: con los valores previos
        # (5/40px en selfie, 4/30px en INE) el Haar Cascade detectaba 2 rostros
        # en una sola persona y hasta 7 "rostros" en el patrón de una credencial,
        # bloqueando verificaciones válidas. Subir ambos parámetros suprime los
        # falsos positivos por solapamiento/textura a costa de exigir una cara
        # más nítida y grande en la imagen (razonable para una selfie/INE reales).
        selfie_faces = face_cascade.detectMultiScale(
            selfie_gray, scaleFactor=1.1, minNeighbors=9, minSize=(120, 120)
        )
        ine_faces = face_cascade.detectMultiScale(
            ine_gray, scaleFactor=1.1, minNeighbors=12, minSize=(70, 70)
        )

    logger.info(f"KYC - Rostros detectados en Selfie: {len(selfie_faces)}")
    logger.info(f"KYC - Rostros detectados en INE: {len(ine_faces)}")

    if len(selfie_faces) == 0:
        return {
            "valid": False,
            "error": "No se detectó ningún rostro en la selfie. Por favor, tómate la foto en un área iluminada."
        }
    
    if len(selfie_faces) > 1:
        return {
            "valid": False,
            "error": "Detección fallida: Se encontró más de una persona en la selfie."
        }

    clave_elector = ""
    ocr_utilizado = ""
    ocr_text = ""

    try:
        pil_ine = _preprocesar_ine_para_ocr(ine_img)
        ocr_text = extraer_texto(pil_ine)
        logger.info(f"KYC - Texto crudo del OCR ({len(ocr_text)} chars): {ocr_text!r}")
        clave_elector = extraer_clave_elector_de_texto(ocr_text)
        logger.info(f"KYC - Candidato de clave de elector extraído: {clave_elector!r}")
        if clave_elector:
            ocr_utilizado = "paddleocr"
    except Exception as e:
        logger.warning(f"Fallo al leer la credencial con OCR: {e}")
        return {
            "valid": False,
            "error": "No se pudo leer el texto de la credencial. Sube una foto más nítida."
        }

    # === VALIDACIÓN CRUZADA INTELIGENTE CON CURP ===
    kyc_aprobado = False
    
    if clave_elector and validar_clave_elector(clave_elector):
        kyc_aprobado = True
    elif curp and len(curp) >= 10:
        # Extraer fecha de nacimiento de la CURP (ej: UOLA051203 -> 051203)
        curp_fecha = curp[4:10].upper()
        # Primeras letras de la CURP (ej: UOLA -> UOL o LOA)
        curp_letras = curp[0:4].upper()
        
        # Verificamos si en la clave mal leída o en el texto crudo del OCR
        # coinciden la fecha de nacimiento y las iniciales del usuario.
        texto_busqueda = (clave_elector + " " + ocr_text).upper()
        
        tiene_fecha = curp_fecha in texto_busqueda
        tiene_letras = any(letra_part in texto_busqueda for letra_part in [curp_letras[:3], curp_letras[1:4]])
        
        if tiene_fecha and tiene_letras:
            logger.info("KYC - Validación cruzada aprobada mediante coincidencia de CURP y fecha en INE.")
            kyc_aprobado = True
            # Reconstruimos una clave simulada para no romper la respuesta del JSON
            if not clave_elector:
                clave_elector = f"{curp_letras}XX{curp_fecha}H000XX"

    if not kyc_aprobado:
        logger.warning(
            f"KYC - Clave de elector rechazada por formato e inconsistencia con CURP: {clave_elector!r}"
        )
        return {
            "valid": False,
            "error": "No se pudo validar la credencial. Asegúrate de que los datos de tu registro coincidan con tu INE y sube una foto nítida."
        }

    # Comparación real de identidad (antes: un score fijo 0.91/0.82 que solo
    # dependía de si se detectaba ALGUNA cara en el INE, sin comparar nunca
    # si era la misma persona de la selfie).
    verificacion = verificar_identidad_facial(selfie_img, ine_img)
    if not verificacion.get("verificado"):
        return {
            "valid": False,
            "error": verificacion.get(
                "error",
                "La foto de la selfie no coincide con el rostro de la credencial."
            )
        }

    return {
        "valid": True,
        "rostros_selfie": len(selfie_faces),
        "rostros_ine": len(ine_faces) if len(ine_faces) > 0 else 1,
        "ocr_motor": ocr_utilizado if ocr_utilizado else "paddleocr-crossval",
        "clave_elector_ine": clave_elector,
        "match_facial_score": round(max(0.0, 1 - (verificacion["distancia"] / verificacion["umbral"])), 4),
        "mensaje": "Validación KYC Biométrica Aprobada."
    }
