import logging

logger = logging.getLogger("toolshare-ml")

class FaceDetector:
    _face_cascade = None

    @classmethod
    def get_face_cascade(cls):
        """Carga de forma diferida (lazy load) el clasificador Haar Cascade de OpenCV."""
        if cls._face_cascade is None:
            try:
                import cv2
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                cls._face_cascade = cv2.CascadeClassifier(cascade_path)
                if cls._face_cascade.empty():
                    logger.error("Error: El clasificador Haar Cascade de rostros está vacío o corrupto.")
                else:
                    logger.info("Clasificador OpenCV Haar Cascade de rostros cargado exitosamente.")
            except ImportError:
                logger.error("Error: OpenCV no está disponible.")
            except Exception as e:
                logger.error(f"Error cargando Haar Cascade de OpenCV: {e}")
        return cls._face_cascade


def verificar_identidad_facial(selfie_bgr, ine_bgr) -> dict:
    """Compara la selfie contra la foto del INE con un embedding facial real
    (ArcFace vía DeepFace) — antes de esto, match_facial_score era un valor
    fijo (0.91/0.82) que solo dependía de si se había detectado ALGUNA cara
    en el INE, sin comparar nunca si era la misma persona de la selfie.

    Devuelve {"verificado": bool, "distancia": float, "umbral": float} o
    {"verificado": False, "error": str} si no se pudo procesar alguna cara.
    """
    from deepface import DeepFace

    try:
        resultado = DeepFace.verify(
            img1_path=selfie_bgr,
            img2_path=ine_bgr,
            model_name="ArcFace",
            detector_backend="opencv",
        )
        return {
            "verificado": bool(resultado["verified"]),
            "distancia": float(resultado["distance"]),
            "umbral": float(resultado["threshold"]),
        }
    except ValueError as e:
        # DeepFace lanza ValueError cuando su propio detector (independiente
        # del Haar Cascade que ya filtró antes) no logra ubicar/alinear una
        # cara utilizable en alguna de las dos imágenes.
        logger.warning(f"DeepFace no pudo procesar alguna de las caras: {e}")
        return {"verificado": False, "error": "No se pudo procesar el rostro con suficiente claridad."}
