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
