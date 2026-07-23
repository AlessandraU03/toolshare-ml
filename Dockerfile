FROM python:3.13-slim

# libglib2.0-0: dependencia de opencv-python-headless en tiempo de ejecucion.
# libgl1/libsm6/libxext6/libxrender1: PaddleOCR/PaddleX traen opencv-python
# completo (no el -headless que usamos nosotros) como dependencia transitiva,
# y ese necesita estas librerias graficas del sistema aunque nunca se abra
# ninguna ventana -- sin ellas truena con "ImportError: libGL.so.1: cannot
# open shared object file" al importar paddleocr.
# libgomp1: runtime de OpenMP que necesita el nucleo compilado de
# paddlepaddle (libpaddle.so) -- sin el truena con "ImportError:
# libgomp.so.1: cannot open shared object file" al importar paddle.
# (tesseract-ocr ya no hace falta: PaddleOCR reemplazo a pytesseract y es
# puro Python + pesos de modelo, no necesita ningun binario de sistema).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Descarga los pesos de PaddleOCR durante el build, no en el primer
# request: en produccion la primera llamada tardaba 2+ minutos bajando
# los modelos (visto localmente), lo que se hubiera visto como un timeout
# o un servicio "colgado" en Railway.
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(text_detection_model_name='PP-OCRv6_small_det', text_recognition_model_name='PP-OCRv6_small_rec', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, enable_mkldnn=False)"

# Mismo motivo: los pesos de ArcFace (verificacion facial KYC) pesan ~137MB
# y se descargan de github.com/serengil/deepface_models la primera vez que
# se usan. Se hornean aqui para que el primer KYC real no tarde minutos.
RUN python -c "from deepface import DeepFace; DeepFace.build_model('ArcFace')"

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
