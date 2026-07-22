FROM python:3.13-slim

# libglib2.0-0: dependencia de opencv-python-headless en tiempo de ejecucion.
# (tesseract-ocr ya no hace falta: PaddleOCR reemplazo a pytesseract y es
# puro Python + pesos de modelo, no necesita ningun binario de sistema).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Descarga los pesos de PaddleOCR durante el build, no en el primer
# request: en produccion la primera llamada tardaba 2+ minutos bajando
# los modelos (visto localmente), lo que se hubiera visto como un timeout
# o un servicio "colgado" en Railway.
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='es', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, enable_mkldnn=False)"

# Mismo motivo: los pesos de ArcFace (verificacion facial KYC) pesan ~137MB
# y se descargan de github.com/serengil/deepface_models la primera vez que
# se usan. Se hornean aqui para que el primer KYC real no tarde minutos.
RUN python -c "from deepface import DeepFace; DeepFace.build_model('ArcFace')"

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
