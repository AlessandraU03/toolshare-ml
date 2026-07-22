FROM python:3.13-slim

# tesseract-ocr: pytesseract solo es un wrapper, necesita el binario del
# sistema para leer tickets/facturas (extract-ticket-price).
# libglib2.0-0: dependencia de opencv-python-headless en tiempo de ejecucion.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
