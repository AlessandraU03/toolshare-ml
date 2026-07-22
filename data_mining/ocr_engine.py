"""Motor de OCR compartido (KYC + tickets de compra).

Reemplaza a Tesseract: PaddleOCR es un motor de detección+reconocimiento
(no un LLM de visión) — pesa decenas de MB, corre en CPU sin problema y es
notablemente más robusto que Tesseract en fotos de celular con luz
despareja o texto chico, que es justo el caso de una credencial INE o un
ticket fotografiado a mano.

Corre en un worker persistente, en un proceso aparte (ver ocr_worker.py),
en vez de en el proceso principal: api/routes_tool.py carga TensorFlow al
arrancar (para el CNN de desgaste), y construir PaddleOCR en ESE MISMO
proceso con TensorFlow ya cargado produce un Segmentation fault en
producción — se confirmó que el mismo init funciona perfecto en un proceso
limpio (el bake-in del Dockerfile durante el build), así que el problema es
el choque de núcleos nativos entre ambos frameworks compartiendo un
proceso, no PaddleOCR en sí.

Es un worker PERSISTENTE (no uno nuevo por request) porque importar
paddleocr + cargar el modelo desde cero tarda decenas de segundos; hacerlo
una sola vez al arrancar el worker y reusar el mismo proceso para cada OCR
mantiene cada request individual en 1-3s.
"""
import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
from PIL import Image

logger = logging.getLogger("toolshare-ml")

_WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_worker.py")
_TIMEOUT_SEGUNDOS = 45

_proc = None
_lock = threading.Lock()


def _readline_con_timeout(stream, timeout: float):
    """readline() normal se queda colgado para siempre si el worker no
    responde; esto le pone un límite real (usa un hilo porque no hay forma
    portable de ponerle timeout a la lectura de un pipe en Windows/Linux por
    igual sin select, que no aplica a pipes de Windows)."""
    resultado: "queue.Queue[str]" = queue.Queue(maxsize=1)

    def _leer():
        try:
            resultado.put(stream.readline())
        except Exception:
            resultado.put("")

    hilo = threading.Thread(target=_leer, daemon=True)
    hilo.start()
    try:
        return resultado.get(timeout=timeout)
    except queue.Empty:
        return None


def _start_worker():
    logger.info("Arrancando worker de PaddleOCR (proceso aislado)...")
    proc = subprocess.Popen(
        [sys.executable, "-u", _WORKER_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    ready_line = _readline_con_timeout(proc.stdout, _TIMEOUT_SEGUNDOS)
    ready = None
    try:
        ready = json.loads(ready_line) if ready_line else None
    except json.JSONDecodeError:
        ready = None
    if not ready or not ready.get("ok"):
        stderr_tail = proc.stderr.read(4000) if proc.stderr else ""
        proc.kill()
        raise RuntimeError(f"El worker de OCR no arrancó correctamente: {stderr_tail}")
    logger.info("Worker de PaddleOCR listo.")
    return proc


def _pedir_ocr(proc, ruta_imagen: str) -> str:
    proc.stdin.write(ruta_imagen + "\n")
    proc.stdin.flush()
    linea = _readline_con_timeout(proc.stdout, _TIMEOUT_SEGUNDOS)
    if linea is None:
        # Se agotó el tiempo esperando: el worker quedó en un estado
        # inservible (a medio procesar), se mata para forzar que la
        # siguiente llamada arranque uno nuevo.
        proc.kill()
        raise TimeoutError(f"El worker de OCR no respondió en {_TIMEOUT_SEGUNDOS}s.")
    return linea


def precalentar():
    """Arranca el worker de PaddleOCR de una vez (llamar al iniciar el
    servidor, en un hilo aparte, para que la primera petición real de un
    usuario ya lo encuentre caliente en vez de pagar el costo de arranque
    ella misma)."""
    global _proc
    with _lock:
        if _proc is None or _proc.poll() is not None:
            try:
                _proc = _start_worker()
            except Exception as e:
                logger.error(f"No se pudo precalentar el worker de OCR: {e}")


def extraer_texto(imagen: Image.Image) -> str:
    """Corre OCR sobre una imagen PIL (vía el worker persistente) y devuelve
    el texto reconocido, una línea por renglón detectado (mismo formato que
    antes producía pytesseract.image_to_string, para no tener que tocar la
    lógica de extracción de montos/clave de elector que ya existe)."""
    global _proc

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        imagen.convert("RGB").save(tmp_path)

        with _lock:
            if _proc is None or _proc.poll() is not None:
                _proc = _start_worker()
            try:
                linea = _pedir_ocr(_proc, tmp_path)
            except (BrokenPipeError, OSError):
                # El worker murió a medio request: reintenta una vez con uno nuevo.
                _proc = _start_worker()
                linea = _pedir_ocr(_proc, tmp_path)

        if not linea:
            logger.error("El worker de OCR no devolvió respuesta.")
            return ""

        resultado = json.loads(linea)
        if not resultado.get("ok"):
            logger.error(f"OCR worker reportó error: {resultado.get('error')}")
            return ""
        return "\n".join(resultado.get("textos", []))
    except Exception as e:
        logger.error(f"Error al ejecutar OCR: {e}")
        return ""
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
