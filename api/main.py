import logging
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes_tool import router as tool_router, get_cnn_model
from api.routes_kyc import router as kyc_router
from data_mining.ocr_engine import precalentar as precalentar_ocr
from data_mining.kyc.face_detector import precalentar_arcface

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("toolshare-ml")


def _precalentar_todo():
    """Carga los 3 modelos pesados (CNN, worker de PaddleOCR, ArcFace) en
    cuanto arranca el servidor, en vez de dejar que cada uno se cargue solo
    hasta que un usuario real hace la primera petición y se queda
    esperando el arranque en frío. Corre en un hilo aparte para no bloquear
    que el servidor empiece a aceptar peticiones (incluido el health check
    de Railway) mientras calienta."""
    logger.info("Precalentando modelos (CNN, PaddleOCR, ArcFace)...")
    get_cnn_model()
    precalentar_ocr()
    precalentar_arcface()
    logger.info("Modelos precalentados.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_precalentar_todo, daemon=True).start()
    yield


app = FastAPI(
    title="ToolShare ML & Data Mining API",
    description="Microservicio de minería de datos y visión computacional para la plataforma ToolShare.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tool_router)
app.include_router(kyc_router)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "description": "ToolShare ML & Data Mining Microservice (Clean MLOps Architecture)"
    }
