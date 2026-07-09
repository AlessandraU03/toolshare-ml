import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes_tool import router as tool_router
from api.routes_kyc import router as kyc_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("toolshare-ml")

app = FastAPI(
    title="ToolShare ML & Data Mining API",
    description="Microservicio de minería de datos y visión computacional para la plataforma ToolShare.",
    version="1.1.0"
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
