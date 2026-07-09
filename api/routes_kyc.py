import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from data_mining.kyc.ocr_scanner import procesar_kyc_biometrico

logger = logging.getLogger("toolshare-ml")

router = APIRouter()

@router.post("/verify-kyc")
async def verify_kyc(
    ine_image: UploadFile = File(..., description="Foto frontal de la credencial INE"),
    selfie_image: UploadFile = File(..., description="Foto selfie del rostro del usuario")
):
    """Verifica la identidad del usuario comparando rostros e interpretando el INE mediante OCR."""
    try:
        ine_bytes = await ine_image.read()
        selfie_bytes = await selfie_image.read()

        res = procesar_kyc_biometrico(ine_bytes, selfie_bytes)
        
        if not res.get("valid", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=res.get("error", "Validación KYC fallida.")
            )

        return res
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error procesando verificación KYC: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en el servidor de verificación biométrica: {str(e)}"
        )
