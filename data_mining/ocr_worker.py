"""Worker persistente para correr PaddleOCR en un proceso separado.

api/routes_tool.py carga TensorFlow al arrancar (para el CNN de desgaste).
Inicializar PaddleOCR en ESE MISMO proceso, con TensorFlow ya cargado,
produce un Segmentation fault en produccion (confirmado: el mismo init
funciona perfecto en un proceso limpio durante el build del Dockerfile,
sin TensorFlow presente -- es un choque de nucleos nativos entre ambos
frameworks compartiendo el mismo proceso, no un problema de PaddleOCR en
si). Este script corre como un proceso aparte que se queda vivo: arranca
una sola vez (paga el costo de importar paddleocr y cargar el modelo una
sola vez), y despues procesa una imagen por linea que le llega en stdin,
devolviendo el resultado como una linea de JSON en stdout.

Protocolo (una linea = una peticion/respuesta):
  stdin  <- ruta_absoluta_de_imagen
  stdout -> {"ok": true, "textos": [...]}  o  {"ok": false, "error": "..."}
"""
import sys
import json


def main():
    from paddleocr import PaddleOCR
    import numpy as np
    from PIL import Image

    ocr = PaddleOCR(
        lang="es",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )
    print(json.dumps({"ok": True, "ready": True}), flush=True)

    for linea in sys.stdin:
        ruta_imagen = linea.strip()
        if not ruta_imagen:
            continue
        try:
            imagen = Image.open(ruta_imagen).convert("RGB")
            resultados = ocr.predict(np.array(imagen))
            textos = resultados[0].get("rec_texts", []) if resultados else []
            print(json.dumps({"ok": True, "textos": textos}), flush=True)
        except Exception as e:
            print(json.dumps({"ok": False, "error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
