# scripts/verificar_dataset.py
import os
from PIL import Image

BASE_DIR = "dataset"
CLASES = ["nuevo", "uso_moderado", "viejo_desgastado"]
SPLITS = ["train", "val"]

def verificar():
    total_general = 0
    for split in SPLITS:
        print(f"\n--- {split.upper()} ---")
        for clase in CLASES:
            path = os.path.join(BASE_DIR, split, clase)
            if not os.path.exists(path):
                print(f"  {clase}: carpeta no existe")
                continue

            archivos = [f for f in os.listdir(path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            corruptas = []

            for f in archivos:
                try:
                    img = Image.open(os.path.join(path, f))
                    img.verify()
                except Exception:
                    corruptas.append(f)

            print(f"  {clase}: {len(archivos)} imágenes "
                  f"({len(corruptas)} corruptas)")
            if corruptas:
                print(f"    -> {corruptas}")

            total_general += len(archivos)

    print(f"\nTotal de imágenes: {total_general}")
    print("Recomendado: mínimo ~100-150 por clase en train")

if __name__ == "__main__":
    verificar()