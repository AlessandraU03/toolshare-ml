# scripts/split_dataset.py
import os
import shutil
import random

random.seed(42)

CLASES = ["nuevo", "uso_moderado", "viejo_desgastado"]
SOURCE_DIR = "dataset_raw"  # pon aquí todas tus imágenes, una carpeta por clase
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
VAL_SPLIT = 0.2

def split():
    for clase in CLASES:
        src = os.path.join(SOURCE_DIR, clase)
        train_dst = os.path.join(TRAIN_DIR, clase)
        val_dst = os.path.join(VAL_DIR, clase)

        # limpiar destinos
        for d in [train_dst, val_dst]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)

        archivos = [f for f in os.listdir(src)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(archivos)

        n_val = int(len(archivos) * VAL_SPLIT)
        val_files = archivos[:n_val]
        train_files = archivos[n_val:]

        for f in train_files:
            shutil.copy(os.path.join(src, f), os.path.join(train_dst, f))
        for f in val_files:
            shutil.copy(os.path.join(src, f), os.path.join(val_dst, f))

        print(f"{clase}: train={len(train_files)} val={len(val_files)}")

if __name__ == "__main__":
    split()