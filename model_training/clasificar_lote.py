# scripts/clasificar_lote.py
"""
Clasifica por estado de desgaste (nuevo / uso_moderado / viejo_desgastado)
todas las imagenes de una carpeta externa (no etiquetada) usando el modelo
ya entrenado, y copia las N mejores candidatas a una clase objetivo a una
carpeta de salida para revision manual.

Usa el mismo preprocesamiento que api/routes_tool.py (resize 224x224, RGB,
sin dividir entre 255) para que las predicciones sean consistentes con las
que ve la API en produccion.

Ejemplo:
    python model_training/clasificar_lote.py \
        --input "C:\\Users\\aless\\Downloads\\archive (1)\\tools-classes" \
        --output "C:\\Users\\aless\\Downloads\\archive (1)\\tools-classes_seminuevo_semiusado" \
        --clase uso_moderado \
        --top 200
"""
import os
import csv
import shutil
import argparse

import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png", ".webp", ".jfif")

# Mismo orden que model_training/entrenar.py y api/routes_tool.py
CLASES = ["nuevo", "uso_moderado", "viejo_desgastado"]
SCORE_MAPPING = {"nuevo": 1.0, "uso_moderado": 0.65, "viejo_desgastado": 0.40}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELOS_CANDIDATOS = [
    os.path.join(BASE_DIR, "models", "modelo_desgaste.keras"),
    os.path.join(BASE_DIR, "models", "finetuned_best.keras"),
    os.path.join(BASE_DIR, "models", "baseline_best.keras"),
]


def cargar_modelo():
    for ruta in MODELOS_CANDIDATOS:
        if os.path.exists(ruta):
            print(f"Cargando modelo: {ruta}")
            return tf.keras.models.load_model(ruta)
    raise FileNotFoundError(
        "No se encontro ningun modelo entrenado en models/. "
        "Corre model_training/entrenar.py primero."
    )


def listar_imagenes(carpeta_entrada):
    rutas = []
    for raiz, _, archivos in os.walk(carpeta_entrada):
        for nombre in archivos:
            if nombre.lower().endswith(EXTENSIONES_VALIDAS):
                rutas.append(os.path.join(raiz, nombre))
    return sorted(rutas)


def cargar_batch(rutas_batch):
    imagenes = []
    rutas_ok = []
    for ruta in rutas_batch:
        try:
            img = Image.open(ruta).convert("RGB").resize(IMG_SIZE)
            imagenes.append(np.array(img, dtype=np.float32))
            rutas_ok.append(ruta)
        except Exception as e:
            print(f"  [omitida, no se pudo abrir] {ruta}: {e}")
    if not imagenes:
        return None, []
    return np.stack(imagenes, axis=0), rutas_ok


def clasificar_carpeta(model, carpeta_entrada):
    rutas = listar_imagenes(carpeta_entrada)
    print(f"Encontradas {len(rutas)} imagenes en {carpeta_entrada}")

    resultados = []
    for i in range(0, len(rutas), BATCH_SIZE):
        batch_rutas = rutas[i:i + BATCH_SIZE]
        batch_array, rutas_ok = cargar_batch(batch_rutas)
        if batch_array is None:
            continue
        probs_batch = model.predict(batch_array, verbose=0)
        for ruta, probs in zip(rutas_ok, probs_batch):
            idx = int(np.argmax(probs))
            clase = CLASES[idx]
            resultados.append({
                "ruta": ruta,
                "tipo_herramienta": os.path.basename(os.path.dirname(ruta)),
                "clase_predicha": clase,
                "confianza": float(probs[idx]),
                "prob_nuevo": float(probs[0]),
                "prob_uso_moderado": float(probs[1]),
                "prob_viejo_desgastado": float(probs[2]),
                "score_condicion": SCORE_MAPPING[clase],
            })
        print(f"  procesadas {min(i + BATCH_SIZE, len(rutas))}/{len(rutas)}")

    return resultados


def guardar_csv(resultados, ruta_csv):
    os.makedirs(os.path.dirname(ruta_csv), exist_ok=True)
    campos = ["ruta", "tipo_herramienta", "clase_predicha", "confianza",
              "prob_nuevo", "prob_uso_moderado", "prob_viejo_desgastado",
              "score_condicion"]
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)
    print(f"\nCSV con todas las predicciones guardado en: {ruta_csv}")


def seleccionar_top_n(resultados, clase_objetivo, top_n):
    prob_key = f"prob_{clase_objetivo}"
    return sorted(resultados, key=lambda r: r[prob_key], reverse=True)[:top_n]


def copiar_seleccion(seleccion, carpeta_salida):
    for r in seleccion:
        subcarpeta = os.path.join(carpeta_salida, r["tipo_herramienta"])
        os.makedirs(subcarpeta, exist_ok=True)
        destino = os.path.join(subcarpeta, os.path.basename(r["ruta"]))
        shutil.copy2(r["ruta"], destino)
    print(f"{len(seleccion)} imagenes copiadas a: {carpeta_salida}")


def resumen_por_tipo(seleccion):
    conteo = {}
    for r in seleccion:
        conteo[r["tipo_herramienta"]] = conteo.get(r["tipo_herramienta"], 0) + 1
    print("\nDistribucion de la seleccion por tipo de herramienta:")
    for tipo, n in sorted(conteo.items(), key=lambda x: -x[1]):
        print(f"  {tipo}: {n}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Carpeta con imagenes sin etiquetar")
    parser.add_argument("--output", required=True, help="Carpeta donde copiar la seleccion")
    parser.add_argument("--clase", default="uso_moderado", choices=CLASES,
                         help="Clase objetivo a priorizar (default: uso_moderado)")
    parser.add_argument("--top", type=int, default=200, help="Numero de imagenes a seleccionar")
    args = parser.parse_args()

    model = cargar_modelo()
    resultados = clasificar_carpeta(model, args.input)

    ruta_csv = os.path.join(args.output, "predicciones_completas.csv")
    guardar_csv(resultados, ruta_csv)

    print("\nConteo global por clase predicha (argmax):")
    for clase in CLASES:
        n = sum(1 for r in resultados if r["clase_predicha"] == clase)
        print(f"  {clase}: {n}")

    seleccion = seleccionar_top_n(resultados, args.clase, args.top)
    copiar_seleccion(seleccion, args.output)
    resumen_por_tipo(seleccion)

    if seleccion:
        peor_prob = seleccion[-1][f"prob_{args.clase}"]
        print(f"\nProbabilidad minima en la seleccion (imagen #{len(seleccion)}): {peor_prob:.3f}")
        if peor_prob < 0.5:
            print("Aviso: algunas imagenes seleccionadas tienen confianza baja "
                  "(<0.5) para la clase objetivo; revisalas manualmente.")


if __name__ == "__main__":
    main()
