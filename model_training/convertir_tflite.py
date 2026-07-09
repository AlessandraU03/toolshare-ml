# scripts/convertir_tflite.py
import tensorflow as tf
import numpy as np
from PIL import Image

# ---- Cargar el modelo baseline (el bueno) ----
model = tf.keras.models.load_model("models/baseline_best.keras")

CLASES = ["nuevo", "uso_moderado", "viejo_desgastado"]
PESOS = {"nuevo": 1.0, "uso_moderado": 0.70, "viejo_desgastado": 0.40}

# ---- Función score_condicion ----
def calcular_score(probs):
    return sum(probs[i] * PESOS[CLASES[i]] for i in range(len(CLASES)))

# ---- Prueba con una imagen ----
def predecir(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

    probs = model.predict(arr, verbose=0)[0]
    score = calcular_score(probs)

    print(f"\nImagen: {img_path}")
    for i, clase in enumerate(CLASES):
        print(f"  {clase}: {probs[i]:.3f}")
    print(f"  -> score_condicion: {score:.3f}")
    return score

# ---- Conversión a TFLite ----
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("models/modelo_condicion.tflite", "wb") as f:
    f.write(tflite_model)

print("\nModelo .tflite guardado en models/modelo_condicion.tflite")

# ---- Prueba con TFLite ----
def predecir_tflite(img_path):
    interpreter = tf.lite.Interpreter(model_path="models/modelo_condicion.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]["index"])[0]

    score = calcular_score(probs)
    print(f"\n[TFLite] {img_path}")
    for i, clase in enumerate(CLASES):
        print(f"  {clase}: {probs[i]:.3f}")
    print(f"  -> score_condicion: {score:.3f}")
    return score

if __name__ == "__main__":
    # Cambia esta ruta por una imagen de prueba (de tu dataset/val por ejemplo)
    test_img = "dataset/val/viejo_desgastado/f4290117-a532-4f79-b0f0-df48d2cf66a4_jpg.rf.IOTt1T5WejOSjPrmRBFb.jpg"
    predecir(test_img)
    predecir_tflite(test_img)