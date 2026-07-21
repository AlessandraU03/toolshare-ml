# scripts/entrenar.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
CLASES = ["nuevo", "uso_moderado", "viejo_desgastado"]

# ---- Carga de datos ----
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASES,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=CLASES,
    shuffle=False
)

# ---- Augmentation ----
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
])

# ---- Preprocesamiento (normalización MobileNetV3) ----
preprocess = tf.keras.applications.mobilenet_v3.preprocess_input

# ---- Modelo base ----
base_model = MobileNetV3Small(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # congelado en fase baseline

# ---- Modelo completo ----
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(len(CLASES), activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---- Pesos de clase (compensa el desbalance entre nuevo/uso_moderado/viejo) ----
y_list = []
for idx, clase in enumerate(CLASES):
    clase_path = os.path.join("dataset/train", clase)
    if os.path.exists(clase_path):
        num_files = len([f for f in os.listdir(clase_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        y_list.extend([idx] * num_files)

y_train = np.array(y_list)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(CLASES)),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# ---- Callbacks ----
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(
        "models/baseline_best.keras",
        save_best_only=True,
        monitor="val_accuracy"
    )
]

# ---- Entrenamiento ----
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# ---- Guardar modelo final ----
# "baseline_final.keras" queda como respaldo con nombre descriptivo, pero el
# nombre que realmente carga la API (api/routes_tool.py) es
# "modelo_desgaste.keras" — se guarda también ahí para que el modelo quede
# desplegado de inmediato sin un paso manual de renombrado. Si después corres
# fine_tuning.py, ese script vuelve a sobreescribir "modelo_desgaste.keras"
# con la versión afinada (mejor que esta).
model.save("models/baseline_final.keras")
model.save("models/modelo_desgaste.keras")

# ---- Graficar resultados ----
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.savefig("models/baseline_history.png")
plt.show()

print("\nEntrenamiento completo. Modelo guardado en models/baseline_final.keras y models/modelo_desgaste.keras (deploy)")