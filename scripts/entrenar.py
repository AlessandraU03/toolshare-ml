# scripts/entrenar.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras import layers, models
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
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
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
    callbacks=callbacks
)

# ---- Guardar modelo final ----
model.save("models/baseline_final.keras")

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

print("\nEntrenamiento completo. Modelo guardado en models/baseline_final.keras")