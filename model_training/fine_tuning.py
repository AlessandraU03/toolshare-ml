# scripts/fine_tuning.py
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(BASE_DIR, "dataset")
models_dir = os.path.join(BASE_DIR, "models")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
CLASES = ["nuevo", "uso_moderado", "viejo_desgastado"]

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(dataset_dir, "train"), image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="categorical", class_names=CLASES, shuffle=True
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(dataset_dir, "val"), image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="categorical", class_names=CLASES, shuffle=False
)

# ---- Reconstruir arquitectura idéntica al baseline ----
model = tf.keras.models.load_model(os.path.join(models_dir, "baseline_best.keras"))
base_model = model.get_layer("MobileNetV3Small")
# ---- Descongelar últimas capas de MobileNetV3 ----
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---- Calcular pesos de clase dinámicamente ----
y_list = []
for idx, clase in enumerate(CLASES):
    clase_path = os.path.join(dataset_dir, "train", clase)
    if os.path.exists(clase_path):
        num_files = len([f for f in os.listdir(clase_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        y_list.extend([idx] * num_files)

y_train = np.array(y_list)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2]),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(models_dir, "finetuned_best.weights.h5"),
        save_best_only=True,
        monitor="val_accuracy",
        save_weights_only=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

model.save(os.path.join(models_dir, "finetuned_final.keras"))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy (Fine-tuning)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss (Fine-tuning)")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(models_dir, "finetuned_history.png"))
plt.show()

print(f"Fine-tuning completo. Modelo guardado en {os.path.join(models_dir, 'finetuned_final.keras')}")
