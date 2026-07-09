# scripts/evaluar_completo.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize

CLASES = ["nuevo", "uso_moderado", "viejo_desgastado"]
PESOS  = [1.0, 0.70, 0.40]
IMG_SIZE = (224, 224)

# ---- Cargar modelo y dataset ----
model = tf.keras.models.load_model("models/baseline_best.keras")

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/val",
    image_size=IMG_SIZE,
    batch_size=16,
    label_mode="categorical",
    class_names=CLASES,
    shuffle=False
)

# ---- Inferencia ----
y_true, y_probs = [], []
for images, labels in val_ds:
    probs = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_probs.extend(probs)

y_true  = np.array(y_true)
y_probs = np.array(y_probs)
y_pred  = np.argmax(y_probs, axis=1)

# ---- 1. Matriz de confusión ----
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=CLASES, yticklabels=CLASES,
            cmap='Blues')
plt.title("Matriz de Confusión — Baseline")
plt.ylabel("Clase Real")
plt.xlabel("Clase Predicha")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.show()

# ---- 2. Precision, Recall, F1 ----
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=CLASES))

# ---- 3. ROC-AUC multiclase ----
y_bin = label_binarize(y_true, classes=[0, 1, 2])
plt.figure(figsize=(8, 6))
for i, clase in enumerate(CLASES):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
    auc = roc_auc_score(y_bin[:, i], y_probs[:, i])
    plt.plot(fpr, tpr, label=f"{clase} (AUC={auc:.2f})")
plt.plot([0,1],[0,1],'k--', label='Azar (0.50)')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC — One vs Rest")
plt.legend()
plt.tight_layout()
plt.savefig("models/roc_curve.png")
plt.show()

# ---- 4. Distribución score_condicion por clase real ----
scores = np.dot(y_probs, PESOS)
plt.figure(figsize=(8, 5))
for i, clase in enumerate(CLASES):
    idx = np.where(y_true == i)[0]
    plt.hist(scores[idx], alpha=0.6, bins=12, label=clase)
plt.axvline(0.60, color='orange', linestyle='--', label='umbral nuevo/moderado')
plt.axvline(0.40, color='red',    linestyle='--', label='umbral moderado/viejo')
plt.xlabel("score_condicion")
plt.ylabel("Frecuencia")
plt.title("Distribución de score_condicion por clase real")
plt.legend()
plt.tight_layout()
plt.savefig("models/score_distribucion.png")
plt.show()

# ---- 5. Resumen numérico ----
auc_macro = roc_auc_score(y_bin, y_probs, average='macro')
print(f"\nAUC macro promedio: {auc_macro:.3f}")
print(f"Confianza promedio predicciones correctas:  "
      f"{y_probs[np.arange(len(y_true)), y_pred][y_pred == y_true].mean():.3f}")
print(f"Confianza promedio predicciones incorrectas: "
      f"{y_probs[np.arange(len(y_true)), y_pred][y_pred != y_true].mean():.3f}")