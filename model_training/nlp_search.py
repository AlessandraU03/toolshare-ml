# scripts/nlp_search.py
import re
import os
import pickle
import unicodedata
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

# Lista de stopwords en español
STOPWORDS_ES = {
    'de', 'la', 'en', 'el', 'que', 'y', 'un', 'una', 'con', 'para', 'por', 'es', 'al', 
    'los', 'las', 'un', 'una', 'unos', 'unas', 'este', 'esta', 'estos', 'estas', 
    'del', 'lo', 'como', 'o', 'su', 'sus', 'a', 'para', 'no', 'si'
}

# 1. Función de preprocesamiento personalizada (Alineado con Expresiones Regulares de Clase)
def preprocesar_texto(texto):
    if not isinstance(texto, str):
        return ""
    # Case folding (minúsculas)
    texto = texto.lower()
    # Eliminar acentos y diacríticos
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    # Expresión regular: extraer tokens alfanuméricos de más de 2 caracteres
    tokens = re.findall(r'\b[a-z0-9ñ]{2,}\b', texto)
    # Filtrar stopwords
    tokens_filtrados = [t for t in tokens if t not in STOPWORDS_ES]
    return " ".join(tokens_filtrados)

# 2. Base de datos/catálogo de herramientas de entrenamiento y prueba
catalog_data = [
    # Construcción Ligera (Clase 0)
    {"id": 1, "nombre": "Taladro Inalámbrico Makita 12V", "descripcion": "Taladro atornillador ideal para colgar cuadros, armar muebles y perforaciones sencillas en madera o tablaroca.", "categoria": "Construcción Ligera"},
    {"id": 2, "nombre": "Juego de Destornilladores Truper", "descripcion": "Kit de desarmadores planos y de cruz para reparaciones domésticas de plomería y electricidad básica.", "categoria": "Construcción Ligera"},
    {"id": 3, "nombre": "Cortadora de Césped Truper Eléctrica", "descripcion": "Podadora eléctrica ligera de 1000W ideal para mantenimiento de jardines domésticos pequeños.", "categoria": "Construcción Ligera"},
    {"id": 4, "nombre": "Lijadora Orbital Bosch", "descripcion": "Herramienta de lijado manual para carpintería ligera y manualidades en el hogar.", "categoria": "Construcción Ligera"},
    {"id": 5, "nombre": "Hidrolavadora Kärcher K2", "descripcion": "Limpiadora de alta presión portátil para lavar el coche, fachadas de casa y patios residenciales.", "categoria": "Construcción Ligera"},
    {"id": 6, "nombre": "Flexómetro Truper 5m", "descripcion": "Cinta métrica metálica de precisión para medición doméstica y bricolaje.", "categoria": "Construcción Ligera"},
    
    # Construcción Pesada (Clase 1)
    {"id": 7, "nombre": "Rotomartillo Demoledor DeWalt SDS Max", "descripcion": "Martillo cincelador demoledor industrial de 15 Joules para romper losas de concreto, vigas y pavimentos.", "categoria": "Construcción Pesada"},
    {"id": 8, "nombre": "Generador Eléctrico a Gasolina 5500W", "descripcion": "Planta de luz a gasolina de gran potencia para alimentar obra pesada y herramientas trifásicas.", "categoria": "Construcción Pesada"},
    {"id": 9, "nombre": "Cortadora de Pavimento Cortadora de Disco 14 pulgadas", "descripcion": "Cortadora a gasolina profesional para disco diamantado usada en cortes de asfalto y concreto reforzado.", "categoria": "Construcción Pesada"},
    {"id": 10, "nombre": "Revolvedora de Concreto Truper 1 Saco", "descripcion": "Mezcladora de cemento con motor de 9 HP a gasolina para preparación de mezclas de concreto estructural.", "categoria": "Construcción Pesada"},
    {"id": 11, "nombre": "Soldadora Inversora Industrial 200A", "descripcion": "Máquina de soldar profesional para perfiles de acero pesado y estructuras metálicas de soporte en obras de construcción.", "categoria": "Construcción Pesada"},
    {"id": 12, "nombre": "Esmeriladora Industrial DeWalt 9 pulgadas", "descripcion": "Esmeril de gran tamaño y potencia para desbaste de acero estructural y corte de vigas metálicas pesadas.", "categoria": "Construcción Pesada"}
]

df_catalog = pd.DataFrame(catalog_data)

# Mapear categorías a etiquetas numéricas
# Ligera = 0, Pesada = 1
category_map = {"Construcción Ligera": 0, "Construcción Pesada": 1}
df_catalog['label'] = df_catalog['categoria'].map(category_map)

# 3. Vectorización TF-IDF
vectorizer = TfidfVectorizer(preprocessor=preprocesar_texto)
tfidf_matrix = vectorizer.fit_transform(df_catalog['nombre'] + " " + df_catalog['descripcion'])

# 4. Entrenar y Comparar Clasificadores (Naïve Bayes vs Regresión Logística de clase)
print("=== ENTRENANDO CLASIFICADORES NLP ===")
X_train = tfidf_matrix
y_train = df_catalog['label']

# Modelo A: Naïve Bayes (Clásico de NLP)
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_train)
acc_nb = accuracy_score(y_train, y_pred_nb)

# Modelo B: Regresión Logística
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_train)
acc_lr = accuracy_score(y_train, y_pred_lr)

print(f"Precisión de Naïve Bayes en entrenamiento: {acc_nb * 100:.1f}%")
print(f"Precisión de Regresión Logística en entrenamiento: {acc_lr * 100:.1f}%")

# Dado que es un dataset pequeño de catálogo de referencia, ambos logran 100%. 
# Usaremos Regresión Logística porque calibra mejor las probabilidades de salida.
best_classifier = lr_model
print("Modelo seleccionado para clasificación NLP: Regresión Logística")

# 5. Guardar base de datos y modelos en pickle
os.makedirs("models", exist_ok=True)
save_path = "models/buscador_nlp.pkl"

with open(save_path, 'wb') as f:
    pickle.dump({
        'vectorizer': vectorizer,
        'classifier': best_classifier,
        'tfidf_matrix': tfidf_matrix,
        'df_catalog': df_catalog,
        'category_map': {v: k for k, v in category_map.items()} # inverso para decodificación
    }, f)

print(f"Modelo y catálogo indexado guardados en {save_path}")

# 6. Simulación de una búsqueda con similitud coseno y predicción de categoría
def realizar_busqueda(query):
    print(f"\nConsulta del usuario: '{query}'")
    
    # Preprocesar query
    query_processed = preprocesar_texto(query)
    print(f"  Texto preprocesado: '{query_processed}'")
    
    # Vectorizar query
    query_vec = vectorizer.transform([query])
    
    # 1. Predecir categoría (Ligera vs Pesada)
    prob = best_classifier.predict_proba(query_vec)[0]
    pred_class = best_classifier.predict(query_vec)[0]
    class_label = "Construcción Pesada" if pred_class == 1 else "Construcción Ligera"
    confianza = prob[pred_class]
    print(f"  Categoría predicha: {class_label} (Confianza: {confianza*100:.1f}%)")
    
    # 2. Calcular similitud coseno con todo el catálogo
    similaridades = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Agregar score de similitud a nuestro catálogo
    df_temp = df_catalog.copy()
    df_temp['score_similitud'] = similaridades
    
    # Ordenar por similitud de mayor a menor y obtener top 3
    resultados = df_temp.sort_values(by='score_similitud', ascending=False).head(3)
    
    print("  Herramientas recomendadas (Top 3 por Similitud Coseno):")
    for idx, row in resultados.iterrows():
        print(f"    - [{row['categoria']}] {row['nombre']} (Score: {row['score_similitud']:.4f})")
        print(f"      Descripción: {row['descripcion']}")

if __name__ == "__main__":
    # Pruebas manuales
    realizar_busqueda("necesito hacer un agujero en la pared para colgar una repisa")
    realizar_busqueda("necesito romper una banqueta de concreto gruesa")
