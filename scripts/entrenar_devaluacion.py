# scripts/entrenar_devaluacion.py
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Cargar datos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "dataset", "devaluacion_herramientas.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"No se encontró el dataset en {csv_path}. Ejecuta primero generar_datos_devaluacion.py")

df = pd.read_csv(csv_path)

# Variables de entrada y de salida
# Predeciremos el precio base de catálogo y el valor depreciado simultáneamente (Multi-output Regression)
X = df[['sector_uso', 'marca', 'tipo_herramienta', 'score_condicion']]
y = df[['precio_base_catalogo', 'valor_real_depreciado']]

# 2. Configurar preprocesamiento (ColumnTransformer)
categorical_features = ['sector_uso', 'marca', 'tipo_herramienta']
numeric_features = ['score_condicion']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 3. Definir los modelos a evaluar (Alineado con los temas de Clase 5)
models = {
    'Regresión Lineal': LinearRegression(),
    'Ridge (L2 Regularized)': Ridge(alpha=1.0),
    'Lasso (L1 Regularized)': Lasso(alpha=1.0),
    'Árbol de Decisión (Poda/max_depth=5)': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Árbol de Decisión (Poda/max_depth=8)': DecisionTreeRegressor(max_depth=8, random_state=42),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
}

# 4. Evaluación usando Validación Cruzada (5-Fold CV)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

print("=== EVALUACIÓN DE MODELOS CON VALIDACIÓN CRUZADA (K-FOLD, K=5) ===")

for name, model in models.items():
    # Crear un pipeline que encadene preprocesamiento y modelo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Evaluar con validación cruzada obteniendo MSE, MAE y R2
    scoring = {
        'neg_mse': 'neg_mean_squared_error',
        'neg_mae': 'neg_mean_absolute_error',
        'r2': 'r2'
    }
    
    cv_results = cross_validate(pipeline, X, y, cv=kf, scoring=scoring, return_train_score=True)
    
    # Calcular promedios de métricas
    train_r2 = np.mean(cv_results['train_r2'])
    val_r2 = np.mean(cv_results['test_r2'])
    
    train_mse = -np.mean(cv_results['train_neg_mse'])
    val_mse = -np.mean(cv_results['test_neg_mse'])
    
    train_rmse = np.sqrt(train_mse)
    val_rmse = np.sqrt(val_mse)
    
    train_mae = -np.mean(cv_results['train_neg_mae'])
    val_mae = -np.mean(cv_results['test_neg_mae'])
    
    results.append({
        'Modelo': name,
        'Train RMSE': train_rmse,
        'Val RMSE': val_rmse,
        'Train MAE': train_mae,
        'Val MAE': val_mae,
        'Train R2': train_r2,
        'Val R2': val_r2
    })
    
    print(f"\nModelo: {name}")
    print(f"  Train: RMSE={train_rmse:.2f}, MAE={train_mae:.2f}, R2={train_r2:.4f}")
    print(f"  Val:   RMSE={val_rmse:.2f}, MAE={val_mae:.2f}, R2={val_r2:.4f}")

# Convertir resultados a DataFrame para visualización
df_results = pd.DataFrame(results)

# 5. Seleccionar y entrenar el mejor modelo final en todo el dataset
best_model_name = df_results.sort_values(by='Val RMSE').iloc[0]['Modelo']
print(f"\nEl mejor modelo seleccionado es: {best_model_name}")

best_model = models[best_model_name]
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', best_model)
])

# Entrenar en la totalidad de los datos
final_pipeline.fit(X, y)

# Crear directorio de modelos si no existe
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)
model_save_path = os.path.join(models_dir, "modelo_devaluacion.pkl")

with open(model_save_path, 'wb') as f:
    pickle.dump(final_pipeline, f)

print(f"\nModelo final guardado exitosamente en: {model_save_path}")

# 6. Graficar comparación de R2 y RMSE (Usando matplotlib estándar para evitar dependencias extras)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.barh(df_results['Modelo'], df_results['Val R2'], color='skyblue')
plt.title("Comparación de R² de Validación (Mayor es mejor)")
plt.xlim(0.7, 1.0)
plt.xlabel("R²")

plt.subplot(1, 2, 2)
plt.barh(df_results['Modelo'], df_results['Val RMSE'], color='salmon')
plt.title("Comparación de RMSE de Validación (Menor es mejor)")
plt.xlabel("RMSE (MXN)")

plt.tight_layout()
plt.savefig(os.path.join(models_dir, "comparativa_regresores.png"))
print("Gráfica comparativa guardada en models/comparativa_regresores.png")
