# =============================================================================
# Reentrenamiento de Valuación de Precios (Random Forest Regressor)
# ToolShare · Python ML
# =============================================================================

import pickle
import os
import psycopg2
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Configuración de conexiones
POSTGRES_DSN = "postgres://postgres:lagartija@localhost:5432/tool_inventory?sslmode=disable"
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MIN_TRANSACTIONS_THRESHOLD = 30

def run_random_forest_training():
    print("Iniciando entrenamiento dinámico de Random Forest por categoría...")
    
    # 1. Conectar a PostgreSQL
    try:
        conn = psycopg2.connect(POSTGRES_DSN)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Error al conectar a PostgreSQL: {e}")
        return

    try:
        # 2. Leer las transacciones limpias de la vista
        query = """
            SELECT category, brand, age_months, condition_score, estimated_value, daily_rate 
            FROM v_transacciones_limpias
        """
        df = pd.read_sql(query, conn)
        
        if df.empty:
            print("No hay transacciones completadas y limpias en el sistema aún. Todas las categorías usarán Catálogo Semilla.")
            return

        # 3. Agrupar por categoría para ver cuáles califican
        categories = df['category'].unique()
        
        for category in categories:
            df_cat = df[df['category'] == category]
            n_transacciones = len(df_cat)
            
            print(f"Categoría '{category}': {n_transacciones} transacciones registradas.")
            
            if n_transacciones < MIN_TRANSACTIONS_THRESHOLD:
                print(f"  -> Insuficientes transacciones (mínimo {MIN_TRANSACTIONS_THRESHOLD}). Se continuará usando K-Means Semilla.")
                continue
                
            print(f"  -> ¡Umbral superado! Iniciando entrenamiento de Random Forest para '{category}'...")
            
            # Features (X) e Target (y)
            # Características: age_months (antigüedad), condition_score (CNN), estimated_value (valor original)
            X = df_cat[['age_months', 'condition_score', 'estimated_value']]
            y = df_cat['daily_rate']
            
            # Dividir en entrenamiento y prueba si hay suficientes datos, de lo contrario usar todo
            if n_transacciones >= 40:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            else:
                X_train, X_test, y_train, y_test = X, X, y, y
                
            # 4. Entrenar Regresor Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
            rf_model.fit(X_train, y_train)
            
            # 5. Calcular métricas
            y_pred = rf_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            # En caso de entrenar con pocos datos y tener R2 negativos o nulos, establecer un mínimo
            if pd.isna(r2):
                r2 = 0.0
            print(f"  -> Modelo entrenado con éxito. Métrica R2 Score: {r2:.4f}")
            
            # 6. Guardar modelo serializado (.pkl)
            model_filename = f"random_forest_{category.lower().replace(' ', '_')}.pkl"
            model_path = os.path.join(MODELS_DIR, model_filename)
            
            os.makedirs(MODELS_DIR, exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(rf_model, f)
            print(f"  -> Guardado modelo en: {model_path}")
            
            # 7. Registrar en pricing_model_runs en Postgres
            cursor.execute(
                """INSERT INTO pricing_model_runs (model_type, n_registros, metric_name, metric_value, notes)
                   VALUES (%s, %s, %s, %s, %s)""",
                ("random_forest", n_transacciones, "r2_score", float(r2), 
                 f"Reentrenamiento completado para la categoría '{category}' con R2={r2:.4f}")
            )
            conn.commit()

    except Exception as e:
        print(f"Error durante el reentrenamiento: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    run_random_forest_training();
