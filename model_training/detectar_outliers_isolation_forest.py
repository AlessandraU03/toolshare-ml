# =============================================================================
# Detección de Anomalías en Transacciones (Isolation Forest)
# ToolShare · Python ML
# =============================================================================

import pickle
import os
import psycopg2
import numpy as np
from sklearn.ensemble import IsolationForest

# Configuración de conexiones
POSTGRES_DSN = os.getenv(
    "DATABASE_URL",
    "postgres://postgres:lagartija@localhost:5432/tool_inventory?sslmode=disable"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "modelo_isolation_forest.pkl")

def run_outlier_detection():
    print("Iniciando análisis de detección de anomalías (Isolation Forest)...")
    
    # 1. Conectar a PostgreSQL
    try:
        conn = psycopg2.connect(POSTGRES_DSN)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Error al conectar a PostgreSQL: {e}")
        return

    try:
        # 2. Extraer transacciones completadas
        cursor.execute("""
            SELECT id, daily_rate, total_amount, 
                   EXTRACT(EPOCH FROM (end_date - start_date))/86400 as total_dias 
            FROM rentals 
            WHERE status = 'completed'
        """)
        records = cursor.fetchall()
        
        # Si hay muy pocos registros, no tiene sentido estadístico entrenar el Isolation Forest
        if len(records) < 10:
            print(f"Información insuficiente ({len(records)} transacciones). Se requieren al menos 10 para detectar anomalías.")
            # Por seguridad, resetear todas a false
            cursor.execute("UPDATE rentals SET is_outlier = FALSE WHERE status = 'completed'")
            conn.commit()
            return

        # 3. Preparar matriz X
        # Features: daily_rate, total_amount, total_dias
        X = np.array([[float(r[1]), float(r[2]), float(r[3]) if r[3] > 0 else 1.0] for r in records])

        # 4. Entrenar Isolation Forest (con un 5% estimado de contaminación/anomalías)
        contamination_rate = 0.05
        print(f"Entrenando Isolation Forest con {len(records)} registros...")
        iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
        
        # fit_predict devuelve 1 para normales y -1 para outliers/anomalías
        predictions = iso_forest.fit_predict(X)
        
        # 5. Actualizar la base de datos de PostgreSQL
        outliers_count = 0
        for i, r in enumerate(records):
            rental_id = r[0]
            is_outlier = bool(predictions[i] == -1)
            
            if is_outlier:
                outliers_count += 1
                
            cursor.execute(
                "UPDATE rentals SET is_outlier = %s WHERE id = %s",
                (is_outlier, rental_id)
            )
        conn.commit()
        print(f"Análisis completado. Se detectaron {outliers_count} transacciones anómalas (outliers) de {len(records)} totales.")

        # 6. Registrar en la bitácora pricing_model_runs
        cursor.execute(
            """INSERT INTO pricing_model_runs (model_type, n_registros, metric_name, metric_value, notes)
               VALUES (%s, %s, %s, %s, %s)""",
            ("isolation_forest", len(records), "contamination", contamination_rate, 
             f"Detección de anomalías finalizada. {outliers_count} outliers marcados en la base de datos.")
        )
        conn.commit()

        # 7. Guardar modelo serializado (.pkl)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(iso_forest, f)
            
        print(f"Modelo Isolation Forest guardado exitosamente en: {MODEL_PATH}")

    except Exception as e:
        print(f"Error durante la detección de anomalías: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    run_outlier_detection()
