# =============================================================================
# Carga de Catálogo Semilla y Entrenamiento K-Means
# ToolShare · Python ML
# =============================================================================

import pickle
import os
import psycopg2
import numpy as np
from sklearn.cluster import KMeans

# Configuración de conexiones
POSTGRES_DSN = os.getenv(
    "DATABASE_URL",
    "postgres://postgres:lagartija@localhost:5432/tool_inventory?sslmode=disable"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "modelo_kmeans_semilla.pkl")

# Catálogo semilla inicial de 15 herramientas de referencia en México
SEED_TOOLS = [
    # (category, precio_base_renta, valor_nuevo)
    ("Taladro", 52.50, 1500.00),
    ("Rotomartillo", 122.50, 3500.00),
    ("Esmeriladora", 63.00, 1800.00),
    ("Sierra", 100.00, 2500.00),
    ("Segueta", 10.00, 200.00),
    ("Lijadora", 36.00, 1200.00),
    ("Llana", 10.00, 150.00),
    ("Pinza", 10.00, 400.00),
    ("Martillo", 10.00, 180.00),
    ("Generador", 360.00, 8000.00),
    ("Soldadora", 202.50, 4500.00),
    ("Compresor", 120.00, 3000.00),
    ("Nivel", 10.00, 300.00),
    ("Andamio", 60.00, 2400.00),
    ("Escalera", 87.50, 3500.00)
]

def run_kmeans_seeding():
    print("Iniciando sembrado de catálogo y entrenamiento K-Means...")
    
    # 1. Conectar a PostgreSQL
    try:
        conn = psycopg2.connect(POSTGRES_DSN)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Error al conectar a PostgreSQL: {e}")
        return

    try:
        # 2. Limpiar y sembrar la tabla catalogo_semilla
        print("Limpiando e insertando herramientas de referencia en catalogo_semilla...")
        cursor.execute("TRUNCATE TABLE catalogo_semilla RESTART IDENTITY")
        
        for category, precio_base, valor_nuevo in SEED_TOOLS:
            cursor.execute(
                "INSERT INTO catalogo_semilla (category, precio_base, valor_nuevo) VALUES (%s, %s, %s)",
                (category, precio_base, valor_nuevo)
            )
        conn.commit()

        # 3. Leer los datos para el entrenamiento de K-Means
        cursor.execute("SELECT id, precio_base, valor_nuevo FROM catalogo_semilla")
        records = cursor.fetchall()
        
        # Preparar matriz de entrenamiento X (monto de renta y valor nuevo)
        X = np.array([[float(r[1]), float(r[2])] for r in records])
        
        # 4. Entrenar K-Means con k=3 (Gama Baja, Media y Alta)
        print("Entrenando modelo K-Means (k=3)...")
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # 5. Guardar los Cluster IDs asignados en PostgreSQL
        print("Guardando cluster_ids en la base de datos...")
        for i, r in enumerate(records):
            record_id = r[0]
            cluster_id = int(labels[i])
            cursor.execute(
                "UPDATE catalogo_semilla SET cluster_id = %s WHERE id = %s",
                (cluster_id, record_id)
            )
        conn.commit()

        # 6. Registrar métricas del modelo (Inercia del K-Means)
        print("Registrando ejecución en pricing_model_runs...")
        cursor.execute(
            """INSERT INTO pricing_model_runs (model_type, n_registros, metric_name, metric_value, notes)
               VALUES (%s, %s, %s, %s, %s)""",
            ("kmeans_semilla", len(records), "inertia", float(kmeans.inertia_), 
             "Modelo K-Means entrenado para segmentación de precios de catálogo semilla")
        )
        conn.commit()

        # 7. Guardar modelo serializado (.pkl)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(kmeans, f)
            
        print(f"Modelo K-Means guardado exitosamente en: {MODEL_PATH}")
        print("Catálogo semilla sembrado y clasificado correctamente en la base de datos.")

    except Exception as e:
        print(f"Error durante el proceso: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    run_kmeans_seeding()
