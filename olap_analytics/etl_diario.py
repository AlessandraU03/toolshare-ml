# =============================================================================
# Proceso ETL (Extract-Transform-Load) Diario
# PostgreSQL (OLTP) -> DuckDB (OLAP)
# ToolShare · Python & SQL
# =============================================================================

import os
import psycopg2
import duckdb
from datetime import datetime

# Configuración de conexiones
POSTGRES_DSN = "postgres://postgres:lagartija@localhost:5432/tool_inventory?sslmode=disable"
DUCKDB_PATH = os.path.join(os.path.dirname(__file__), "toolshare_olap.duckdb")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

def init_duckdb_schema(duck_conn):
    """Carga e inicializa el esquema estrella en DuckDB si no existe."""
    print("Inicializando esquema estrella en DuckDB...")
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    duck_conn.execute(schema_sql)
    print("Esquema inicializado correctamente.")

def run_etl():
    print(f"Iniciando proceso ETL ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    
    # 1. Conectar a PostgreSQL (OLTP)
    try:
        pg_conn = psycopg2.connect(POSTGRES_DSN)
        pg_cursor = pg_conn.cursor()
    except Exception as e:
        print(f"Error al conectar a PostgreSQL: {e}")
        return

    # 2. Conectar a DuckDB (OLAP)
    duck_conn = duckdb.connect(DUCKDB_PATH)
    init_duckdb_schema(duck_conn)

    try:
        # ==========================================
        # E - EXTRACT & T - TRANSFORM & L - LOAD
        # ==========================================
        
        # --- DIMENSIÓN: USUARIOS ---
        print("Cargando dimensión: dim_usuario...")
        pg_cursor.execute("SELECT id, name, role FROM users")
        usuarios = pg_cursor.fetchall()
        
        # Mapeo a DuckDB (limpieza y hash de IDs UUID a claves enteras)
        duck_conn.execute("DELETE FROM dim_usuario")
        for u in usuarios:
            u_id, u_name, u_role = u
            u_key = hash(u_id) & 0x7FFFFFFF # Convertir UUID a clave entera de 32 bits
            duck_conn.execute(
                "INSERT INTO dim_usuario (usuario_key, nombre, tipo_usuario) VALUES (?, ?, ?)",
                (u_key, u_name, u_role)
            )

        # --- DIMENSIÓN: CATEGORÍA ---
        print("Cargando dimensión: dim_categoria...")
        pg_cursor.execute("SELECT DISTINCT category FROM tools WHERE category IS NOT NULL AND category != ''")
        categorias = pg_cursor.fetchall()
        
        duck_conn.execute("DELETE FROM dim_categoria")
        for idx, cat in enumerate(categorias):
            cat_name = cat[0]
            # Determinar sector macro
            sector = "Manual"
            if cat_name in ["Taladro", "Rotomartillo", "Esmeriladora", "Cortadora", "Mezcladora", "Vibrador"]:
                sector = "Eléctrico"
            elif cat_name in ["Segueta", "Serrucho", "Sierra"]:
                sector = "Corte"
            elif cat_name in ["Lijadora", "Cepillo", "Llana"]:
                sector = "Acabado"
            elif cat_name in ["Generador", "Soldadora"]:
                sector = "Energía"
            elif cat_name in ["Compresor"]:
                sector = "Neumático"
            elif cat_name in ["Nivel", "Plomada", "Flexometro", "Escuadra"]:
                sector = "Medición"
                
            duck_conn.execute(
                "INSERT INTO dim_categoria (categoria_key, nombre, tipo_general) VALUES (?, ?, ?)",
                (idx + 1, cat_name, sector)
            )

        # --- DIMENSIÓN: ZONA (Jerarquía Geográfica) ---
        print("Cargando dimensión: dim_zona...")
        pg_cursor.execute("SELECT DISTINCT COALESCE(city, 'Desconocida'), COALESCE(state, 'Desconocido') FROM tools")
        zonas = pg_cursor.fetchall()
        
        duck_conn.execute("DELETE FROM dim_zona")
        for idx, z in enumerate(zonas):
            ciudad, estado = z
            duck_conn.execute(
                "INSERT INTO dim_zona (zona_key, ciudad, estado) VALUES (?, ?, ?)",
                (idx + 1, ciudad, estado)
            )

        # --- DIMENSIÓN: TIEMPO ---
        print("Cargando dimensión: dim_tiempo...")
        pg_cursor.execute("SELECT DISTINCT start_date::date FROM rentals WHERE start_date IS NOT NULL")
        fechas = pg_cursor.fetchall()
        
        duck_conn.execute("DELETE FROM dim_tiempo")
        for f in fechas:
            fecha = f[0]
            t_key = int(fecha.strftime("%Y%m%d"))
            dia = fecha.day
            mes = fecha.month
            anio = fecha.year
            nombre_mes = fecha.strftime("%B").capitalize()
            trimestre = f"Q{(mes-1)//3 + 1}"
            
            duck_conn.execute(
                "INSERT INTO dim_tiempo (tiempo_key, fecha, dia, mes, anio, nombre_mes, trimestre) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (t_key, fecha, dia, mes, anio, nombre_mes, trimestre)
            )

        # --- TABLA DE HECHOS: fact_rentas ---
        print("Cargando tabla de hechos: fact_rentas (Filtrando outliers)...")
        # Extraemos de la vista limpia de Postgres (Transformación por Isolation Forest incorporada)
        pg_cursor.execute("""
            SELECT 
                start_date::date,
                category,
                COALESCE(city, 'Desconocida'),
                COALESCE(state, 'Desconocido'),
                requester_id,
                daily_rate,
                total_amount,
                condition_score,
                (end_date::date - start_date::date) as total_dias
            FROM v_transacciones_limpias
        """)
        transacciones = pg_cursor.fetchall()

        duck_conn.execute("DELETE FROM fact_rentas")
        fact_count = 0
        for tx in transacciones:
            f_date, cat, city, state, req_id, rate, total, cnn_score, dias = tx
            
            # Buscar llaves subrogadas correspondientes en DuckDB
            t_key = int(f_date.strftime("%Y%m%d"))
            
            c_res = duck_conn.execute("SELECT categoria_key FROM dim_categoria WHERE nombre = ?", (cat,)).fetchone()
            c_key = c_res[0] if c_res else 1
            
            z_res = duck_conn.execute("SELECT zona_key FROM dim_zona WHERE ciudad = ? AND estado = ?", (city, state)).fetchone()
            z_key = z_res[0] if z_res else 1
            
            u_key = hash(req_id) & 0x7FFFFFFF
            
            # Insertar en hechos
            duck_conn.execute(
                "INSERT INTO fact_rentas (tiempo_key, categoria_key, zona_key, usuario_key, precio_final, condicion_cnn, total_dias) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (t_key, c_key, z_key, u_key, float(total), float(cnn_score), int(dias) if dias > 0 else 1)
            )
            fact_count += 1
            
        print(f"Proceso ETL completado con éxito. {fact_count} hechos cargados en DuckDB.")

    except Exception as e:
        print(f"Error durante la ejecución del ETL: {e}")
        
    finally:
        pg_cursor.close()
        pg_conn.close()
        duck_conn.close()

if __name__ == "__main__":
    run_etl()
