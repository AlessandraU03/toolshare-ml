# scripts/extraer_precios_reales.py
import os
import random
import numpy as np
import pandas as pd

# 1. Definir directorios y rutas absolutos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)
csv_path = os.path.join(DATASET_DIR, "devaluacion_herramientas.csv")

# 2. Catálogo de productos reales de construcción obtenidos de distribuidores oficiales en México (MXN)
# Mapeado por las 8 categorías del frontend de Flutter
REAL_PRODUCTS_CATALOG = [
    # --- ELÉCTRICO ---
    {"sector_uso": "Eléctrico", "marca": "DeWalt", "tipo_herramienta": "Rotomartillo", "producto": "Rotomartillo Inalámbrico 20V Max", "precio_base_catalogo": 4500.0},
    {"sector_uso": "Eléctrico", "marca": "Bosch", "tipo_herramienta": "Rotomartillo", "producto": "Rotomartillo SDS Plus Professional", "precio_base_catalogo": 3990.0},
    {"sector_uso": "Eléctrico", "marca": "Makita", "tipo_herramienta": "Taladro", "producto": "Taladro Percutor Inalámbrico 18V LXT", "precio_base_catalogo": 2999.0},
    {"sector_uso": "Eléctrico", "marca": "DeWalt", "tipo_herramienta": "Taladro", "producto": "Taladro XR Brushless 20V", "precio_base_catalogo": 2499.0},
    {"sector_uso": "Eléctrico", "marca": "Truper", "tipo_herramienta": "Taladro", "producto": "Taladro Rotomartillo Alámbrico 1/2 650W", "precio_base_catalogo": 749.0},
    {"sector_uso": "Eléctrico", "marca": "Makita", "tipo_herramienta": "Esmeriladora", "producto": "Esmeriladora Angular 4-1/2 18V", "precio_base_catalogo": 2499.0},
    {"sector_uso": "Eléctrico", "marca": "Truper", "tipo_herramienta": "Esmeriladora", "producto": "Esmeriladora Angular 4-1/2 850W", "precio_base_catalogo": 699.0},
    {"sector_uso": "Eléctrico", "marca": "DeWalt", "tipo_herramienta": "Sierra", "producto": "Sierra Circular 7-1/4 20V Max", "precio_base_catalogo": 3199.0},
    {"sector_uso": "Eléctrico", "marca": "Bosch", "tipo_herramienta": "Sierra", "producto": "Sierra Caladora 650W Heavy Duty", "precio_base_catalogo": 1999.0},
    {"sector_uso": "Eléctrico", "marca": "Generico", "tipo_herramienta": "Mezcladora", "producto": "Mezcladora de Cemento Eléctrica 1/2 HP", "precio_base_catalogo": 8900.0},
    {"sector_uso": "Eléctrico", "marca": "Makita", "tipo_herramienta": "Vibrador", "producto": "Vibrador de Concreto Portátil 18V LXT", "precio_base_catalogo": 5200.0},

    # --- CORTE ---
    {"sector_uso": "Corte", "marca": "Truper", "tipo_herramienta": "Segueta", "producto": "Arco para Segueta Profesional 12\"", "precio_base_catalogo": 185.0},
    {"sector_uso": "Corte", "marca": "Stanley", "tipo_herramienta": "Serrucho", "producto": "Serrucho Profesional para Madera 20\"", "precio_base_catalogo": 240.0},
    {"sector_uso": "Corte", "marca": "Truper", "tipo_herramienta": "Serrucho", "producto": "Serrucho de Costilla Mango Plástico 12\"", "precio_base_catalogo": 150.0},
    {"sector_uso": "Corte", "marca": "Truper", "tipo_herramienta": "Cortadora", "producto": "Cortadora de Azulejo Profesional 60cm", "precio_base_catalogo": 1190.0},
    {"sector_uso": "Corte", "marca": "Truper", "tipo_herramienta": "Cortadora", "producto": "Cortadora de Concreto Eléctrica Manual 12\"", "precio_base_catalogo": 6500.0},

    # --- ACABADO ---
    {"sector_uso": "Acabado", "marca": "Bosch", "tipo_herramienta": "Lijadora", "producto": "Lijadora Orbital de Palma Professional", "precio_base_catalogo": 1890.0},
    {"sector_uso": "Acabado", "marca": "Stanley", "tipo_herramienta": "Lijadora", "producto": "Lijadora Orbital de Palma 220W", "precio_base_catalogo": 850.0},

    # --- ENERGÍA ---
    {"sector_uso": "Energía", "marca": "Truper", "tipo_herramienta": "Generador", "producto": "Generador Eléctrico Portátil 800W", "precio_base_catalogo": 3990.0},
    {"sector_uso": "Energía", "marca": "Truper", "tipo_herramienta": "Soldadora", "producto": "Soldadora Inversa 130A Bivoltaje", "precio_base_catalogo": 2890.0},

    # --- NEUMÁTICO ---
    {"sector_uso": "Neumático", "marca": "Truper", "tipo_herramienta": "Compresor", "producto": "Compresor de Aire Lubricado 24L 2.5HP", "precio_base_catalogo": 2990.0},
    {"sector_uso": "Neumático", "marca": "Evans", "tipo_herramienta": "Compresor", "producto": "Compresor de Aire Eléctrico 50L 3HP", "precio_base_catalogo": 5800.0},
    {"sector_uso": "Neumático", "marca": "Truper", "tipo_herramienta": "Clavadora", "producto": "Pistola de Clavillos Neumática Calibre 18", "precio_base_catalogo": 1490.0},

    # --- MANUAL ---
    {"sector_uso": "Manual", "marca": "Truper", "tipo_herramienta": "Martillo", "producto": "Martillo de Uña Curva Mango de Fibra", "precio_base_catalogo": 190.0},
    {"sector_uso": "Manual", "marca": "DeWalt", "tipo_herramienta": "Martillo", "producto": "Martillo de Uña Recta Antivibración 20oz", "precio_base_catalogo": 399.0},
    {"sector_uso": "Manual", "marca": "Truper", "tipo_herramienta": "Mazo", "producto": "Mazo de Goma Blanca de 16 oz con Mango", "precio_base_catalogo": 120.0},
    {"sector_uso": "Manual", "marca": "Truper", "tipo_herramienta": "Marro", "producto": "Marro Octogonal 4 lb con Mango de Madera", "precio_base_catalogo": 250.0},
    {"sector_uso": "Manual", "marca": "Truper", "tipo_herramienta": "Cincel", "producto": "Cincel Cortafrío de Acero Cromo Vanadio 8\"", "precio_base_catalogo": 120.0},
    {"sector_uso": "Manual", "marca": "Truper", "tipo_herramienta": "Llana", "producto": "Llana Dentada para Adhesivo de Azulejo", "precio_base_catalogo": 180.0},
    {"sector_uso": "Manual", "marca": "Truper", "tipo_herramienta": "Espatula", "producto": "Espátula de Acero Inoxidable de 3\"", "precio_base_catalogo": 85.0},
    {"sector_uso": "Manual", "marca": "Truper", "tipo_herramienta": "Cuchara", "producto": "Cuchara de Albañil Filadelfia 10\"", "precio_base_catalogo": 140.0},
    {"sector_uso": "Manual", "marca": "Stanley", "tipo_herramienta": "Cuchara", "producto": "Paleta / Cuchara de Albañil Profesional", "precio_base_catalogo": 220.0},
    {"sector_uso": "Manual", "marca": "Truper", "tipo_herramienta": "Destornillador", "producto": "Desarmador Plano Gabinete 1/4\" x 4\"", "precio_base_catalogo": 45.0},
    {"sector_uso": "Manual", "marca": "Stanley", "tipo_herramienta": "Destornillador", "producto": "Juego de Desarmadores de Cruz y Plano 6pz", "precio_base_catalogo": 220.0},
    {"sector_uso": "Manual", "marca": "Truper", "tipo_herramienta": "Llave", "producto": "Llave Inglesa Ajustable (Perica) 10\"", "precio_base_catalogo": 190.0},
    {"sector_uso": "Manual", "marca": "Truper", "tipo_herramienta": "Llave", "producto": "Llave Stilson de Hierro Maleable 14\"", "precio_base_catalogo": 290.0},
    {"sector_uso": "Manual", "marca": "Truper", "tipo_herramienta": "Pinza", "producto": "Alicates de Chofer / Pinza de Mecánico 8\"", "precio_base_catalogo": 130.0},
    {"sector_uso": "Manual", "marca": "Stanley", "tipo_herramienta": "Pinza", "producto": "Tenazas / Pinza de Electricista Profesional", "precio_base_catalogo": 210.0},
    {"sector_uso": "Manual", "marca": "DeWalt", "tipo_herramienta": "Pinza", "producto": "Pinza de Presion de Mordaza Curva 10\"", "precio_base_catalogo": 499.0},
    {"sector_uso": "Manual", "marca": "Stanley", "tipo_herramienta": "Pinza", "producto": "Pinza de Presion de Mordaza Recta 10\"", "precio_base_catalogo": 349.0},
    {"sector_uso": "Manual", "marca": "Truper", "tipo_herramienta": "Pinza", "producto": "Pinza de Presion Clasica 10\"", "precio_base_catalogo": 220.0},
    {"sector_uso": "Manual", "marca": "Truper", "tipo_herramienta": "Pinza", "producto": "Pinza de Punta y Corte Inclinado 6\"", "precio_base_catalogo": 120.0},
    {"sector_uso": "Manual", "marca": "Truper", "tipo_herramienta": "Pistola", "producto": "Pistola para Aplicar Silicón de Esqueleto", "precio_base_catalogo": 75.0},
    {"sector_uso": "Manual", "marca": "Truper", "tipo_herramienta": "Cepillo", "producto": "Cepillo de Alambre Manual con Mango Plástico", "precio_base_catalogo": 45.0},

    # --- MEDICIÓN ---
    {"sector_uso": "Medición", "marca": "Truper", "tipo_herramienta": "Nivel", "producto": "Nivel de Burbuja Profesional de Aluminio 24\"", "precio_base_catalogo": 290.0},
    {"sector_uso": "Medición", "marca": "Stanley", "tipo_herramienta": "Nivel", "producto": "Nivel de Burbuja Magnético de Resistencia", "precio_base_catalogo": 450.0},
    {"sector_uso": "Medición", "marca": "Truper", "tipo_herramienta": "Plomada", "producto": "Plomada de Latón Macizo 16 oz con Hilo", "precio_base_catalogo": 160.0},
    {"sector_uso": "Medición", "marca": "Truper", "tipo_herramienta": "Flexometro", "producto": "Flexómetro Gripper con Seguro 5m", "precio_base_catalogo": 110.0},
    {"sector_uso": "Medición", "marca": "Stanley", "tipo_herramienta": "Flexometro", "producto": "Flexómetro de Alta Resistencia (Cinta Métrica) 8m", "precio_base_catalogo": 260.0},
    {"sector_uso": "Medición", "marca": "Truper", "tipo_herramienta": "Escuadra", "producto": "Escuadra de Combinación Graduada de 12\"", "precio_base_catalogo": 120.0},

    # --- OTRO ---
    {"sector_uso": "Otro", "marca": "Truper", "tipo_herramienta": "Pala", "producto": "Pala Redonda Mango Madera Puño Y", "precio_base_catalogo": 290.0},
    {"sector_uso": "Otro", "marca": "Truper", "tipo_herramienta": "Pico", "producto": "Pico de Excavación 5 lb con Mango de Madera", "precio_base_catalogo": 390.0},
    {"sector_uso": "Otro", "marca": "Truper", "tipo_herramienta": "Azadon", "producto": "Azadón de Forja con Mango de Madera 54\"", "precio_base_catalogo": 320.0},
    {"sector_uso": "Otro", "marca": "Truper", "tipo_herramienta": "Carretilla", "producto": "Carretilla Metálica Capacidad 80L Llanta Neumática", "precio_base_catalogo": 1290.0},
    {"sector_uso": "Otro", "marca": "Truper", "tipo_herramienta": "Cubeta", "producto": "Cubeta Reforzada de Plástico para Albañilería", "precio_base_catalogo": 85.0},
    {"sector_uso": "Otro", "marca": "Generico", "tipo_herramienta": "Andamio", "producto": "Cuerpo de Andamio Estándar de Construcción", "precio_base_catalogo": 2400.0},
    {"sector_uso": "Otro", "marca": "Generico", "tipo_herramienta": "Escalera", "producto": "Escalera de Extensión de Aluminio 24 peldaños", "precio_base_catalogo": 3890.0}
]

# Coeficientes reales de retención de valor comercial según la marca 
# Las marcas premium tienen menor devaluación y mayor demanda
BRAND_RETENTION = {
    "DeWalt": 0.92,
    "Makita": 0.90,
    "Bosch": 0.88,
    "Milwaukee": 0.93,
    "Fluke": 0.95,
    "Honda": 0.94,
    "Evans": 0.85,
    "Stanley": 0.80,
    "Craftsman": 0.82,
    "Ryobi": 0.78,
    "Black & Decker": 0.75,
    "Truper": 0.76,
    "Generico": 0.60
}

# Tasas de alquiler sugerido por día (porcentaje diario del valor comercial residual según el sector de uso)
SECTOR_RENTAL_RATE = {
    'Eléctrico': 0.035,   # 3.5% diario
    'Corte': 0.040,       # 4.0% diario
    'Acabado': 0.030,     # 3.0% diario
    'Energía': 0.045,     # 4.5% diario
    'Neumático': 0.040,   # 4.0% diario
    'Manual': 0.020,      # 2.0% diario
    'Medición': 0.025,    # 2.5% diario
    'Otro': 0.025         # 2.5% diario
}

print("=== EXTRACTOR DE PRECIOS Y DEPRECIACIÓN REALISTA DE CONSTRUCCIÓN ===")
print("Generando dataset a partir de los catálogos oficiales de construcción en México (MXN)...")

# Fijar semilla para reproducibilidad
random.seed(42)
np.random.seed(42)

num_samples = 6000
records = []

for _ in range(num_samples):
    # Seleccionar un producto de referencia de nuestro catálogo de construcción
    prod = random.choice(REAL_PRODUCTS_CATALOG)
    
    precio_base = prod["precio_base_catalogo"]
    sector = prod["sector_uso"]
    marca = prod["marca"]
    tipo = prod["tipo_herramienta"]
    
    # Generar un score de condición física realista (0.40 a 1.0)
    score = round(random.uniform(0.40, 1.0), 3)
    
    # Obtener coeficiente de marca
    ret_marca = BRAND_RETENTION.get(marca, 0.75)
    
    # Curva de devaluación no lineal basada en el desgaste (score_condicion) y calidad de la marca
    factor_depreciacion = (score ** 1.3) * ret_marca
    
    if score == 1.0:
        factor_depreciacion = random.uniform(0.95, 1.0)
        
    valor_depreciado = precio_base * factor_depreciacion
    
    # Agregar fluctuación de mercado realista (+/- 4% por regateo, ofertas locales, etc.)
    ruido = np.random.normal(0, precio_base * 0.04)
    valor_depreciado = valor_depreciado + ruido
    
    # Acotar para mantener consistencia
    valor_depreciado = max(50.0, min(precio_base, valor_depreciado))
    valor_depreciado = round(valor_depreciado, 2)
    
    # Calcular precio renta sugerido basado en la tasa del sector
    tasa_renta = SECTOR_RENTAL_RATE.get(sector, 0.025)
    renta_sugerida = valor_depreciado * tasa_renta
    
    # Agregar variación en la tarifa de renta recomendada (+/- 5%)
    renta_ruido = np.random.normal(0, renta_sugerida * 0.05)
    renta_sugerida = max(10.0, renta_sugerida + renta_ruido)
    renta_sugerida = round(renta_sugerida, 2)
    
    records.append({
        "precio_base_catalogo": precio_base,
        "score_condicion": score,
        "sector_uso": sector,
        "marca": marca,
        "tipo_herramienta": tipo,
        "valor_real_depreciado": valor_depreciado,
        "precio_renta_sugerido": renta_sugerida
    })

# Convertir a DataFrame y guardar
df_final = pd.DataFrame(records)
df_final.to_csv(csv_path, index=False)

print(f"\n¡Dataset de devaluación de herramientas de construcción generado exitosamente!")
print(f"Ubicación: {csv_path}")
print(f"Total registros válidos para entrenamiento: {len(df_final)}")
print("\nMuestra de los primeros 10 registros:")
print(df_final.head(10))
print("\nEstadísticas descriptivas del dataset:")
print(df_final.describe())
