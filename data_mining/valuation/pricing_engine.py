import os
import pickle
import psycopg2
import logging
from typing import Optional
from data_mining.nlp.preprocessor import inferir_tipo_herramienta, TIPO_TO_SECTOR

logger = logging.getLogger("toolshare-ml")

POSTGRES_DSN = os.getenv(
    "DATABASE_URL",
    "postgres://postgres:lagartija@localhost:5432/tool_inventory?sslmode=disable"
)
MIN_TRANSACTIONS_THRESHOLD = 30
FALLBACK_PRECIO_BASE = 1000.0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def _obtener_precio_base(cursor, tipo: str, sector_real: str, marca: str = "") -> dict:
    """Cascada de fuentes de precio ordenada por confianza.

    No hay ningún modelo entrenado prediciendo el precio: el catálogo semilla
    se actualiza en la base de datos sin reentrenar nada, así que agregar una
    herramienta nueva al catálogo escala sin tocar código ni modelos.

    marca=NULL en catalogo_semilla es el precio promedio de la categoría (la
    fila que ya existía antes de agregar variación por marca); las filas con
    marca son niveles específicos (ej. económica vs profesional) que evitan
    que un Truper y un DeWalt del mismo tipo de herramienta salgan al mismo
    precio cuando el usuario no sube ticket de compra.
    """
    marca_norm = (marca or "").strip()
    if marca_norm:
        cursor.execute(
            "SELECT valor_nuevo FROM catalogo_semilla WHERE category = %s AND marca ILIKE %s",
            (tipo, marca_norm)
        )
        row = cursor.fetchone()
        if row is not None:
            return {"precio_base": float(row[0]), "fuente_precio": "catalogo_semilla_marca"}

    cursor.execute(
        "SELECT valor_nuevo FROM catalogo_semilla WHERE category = %s AND marca IS NULL",
        (tipo,)
    )
    row = cursor.fetchone()
    if row is not None:
        return {"precio_base": float(row[0]), "fuente_precio": "catalogo_semilla_categoria"}

    categorias_del_sector = [cat for cat, sec in TIPO_TO_SECTOR.items() if sec == sector_real]
    if categorias_del_sector:
        cursor.execute(
            "SELECT AVG(valor_nuevo) FROM catalogo_semilla WHERE category = ANY(%s) AND marca IS NULL",
            (categorias_del_sector,)
        )
        row = cursor.fetchone()
        if row is not None and row[0] is not None:
            return {"precio_base": float(row[0]), "fuente_precio": "catalogo_semilla_sector"}

    return {"precio_base": None, "fuente_precio": None}


def _rango_plausible_categoria(cursor, tipo: str) -> Optional[tuple]:
    """Rango de precio nuevo plausible para la categoría, a partir del spread
    completo del catálogo semilla (económica..profesional). Márgenes 0.5x/1.5x
    para no rechazar tickets legítimos de ofertas o tiendas caras. Devuelve
    None si la categoría no tiene ninguna fila en el catálogo — en ese caso no
    se puede acotar, así que no se rechaza nada (fail open).
    """
    cursor.execute(
        "SELECT MIN(valor_nuevo), MAX(valor_nuevo) FROM catalogo_semilla WHERE category = %s",
        (tipo,)
    )
    row = cursor.fetchone()
    if row is None or row[0] is None:
        return None
    return (float(row[0]) * 0.5, float(row[1]) * 1.5)


def calcular_pricing_motor(
    nombre_herramienta: str,
    score_condicion: float,
    sector: str,
    marca: str,
    age_months: int = 12,
    precio_base_manual: Optional[float] = None,
    ticket_validado: bool = False,
) -> dict:
    """Motor central de cálculo de devaluación y tarifas de renta sugerida.

    Orden de confianza del precio base (valor si fuera nuevo):
      1. Ticket de compra validado por OCR (ticket_validado=True)
      2. Catálogo semilla: categoría exacta
      3. Catálogo semilla: promedio del sector de uso
      4. Declaración libre del propietario sin comprobante (baja confianza,
         se marca para revisión manual del administrador)
      5. Valor mínimo de emergencia (sin ningún dato disponible)
    """
    tipo = inferir_tipo_herramienta(nombre_herramienta)
    # "Otro" es un sector real y valido (Pala, Pico, Carretilla, Escalera, etc.
    # en TIPO_TO_SECTOR) — solo se cae a "Manual" cuando no hay sector en
    # absoluto o llega el placeholder de UI "Categoría" sin resolver.
    sector_real = TIPO_TO_SECTOR.get(tipo, sector)
    if not sector_real or sector_real.strip() in ["", "Categoría"]:
        sector_real = "Manual"

    n_transacciones = 0
    precio_base = None
    fuente_precio = None
    requiere_revision_manual = False

    try:
        conn = psycopg2.connect(POSTGRES_DSN)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT n_transacciones FROM v_conteo_transacciones_categoria WHERE category = %s",
            (tipo,)
        )
        row = cursor.fetchone()
        if row:
            n_transacciones = row[0]

        if ticket_validado and precio_base_manual is not None:
            precio_base = precio_base_manual
            rango = _rango_plausible_categoria(cursor, tipo)
            if rango is not None and not (rango[0] <= precio_base_manual <= rango[1]):
                # El ticket no corresponde a un precio creible para este tipo
                # de herramienta (ej. ticket de otra cosa subido para
                # "validar" el precio). No lo descartamos ni lo reemplazamos
                # — se respeta lo que subió el usuario — pero se le quita la
                # etiqueta de "confiable al 100%".
                fuente_precio = "ticket_fuera_de_rango"
                requiere_revision_manual = True
                logger.warning(
                    f"Ticket fuera de rango plausible para '{tipo}': "
                    f"${precio_base_manual:.2f} (rango esperado "
                    f"${rango[0]:.2f}-${rango[1]:.2f})"
                )
            else:
                fuente_precio = "ticket_validado"
        else:
            resultado_catalogo = _obtener_precio_base(cursor, tipo, sector_real, marca)
            precio_base = resultado_catalogo["precio_base"]
            fuente_precio = resultado_catalogo["fuente_precio"]

        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error al consultar base de datos Postgres: {e}")

    if precio_base is None:
        if precio_base_manual is not None:
            precio_base = precio_base_manual
            fuente_precio = "declaracion_usuario_sin_verificar"
            requiere_revision_manual = True
        else:
            precio_base = FALLBACK_PRECIO_BASE
            fuente_precio = "fallback_sin_datos"
            requiere_revision_manual = True

    model_filename = f"random_forest_{tipo.lower().replace(' ', '_')}.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)

    deprec_factor = max(0.15, 1.0 - (age_months * 0.015))
    valor_depreciado = precio_base * deprec_factor * score_condicion
    precio_renta_sugerido = 0.0
    modelo_utilizado = ""

    if n_transacciones >= MIN_TRANSACTIONS_THRESHOLD and os.path.exists(model_path):
        try:
            import pandas as pd
            with open(model_path, 'rb') as f:
                rf_model = pickle.load(f)

            input_df = pd.DataFrame([{
                'age_months': age_months,
                'condition_score': score_condicion,
                'estimated_value': precio_base
            }])
            precio_renta_sugerido = float(rf_model.predict(input_df)[0])
            modelo_utilizado = f"Random Forest ({tipo}, entrenado con {n_transacciones} rentas reales)"
        except Exception as e:
            logger.error(f"Error usando Random Forest específico: {e}")

    if precio_renta_sugerido <= 0.0:
        # Tasa progresiva en base al valor estimado (precio_base)
        if precio_base <= 250:
            tasa = 0.12 # 12% para herramientas baratas
        elif precio_base <= 1500:
            tasa = 0.06 # 6% para herramientas de costo medio
        else:
            tasa = 0.045 # 4.5% para herramientas profesionales/caras
            
        if sector_real in ["Eléctrico", "Neumático", "Energía"]:
            tasa += 0.015 # Sumar 1.5% extra para equipos motorizados
            
        precio_renta_sugerido = valor_depreciado * tasa
        modelo_utilizado = "Fórmula heurística de devaluación (categoría con menos de 30 rentas reales)"

    precio_renta_sugerido = max(10.0, min(2500.0, precio_renta_sugerido))
    
    # El precio mínimo permitido de renta debe ser el 50% de la sugerencia del día
    precio_renta_minimo = round(max(10.0, precio_renta_sugerido * 0.50), 2)

    tope_garantia = round(precio_base, 2)
    deducible_sugerido = round(precio_base * 0.10, 2)

    return {
        "tipo_herramienta": tipo,
        "sector_uso": sector_real,
        "marca": marca,
        "fuente_precio_catalogo": fuente_precio,
        "requiere_revision_manual": requiere_revision_manual,
        "valor_nuevo_estimado": round(precio_base, 2),
        "valor_depreciado_estimado": round(valor_depreciado, 2),
        "precio_renta_sugerido": round(precio_renta_sugerido, 2),
        "precio_renta_minimo": precio_renta_minimo,
        "deducible_garantia_sugerido": deducible_sugerido,
        "tope_cooperativo_garantia": tope_garantia,
        "modelo_pricing_utilizado": modelo_utilizado,
    }
