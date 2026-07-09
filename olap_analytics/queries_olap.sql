-- =============================================================================
-- Consultas Analíticas OLAP (DuckDB)
-- ToolShare · DuckDB
-- =============================================================================
-- Estas consultas ejecutan operaciones multidimensionales sobre el esquema
-- estrella de DuckDB. Se pueden integrar en el backend para reportes.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1) OPERACIÓN CUBE: Ingresos totales y días rentados por categoría y ciudad
--    (Agrupa por todas las combinaciones posibles de jerarquías y totales generales)
-- -----------------------------------------------------------------------------
SELECT 
    categoria, 
    ciudad, 
    SUM(precio_final) AS ingresos_totales,
    AVG(total_dias) AS promedio_dias_renta,
    COUNT(*) AS cantidad_rentas
FROM v_rentas
GROUP BY CUBE (categoria, ciudad)
ORDER BY ingresos_totales DESC;

-- -----------------------------------------------------------------------------
-- 2) OPERACIÓN ROLL-UP: Jerarquía temporal y geográfica
--    (Útil para ver totales acumulados de Estado > Ciudad, y de Año > Mes)
-- -----------------------------------------------------------------------------
SELECT 
    estado,
    ciudad,
    anio,
    mes,
    SUM(precio_final) AS ingresos_acumulados
FROM v_rentas
GROUP BY ROLLUP (estado, ciudad, anio, mes)
ORDER BY estado, ciudad, anio, mes;

-- -----------------------------------------------------------------------------
-- 3) DRILL-DOWN: Desglose de precios promedio por categoría a nivel de condición cnn
--    (Para ver cómo influye la condición física en la rentabilidad de renta)
-- -----------------------------------------------------------------------------
SELECT 
    categoria,
    CASE 
        WHEN condicion_cnn >= 0.8 THEN 'Nuevo (0.8 - 1.0)'
        WHEN condicion_cnn >= 0.5 THEN 'Uso Moderado (0.5 - 0.79)'
        ELSE 'Viejo / Desgastado (0.0 - 0.49)'
    END AS estado_fisico,
    AVG(precio_final) AS precio_promedio_renta
FROM v_rentas
GROUP BY ROLLUP (categoria, estado_fisico)
ORDER BY categoria, estado_fisico;

-- -----------------------------------------------------------------------------
-- 4) SLICE: Filtro específico para el cálculo de Media Móvil Mensual
--    (Selecciona solo el año y mes actuales para actualizar el factor de precios)
-- -----------------------------------------------------------------------------
SELECT 
    categoria, 
    AVG(precio_final) AS precio_promedio_mes_actual,
    COUNT(*) AS transacciones_mes
FROM v_rentas
WHERE anio = 2026 AND mes = 'Julio'
GROUP BY categoria;

-- -----------------------------------------------------------------------------
-- 5) DICE: Vista filtrada bidimensional (Categorías Eléctricas en Guadalajara)
-- -----------------------------------------------------------------------------
SELECT 
    categoria,
    ciudad,
    mes,
    SUM(precio_final) AS ingresos
FROM v_rentas
WHERE sector = 'Eléctrico' 
  AND ciudad = 'Guadalajara'
GROUP BY categoria, ciudad, mes;
