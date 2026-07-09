-- =============================================================================
-- Capa Analítica OLAP (DuckDB) · Esquema Estrella
-- ToolShare · DuckDB
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1) Dimensiones
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_tiempo (
    tiempo_key INTEGER PRIMARY KEY,
    fecha DATE NOT NULL,
    dia INT NOT NULL,
    mes INT NOT NULL,
    anio INT NOT NULL,
    nombre_mes VARCHAR(20) NOT NULL,
    trimestre VARCHAR(5) NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_categoria (
    categoria_key INTEGER PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    tipo_general VARCHAR(100) NOT NULL  -- Ej: 'Eléctrico', 'Manual', 'Corte'
);

CREATE TABLE IF NOT EXISTS dim_zona (
    zona_key INTEGER PRIMARY KEY,
    ciudad VARCHAR(100) NOT NULL,
    estado VARCHAR(100) NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_usuario (
    usuario_key INTEGER PRIMARY KEY,
    nombre VARCHAR(150) NOT NULL,
    tipo_usuario VARCHAR(50) NOT NULL -- 'owner', 'requester', 'admin'
);

-- -----------------------------------------------------------------------------
-- 2) Tabla de Hechos: fact_rentas
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_rentas (
    tiempo_key INTEGER,
    categoria_key INTEGER,
    zona_key INTEGER,
    usuario_key INTEGER,
    precio_final DECIMAL(10,2) NOT NULL,
    condicion_cnn FLOAT NOT NULL,
    total_dias INT NOT NULL
);

-- -----------------------------------------------------------------------------
-- 3) Vista OLAP: Cubo Lógico para Consultas Multidimensionales
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_rentas AS
SELECT 
    t.anio, 
    t.nombre_mes AS mes, 
    c.nombre AS categoria, 
    c.tipo_general AS sector,
    z.ciudad, 
    z.estado,
    u.nombre AS usuario,
    u.tipo_usuario,
    f.precio_final, 
    f.condicion_cnn,
    f.total_dias
FROM fact_rentas f
JOIN dim_tiempo t ON f.tiempo_key = t.tiempo_key
JOIN dim_categoria c ON f.categoria_key = c.categoria_key
JOIN dim_zona z ON f.zona_key = z.zona_key
JOIN dim_usuario u ON f.usuario_key = u.usuario_key;
