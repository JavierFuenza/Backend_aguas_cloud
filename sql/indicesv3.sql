-- =========================================================================
-- ÍNDICES v3 — basados en benchmark de endpoints en producción
-- Tablas sin índices: dw.Series_Tiempo, dw.Cuenca_Stats
-- Puntos_Mapa: falta índice con Cod_Cuenca como columna líder para /shacs filtrado
-- =========================================================================


-- =========================================================================
-- 1. dw.Series_Tiempo (sin índices — causa lentitud en todos los endpoints de series)
-- =========================================================================

-- Búsquedas por punto (/puntos/series_de_tiempo/*)
-- Cubre WHERE UTM_NORTE = ? AND UTM_ESTE = ? con ORDER BY FECHA_MEDICION
CREATE NONCLUSTERED INDEX IX_Series_Tiempo_Punto_Fecha
ON dw.Series_Tiempo (UTM_NORTE, UTM_ESTE, FECHA_MEDICION DESC)
INCLUDE (CAUDAL, ALTURA_LIMNIMETRICA, NIVEL_FREATICO, TOTALIZADOR,
         COD_CUENCA, COD_SUBCUENCA, COD_SUBSUBCUENCA);

-- Búsquedas por cuenca (/cuencas/cuenca/series_de_tiempo/*)
-- Cubre WHERE COD_CUENCA = ? (+ optional fecha range)
CREATE NONCLUSTERED INDEX IX_Series_Tiempo_Cuenca_Fecha
ON dw.Series_Tiempo (COD_CUENCA, FECHA_MEDICION DESC)
INCLUDE (UTM_NORTE, UTM_ESTE, CAUDAL, ALTURA_LIMNIMETRICA, NIVEL_FREATICO, TOTALIZADOR,
         COD_SUBCUENCA, COD_SUBSUBCUENCA);

-- Búsquedas por subcuenca (/cuencas/subcuenca/series_de_tiempo/*)
CREATE NONCLUSTERED INDEX IX_Series_Tiempo_Subcuenca_Fecha
ON dw.Series_Tiempo (COD_SUBCUENCA, FECHA_MEDICION DESC)
INCLUDE (UTM_NORTE, UTM_ESTE, CAUDAL, ALTURA_LIMNIMETRICA, NIVEL_FREATICO, TOTALIZADOR,
         COD_CUENCA, COD_SUBSUBCUENCA);

-- Búsquedas por subsubcuenca (/cuencas/subsubcuenca/series_de_tiempo/*)
CREATE NONCLUSTERED INDEX IX_Series_Tiempo_Subsubcuenca_Fecha
ON dw.Series_Tiempo (COD_SUBSUBCUENCA, FECHA_MEDICION DESC)
INCLUDE (UTM_NORTE, UTM_ESTE, CAUDAL, ALTURA_LIMNIMETRICA, NIVEL_FREATICO, TOTALIZADOR,
         COD_CUENCA, COD_SUBCUENCA);

-- Búsquedas por SHAC (/cuencas/shac/series_de_tiempo/*)
-- Series_Tiempo no tiene COD_SECTOR_SHA directamente — el router joins Puntos_Mapa
-- Este índice acelera el JOIN por coordenadas en ese caso
CREATE NONCLUSTERED INDEX IX_Series_Tiempo_Coords
ON dw.Series_Tiempo (UTM_NORTE, UTM_ESTE)
INCLUDE (FECHA_MEDICION, CAUDAL, ALTURA_LIMNIMETRICA, NIVEL_FREATICO, TOTALIZADOR);


-- =========================================================================
-- 2. dw.Cuenca_Stats (sin índices — causa lentitud en /cuencas/stats y /derechos)
-- =========================================================================

-- Cubre /cuencas/stats?cod_cuenca=X y /cuencas/derechos + /subcuencas/derechos
-- Incluye todas las columnas de derechos para evitar key lookup
CREATE NONCLUSTERED INDEX IX_Cuenca_Stats_Jerarquia
ON dw.Cuenca_Stats (Cod_Cuenca, Cod_Subcuenca, Cod_Subsubcuenca)
INCLUDE (Nom_Cuenca, Nom_Subcuenca, Nom_Subsubcuenca, Cod_Region,
         caudal_promedio, caudal_minimo, caudal_maximo, caudal_desviacion_estandar,
         total_puntos_unicos, total_mediciones,
         puntos_con_derechos, volumen_anual_total,
         caudal_enero_sum, caudal_febrero_sum, caudal_marzo_sum, caudal_abril_sum,
         caudal_mayo_sum, caudal_junio_sum, caudal_julio_sum, caudal_agosto_sum,
         caudal_septiembre_sum, caudal_octubre_sum, caudal_noviembre_sum, caudal_diciembre_sum);


-- =========================================================================
-- 3. dw.Puntos_Mapa — índice complementario para /shacs filtrado por cuenca
-- =========================================================================
-- IX_Puntos_Mapa_Filtros_v2 lidera con Region, por lo que Cod_Cuenca-only queries
-- no pueden hacer seek eficiente. Este índice cubre el nuevo /shacs?cod_cuenca=X.

CREATE NONCLUSTERED INDEX IX_Puntos_Mapa_SHAC_PorCuenca
ON dw.Puntos_Mapa (Cod_Cuenca, Cod_Subcuenca, COD_SECTOR_SHA)
INCLUDE (SECTOR_SHA);

-- ---------------------------------------------------------------------------
-- dw.Mediciones_full
-- ---------------------------------------------------------------------------
-- La tabla base es un heap con dos índices que NO están declarados en este
-- archivo (creados a mano en la base): IX_temp_export sobre
-- (REGION, FECHA_MEDICION) e IX_Mediciones_full_Punto_Fecha sobre
-- (UTM_NORTE, UTM_ESTE, FECHA_MEDICION). En ninguno CODIGO es columna
-- principal, así que la descarga de mediciones por obra
-- (/api/mediciones/descarga) hace un scan de los ~71,8 M de registros.
--
-- No lleva INCLUDE: la descarga proyecta hasta 24 columnas y cubrirlas todas
-- duplicaría la tabla. Con ~11.800 filas por obra (0,016 % del total) los key
-- lookups sobre el heap salen mucho más baratos que el scan.
--
-- FECHA_MEDICION va como segunda clave porque /mediciones/preview pide
-- MIN(FECHA_MEDICION) y MAX(FECHA_MEDICION) junto con el COUNT. Con sólo
-- CODIGO el motor localiza las filas por índice pero va al heap a leer cada
-- fecha, y eso costaba ~22 s por preview.
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = 'IX_Mediciones_full_Codigo_Fecha'
      AND object_id = OBJECT_ID('dw.Mediciones_full')
)
CREATE NONCLUSTERED INDEX IX_Mediciones_full_Codigo_Fecha
ON dw.Mediciones_full (CODIGO, FECHA_MEDICION);
