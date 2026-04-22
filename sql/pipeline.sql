-- Datos
IF OBJECT_ID('dw.Datos', 'U') IS NOT NULL EXEC('DROP TABLE dw.Datos');

SELECT DISTINCT
    UTM_NORTE               AS UTM_Norte,
    UTM_ESTE                AS UTM_Este,
    HUSO                    AS Huso,
    CAUDAL                  AS Caudal,
    ALTURA_LIMNIMETRICA     AS Altura_Limnimetrica,
    TOTALIZADOR             AS Totalizador,
    NIVEL_FREATICO          AS Nivel_Freatico,
    FECHA_MEDICION          AS Fecha_Medicion,
    CAST(CASE WHEN NATURALEZA = 1 THEN 1 ELSE 0 END AS bit) AS es_pozo_subterraneo
INTO dw.Datos
FROM dw.Mediciones_full;

-- Cuencas_Regiones
IF OBJECT_ID('dw.Cuencas_Regiones', 'U') IS NOT NULL EXEC('DROP TABLE dw.Cuencas_Regiones');

SELECT DISTINCT
    COD_CUENCA        AS Cod_Cuenca,
    NOM_CUENCA        AS Nom_Cuenca,
    COD_SUBCUENCA     AS Cod_Subcuenca,
    NOM_SUBCUENCA     AS Nom_Subcuenca,
    COD_SUBSUBCUENCA  AS Cod_Subsubcuenca,
    NOM_SUBSUBCUENCA  AS Nom_Subsubcuenca,
    REGION            AS Cod_Region
INTO dw.Cuencas_Regiones
FROM dw.Mediciones_full
WHERE COD_CUENCA IS NOT NULL;

-- Informante
IF OBJECT_ID('dw.Informante', 'U') IS NOT NULL EXEC('DROP TABLE dw.Informante');

SELECT DISTINCT
    UTM_NORTE,
    UTM_ESTE,
    NOMB_INF,
    A_PAT_INF,
    A_MAT_INF,
    COD_CUENCA,
    COD_SUBCUENCA,
    COD_SUBSUBCUENCA,
    COUNT(*)            AS CANTIDAD_REPORTES,
    MAX(FECHA_MEDICION) AS ULTIMA_FECHA_MEDICION
INTO dw.Informante
FROM dw.Mediciones_full
WHERE INF IS NOT NULL
GROUP BY
    UTM_NORTE, UTM_ESTE,
    NOMB_INF, A_PAT_INF, A_MAT_INF,
    COD_CUENCA, COD_SUBCUENCA, COD_SUBSUBCUENCA;

-- Puntos_Mapa
IF OBJECT_ID('dw.Puntos_Mapa', 'U') IS NOT NULL EXEC('DROP TABLE dw.Puntos_Mapa');

SELECT
    UTM_NORTE                                                       AS UTM_Norte,
    UTM_ESTE                                                        AS UTM_Este,
    HUSO                                                            AS Huso,
    REGION                                                          AS Region,
    PROVINCIA                                                       AS Provincia,
    COMUNA                                                          AS Comuna,
    COD_CUENCA                                                      AS Cod_Cuenca,
    NOM_CUENCA                                                      AS Nom_Cuenca,
    COD_SUBCUENCA                                                   AS Cod_Subcuenca,
    NOM_SUBCUENCA                                                   AS Nom_Subcuenca,
    COD_SUBSUBCUENCA                                                AS Cod_Subsubcuenca,
    NOM_SUBSUBCUENCA                                                AS Nom_Subsubcuenca,
    AVG(CAST(CAUDAL AS float))                                      AS caudal_promedio,
    MIN(CAST(CAUDAL AS float))                                      AS caudal_minimo,
    MAX(CAST(CAUDAL AS float))                                      AS caudal_maximo,
    STDEV(CAST(CAUDAL AS float))                                    AS caudal_desviacion_estandar,
    COUNT(*)                                                        AS n_mediciones,
    CODIGO                                                          AS codigo,
    CAST(CASE WHEN MAX(NATURALEZA) = 1 THEN 1 ELSE 0 END AS bit)   AS es_pozo_subterraneo,
    COD_SECTOR_SHA,
    SECTOR_SHA,
    CAST(APR AS int)                                                AS APR,
    ID_JUNTA,
    CAST(PARTE_JUNTA AS int)                                        AS PARTE_JUNTA,
    CAST(REPRESENTA_JUNTA AS int)                                   AS REPRESENTA_JUNTA,
    CANAL_TRANSMISION,
    MAX(TIPO_DERECHO)                                               AS TIPO_DERECHO,
    MAX(VOLUMEN_ANUAL)                                              AS VOLUMEN_ANUAL,
    MAX(CAUDAL_ENERO)                                               AS CAUDAL_ENERO,
    MAX(CAUDAL_FEBRERO)                                             AS CAUDAL_FEBRERO,
    MAX(CAUDAL_MARZO)                                               AS CAUDAL_MARZO,
    MAX(CAUDAL_ABRIL)                                               AS CAUDAL_ABRIL,
    MAX(CAUDAL_MAYO)                                                AS CAUDAL_MAYO,
    MAX(CAUDAL_JUNIO)                                               AS CAUDAL_JUNIO,
    MAX(CAUDAL_JULIO)                                               AS CAUDAL_JULIO,
    MAX(CAUDAL_AGOSTO)                                              AS CAUDAL_AGOSTO,
    MAX(CAUDAL_SEPTIEMBRE)                                          AS CAUDAL_SEPTIEMBRE,
    MAX(CAUDAL_OCTUBRE)                                             AS CAUDAL_OCTUBRE,
    MAX(CAUDAL_NOVIEMBRE)                                           AS CAUDAL_NOVIEMBRE,
    MAX(CAUDAL_DICIEMBRE)                                           AS CAUDAL_DICIEMBRE
INTO dw.Puntos_Mapa
FROM dw.Mediciones_full
GROUP BY
    UTM_NORTE, UTM_ESTE, HUSO, REGION, PROVINCIA, COMUNA,
    COD_CUENCA, NOM_CUENCA, COD_SUBCUENCA, NOM_SUBCUENCA,
    COD_SUBSUBCUENCA, NOM_SUBSUBCUENCA, CODIGO,
    COD_SECTOR_SHA, SECTOR_SHA, APR, ID_JUNTA,
    PARTE_JUNTA, REPRESENTA_JUNTA, CANAL_TRANSMISION;

-- Series_Tiempo
IF OBJECT_ID('dw.Series_Tiempo', 'U') IS NOT NULL EXEC('DROP TABLE dw.Series_Tiempo');

SELECT
    COD_CUENCA,
    NOM_CUENCA,
    COD_SUBCUENCA,
    NOM_SUBCUENCA,
    COD_SUBSUBCUENCA,
    NOM_SUBSUBCUENCA,
    UTM_NORTE,
    UTM_ESTE,
    CAST(FECHA_MEDICION AS date)                AS FECHA_MEDICION,
    AVG(CAST(CAUDAL AS float))                  AS CAUDAL,
    AVG(CAST(ALTURA_LIMNIMETRICA AS float))     AS ALTURA_LIMNIMETRICA,
    AVG(CAST(NIVEL_FREATICO AS float))          AS NIVEL_FREATICO,
    MAX(TOTALIZADOR)                            AS TOTALIZADOR
INTO dw.Series_Tiempo
FROM dw.Mediciones_full
GROUP BY
    COD_CUENCA, NOM_CUENCA,
    COD_SUBCUENCA, NOM_SUBCUENCA,
    COD_SUBSUBCUENCA, NOM_SUBSUBCUENCA,
    UTM_NORTE, UTM_ESTE,
    CAST(FECHA_MEDICION AS date);

-- Cuenca_Stats
IF OBJECT_ID('dw.Cuenca_Stats', 'U') IS NOT NULL EXEC('DROP TABLE dw.Cuenca_Stats');

-- Deduplicate points with rights before aggregating to avoid double-counting
WITH derechos_por_subsubcuenca AS (
    SELECT
        COD_CUENCA, COD_SUBCUENCA, COD_SUBSUBCUENCA, REGION,
        COUNT(*)               AS puntos_con_derechos,
        SUM(VOLUMEN_ANUAL)     AS volumen_anual_total,
        SUM(CAUDAL_ENERO)      AS caudal_enero_sum,
        SUM(CAUDAL_FEBRERO)    AS caudal_febrero_sum,
        SUM(CAUDAL_MARZO)      AS caudal_marzo_sum,
        SUM(CAUDAL_ABRIL)      AS caudal_abril_sum,
        SUM(CAUDAL_MAYO)       AS caudal_mayo_sum,
        SUM(CAUDAL_JUNIO)      AS caudal_junio_sum,
        SUM(CAUDAL_JULIO)      AS caudal_julio_sum,
        SUM(CAUDAL_AGOSTO)     AS caudal_agosto_sum,
        SUM(CAUDAL_SEPTIEMBRE) AS caudal_septiembre_sum,
        SUM(CAUDAL_OCTUBRE)    AS caudal_octubre_sum,
        SUM(CAUDAL_NOVIEMBRE)  AS caudal_noviembre_sum,
        SUM(CAUDAL_DICIEMBRE)  AS caudal_diciembre_sum
    FROM (
        SELECT
            COD_CUENCA, COD_SUBCUENCA, COD_SUBSUBCUENCA, REGION,
            UTM_NORTE, UTM_ESTE,
            MAX(VOLUMEN_ANUAL)     AS VOLUMEN_ANUAL,
            MAX(CAUDAL_ENERO)      AS CAUDAL_ENERO,
            MAX(CAUDAL_FEBRERO)    AS CAUDAL_FEBRERO,
            MAX(CAUDAL_MARZO)      AS CAUDAL_MARZO,
            MAX(CAUDAL_ABRIL)      AS CAUDAL_ABRIL,
            MAX(CAUDAL_MAYO)       AS CAUDAL_MAYO,
            MAX(CAUDAL_JUNIO)      AS CAUDAL_JUNIO,
            MAX(CAUDAL_JULIO)      AS CAUDAL_JULIO,
            MAX(CAUDAL_AGOSTO)     AS CAUDAL_AGOSTO,
            MAX(CAUDAL_SEPTIEMBRE) AS CAUDAL_SEPTIEMBRE,
            MAX(CAUDAL_OCTUBRE)    AS CAUDAL_OCTUBRE,
            MAX(CAUDAL_NOVIEMBRE)  AS CAUDAL_NOVIEMBRE,
            MAX(CAUDAL_DICIEMBRE)  AS CAUDAL_DICIEMBRE
        FROM dw.Mediciones_full
        WHERE TIPO_DERECHO IS NOT NULL
        GROUP BY COD_CUENCA, COD_SUBCUENCA, COD_SUBSUBCUENCA, REGION, UTM_NORTE, UTM_ESTE
    ) pts_unicos
    GROUP BY COD_CUENCA, COD_SUBCUENCA, COD_SUBSUBCUENCA, REGION
)
SELECT
    mf.COD_CUENCA                                                          AS Cod_Cuenca,
    mf.NOM_CUENCA                                                          AS Nom_Cuenca,
    mf.COD_SUBCUENCA                                                       AS Cod_Subcuenca,
    mf.NOM_SUBCUENCA                                                       AS Nom_Subcuenca,
    mf.COD_SUBSUBCUENCA                                                    AS Cod_Subsubcuenca,
    mf.NOM_SUBSUBCUENCA                                                    AS Nom_Subsubcuenca,
    mf.REGION                                                              AS Cod_Region,
    AVG(CAST(mf.CAUDAL AS float))                                          AS caudal_promedio,
    MIN(CAST(mf.CAUDAL AS float))                                          AS caudal_minimo,
    MAX(CAST(mf.CAUDAL AS float))                                          AS caudal_maximo,
    STDEV(CAST(mf.CAUDAL AS float))                                        AS caudal_desviacion_estandar,
    COUNT(DISTINCT CONCAT(mf.UTM_NORTE, '-', mf.UTM_ESTE))                 AS total_puntos_unicos,
    COUNT(*)                                                                AS total_mediciones,
    MAX(ISNULL(dr.puntos_con_derechos, 0))                                  AS puntos_con_derechos,
    MAX(ISNULL(dr.volumen_anual_total, 0))                                  AS volumen_anual_total,
    MAX(ISNULL(dr.caudal_enero_sum, 0))                                     AS caudal_enero_sum,
    MAX(ISNULL(dr.caudal_febrero_sum, 0))                                   AS caudal_febrero_sum,
    MAX(ISNULL(dr.caudal_marzo_sum, 0))                                     AS caudal_marzo_sum,
    MAX(ISNULL(dr.caudal_abril_sum, 0))                                     AS caudal_abril_sum,
    MAX(ISNULL(dr.caudal_mayo_sum, 0))                                      AS caudal_mayo_sum,
    MAX(ISNULL(dr.caudal_junio_sum, 0))                                     AS caudal_junio_sum,
    MAX(ISNULL(dr.caudal_julio_sum, 0))                                     AS caudal_julio_sum,
    MAX(ISNULL(dr.caudal_agosto_sum, 0))                                    AS caudal_agosto_sum,
    MAX(ISNULL(dr.caudal_septiembre_sum, 0))                                AS caudal_septiembre_sum,
    MAX(ISNULL(dr.caudal_octubre_sum, 0))                                   AS caudal_octubre_sum,
    MAX(ISNULL(dr.caudal_noviembre_sum, 0))                                 AS caudal_noviembre_sum,
    MAX(ISNULL(dr.caudal_diciembre_sum, 0))                                 AS caudal_diciembre_sum
INTO dw.Cuenca_Stats
FROM dw.Mediciones_full mf
LEFT JOIN derechos_por_subsubcuenca dr
    ON  mf.COD_CUENCA       = dr.COD_CUENCA
    AND mf.COD_SUBCUENCA    = dr.COD_SUBCUENCA
    AND mf.COD_SUBSUBCUENCA = dr.COD_SUBSUBCUENCA
    AND mf.REGION           = dr.REGION
GROUP BY
    mf.COD_CUENCA, mf.NOM_CUENCA,
    mf.COD_SUBCUENCA, mf.NOM_SUBCUENCA,
    mf.COD_SUBSUBCUENCA, mf.NOM_SUBSUBCUENCA,
    mf.REGION;

-- Filtros_Reactivos_Stats
IF OBJECT_ID('dw.Filtros_Reactivos_Stats', 'U') IS NOT NULL EXEC('DROP TABLE dw.Filtros_Reactivos_Stats');

SELECT
    'cuenca'                                                        AS nivel,
    NOM_CUENCA                                                      AS nom_cuenca,
    NULL                                                            AS nom_subcuenca,
    MIN(CAST(CAUDAL AS float))                                      AS avgMin,
    MAX(CAST(CAUDAL AS float))                                      AS avgMax,
    COUNT(DISTINCT CONCAT(UTM_NORTE, '-', UTM_ESTE))                AS total_puntos
INTO dw.Filtros_Reactivos_Stats
FROM dw.Mediciones_full
GROUP BY NOM_CUENCA

UNION ALL

SELECT
    'subcuenca',
    NOM_CUENCA,
    NOM_SUBCUENCA,
    MIN(CAST(CAUDAL AS float)),
    MAX(CAST(CAUDAL AS float)),
    COUNT(DISTINCT CONCAT(UTM_NORTE, '-', UTM_ESTE))
FROM dw.Mediciones_full
GROUP BY NOM_CUENCA, NOM_SUBCUENCA;