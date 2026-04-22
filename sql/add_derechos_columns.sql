-- Migration: add water rights columns to pre-aggregated tables
-- Run once against the database, then update ETL to include these columns on rebuild.

-- ============================================================
-- 1. dw.Puntos_Mapa — one row per unique (UTM_Norte, UTM_Este)
-- ============================================================

ALTER TABLE dw.Puntos_Mapa ADD
    TIPO_DERECHO       INT          NULL,
    VOLUMEN_ANUAL      FLOAT        NULL,
    CAUDAL_ENERO       FLOAT        NULL,
    CAUDAL_FEBRERO     FLOAT        NULL,
    CAUDAL_MARZO       FLOAT        NULL,
    CAUDAL_ABRIL       FLOAT        NULL,
    CAUDAL_MAYO        FLOAT        NULL,
    CAUDAL_JUNIO       FLOAT        NULL,
    CAUDAL_JULIO       FLOAT        NULL,
    CAUDAL_AGOSTO      FLOAT        NULL,
    CAUDAL_SEPTIEMBRE  FLOAT        NULL,
    CAUDAL_OCTUBRE     FLOAT        NULL,
    CAUDAL_NOVIEMBRE   FLOAT        NULL,
    CAUDAL_DICIEMBRE   FLOAT        NULL;
GO

-- Populate from Mediciones_full (deduplicate points with MAX)
UPDATE pm
SET
    pm.TIPO_DERECHO      = src.TIPO_DERECHO,
    pm.VOLUMEN_ANUAL     = src.VOLUMEN_ANUAL,
    pm.CAUDAL_ENERO      = src.CAUDAL_ENERO,
    pm.CAUDAL_FEBRERO    = src.CAUDAL_FEBRERO,
    pm.CAUDAL_MARZO      = src.CAUDAL_MARZO,
    pm.CAUDAL_ABRIL      = src.CAUDAL_ABRIL,
    pm.CAUDAL_MAYO       = src.CAUDAL_MAYO,
    pm.CAUDAL_JUNIO      = src.CAUDAL_JUNIO,
    pm.CAUDAL_JULIO      = src.CAUDAL_JULIO,
    pm.CAUDAL_AGOSTO     = src.CAUDAL_AGOSTO,
    pm.CAUDAL_SEPTIEMBRE = src.CAUDAL_SEPTIEMBRE,
    pm.CAUDAL_OCTUBRE    = src.CAUDAL_OCTUBRE,
    pm.CAUDAL_NOVIEMBRE  = src.CAUDAL_NOVIEMBRE,
    pm.CAUDAL_DICIEMBRE  = src.CAUDAL_DICIEMBRE
FROM dw.Puntos_Mapa pm
INNER JOIN (
    SELECT
        UTM_NORTE, UTM_ESTE,
        MAX(TIPO_DERECHO)      AS TIPO_DERECHO,
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
    GROUP BY UTM_NORTE, UTM_ESTE
) AS src ON pm.UTM_Norte = src.UTM_NORTE AND pm.UTM_Este = src.UTM_ESTE;
GO

-- ============================================================
-- 2. dw.Cuenca_Stats — one row per (Cod_Cuenca, Cod_Subcuenca, Cod_Subsubcuenca)
-- ============================================================

ALTER TABLE dw.Cuenca_Stats ADD
    puntos_con_derechos    INT    NULL,
    volumen_anual_total    FLOAT  NULL,
    caudal_enero_sum       FLOAT  NULL,
    caudal_febrero_sum     FLOAT  NULL,
    caudal_marzo_sum       FLOAT  NULL,
    caudal_abril_sum       FLOAT  NULL,
    caudal_mayo_sum        FLOAT  NULL,
    caudal_junio_sum       FLOAT  NULL,
    caudal_julio_sum       FLOAT  NULL,
    caudal_agosto_sum      FLOAT  NULL,
    caudal_septiembre_sum  FLOAT  NULL,
    caudal_octubre_sum     FLOAT  NULL,
    caudal_noviembre_sum   FLOAT  NULL,
    caudal_diciembre_sum   FLOAT  NULL;
GO

-- Populate: aggregate unique points per (cuenca, subcuenca, subsubcuenca)
UPDATE cs
SET
    cs.puntos_con_derechos   = src.puntos_con_derechos,
    cs.volumen_anual_total   = src.volumen_anual_total,
    cs.caudal_enero_sum      = src.caudal_enero_sum,
    cs.caudal_febrero_sum    = src.caudal_febrero_sum,
    cs.caudal_marzo_sum      = src.caudal_marzo_sum,
    cs.caudal_abril_sum      = src.caudal_abril_sum,
    cs.caudal_mayo_sum       = src.caudal_mayo_sum,
    cs.caudal_junio_sum      = src.caudal_junio_sum,
    cs.caudal_julio_sum      = src.caudal_julio_sum,
    cs.caudal_agosto_sum     = src.caudal_agosto_sum,
    cs.caudal_septiembre_sum = src.caudal_septiembre_sum,
    cs.caudal_octubre_sum    = src.caudal_octubre_sum,
    cs.caudal_noviembre_sum  = src.caudal_noviembre_sum,
    cs.caudal_diciembre_sum  = src.caudal_diciembre_sum
FROM dw.Cuenca_Stats cs
INNER JOIN (
    SELECT
        COD_CUENCA, COD_SUBCUENCA, COD_SUBSUBCUENCA,
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
        -- deduplicate points within each subsubcuenca first
        SELECT
            COD_CUENCA, COD_SUBCUENCA, COD_SUBSUBCUENCA,
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
        GROUP BY COD_CUENCA, COD_SUBCUENCA, COD_SUBSUBCUENCA, UTM_NORTE, UTM_ESTE
    ) AS pts_unicos
    GROUP BY COD_CUENCA, COD_SUBCUENCA, COD_SUBSUBCUENCA
) AS src ON cs.Cod_Cuenca        = src.COD_CUENCA
        AND cs.Cod_Subcuenca     = src.COD_SUBCUENCA
        AND cs.Cod_Subsubcuenca  = src.COD_SUBSUBCUENCA;
GO
