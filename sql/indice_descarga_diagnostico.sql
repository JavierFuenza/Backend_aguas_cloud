/* ===========================================================================
   IX_Mediciones_full_Codigo — medición antes/después y creación
   ---------------------------------------------------------------------------
   Sirve a la descarga de mediciones por obra (/api/mediciones/descarga), que
   filtra por CODIGO. La tabla es un heap de ~71,8 M de filas cuyos índices
   actuales —IX_temp_export (REGION, FECHA_MEDICION) e
   IX_Mediciones_full_Punto_Fecha (UTM_NORTE, UTM_ESTE, FECHA_MEDICION)— no
   sirven para ese filtro, así que hoy cada descarga es un scan completo.

   MOTOR: Azure SQL Database, GP_S_Gen5_4 (General Purpose serverless, 0.5-4
   vCores, auto-pause 60 min). No es Synapse, pese al nombre de las variables
   de entorno.

   CÓMO EJECUTARLO
     Corré el archivo completo, de una, en UNA sola conexión (F5 en la
     extensión mssql). Es importante que sea una sola sesión: los tiempos se
     acumulan en la tabla temporal #tiempos, que vive mientras dure la
     conexión. El query editor del portal de Azure no sirve para esto porque
     corta por timeout; la extensión mssql tiene executionTimeout = 0.

     El último SELECT es el que te interesa: la comparación.
   =========================================================================== */

SET NOCOUNT ON;

IF OBJECT_ID('tempdb..#tiempos') IS NOT NULL DROP TABLE #tiempos;
CREATE TABLE #tiempos (
    etapa    varchar(10),
    consulta varchar(20),
    corrida  tinyint,
    ms       bigint,
    filas    bigint NULL
);

DECLARE @obra varchar(50) = 'OB-0202-591';  -- cambiala si querés otra obra
GO


/* ===========================================================================
   1. SIN ÍNDICE
   ---------------------------------------------------------------------------
   Cada consulta se corre dos veces. La primera paga el arranque en frío (y con
   auto-pause a 60 minutos puede pagar además el resume de la base); la que
   vale para comparar es la segunda.
   =========================================================================== */

DECLARE @i tinyint = 1;
WHILE @i <= 2
BEGIN
    DECLARE @t0 datetime2 = SYSUTCDATETIME();
    DECLARE @total bigint;

    SELECT @total = COUNT(*)
    FROM dw.Mediciones_full
    WHERE CODIGO = 'OB-0202-591';

    INSERT #tiempos VALUES
        ('SIN', 'resumen', @i, DATEDIFF_BIG(millisecond, @t0, SYSUTCDATETIME()), @total);

    DECLARE @t1 datetime2 = SYSUTCDATETIME();
    DECLARE @sink float;

    SELECT @sink = MAX(TRY_CAST(CAUDAL AS float))
    FROM dw.Mediciones_full
    WHERE CODIGO = 'OB-0202-591';

    INSERT #tiempos VALUES
        ('SIN', 'lectura', @i, DATEDIFF_BIG(millisecond, @t1, SYSUTCDATETIME()), NULL);

    SET @i += 1;
END
GO


/* ===========================================================================
   2. CREAR EL ÍNDICE
   ---------------------------------------------------------------------------
   ONLINE = ON     la tabla sigue disponible mientras se construye. La base
                   está en producción y el visualizador la consulta.
   RESUMABLE = ON  si se corta, se retoma con RESUME en vez de empezar de cero.
   MAXDOP = 2      de los 4 vCores deja dos libres, para que el visualizador
                   siga respondiendo durante la construcción.
   =========================================================================== */

DECLARE @t2 datetime2 = SYSUTCDATETIME();

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = 'IX_Mediciones_full_Codigo'
      AND object_id = OBJECT_ID('dw.Mediciones_full')
)
CREATE NONCLUSTERED INDEX IX_Mediciones_full_Codigo
ON dw.Mediciones_full (CODIGO)
WITH (ONLINE = ON, RESUMABLE = ON, MAXDOP = 2);

INSERT #tiempos VALUES
    ('INDICE', 'construccion', 1, DATEDIFF_BIG(millisecond, @t2, SYSUTCDATETIME()), NULL);
GO

/* Si la construcción se corta, NO relances el bloque de arriba. Retomá con:
     ALTER INDEX IX_Mediciones_full_Codigo ON dw.Mediciones_full RESUME;
   Para ver el avance desde OTRA pestaña mientras corre:
     SELECT name, state_desc, percent_complete FROM sys.index_resumable_operations;
   Para abandonarla del todo:
     ALTER INDEX IX_Mediciones_full_Codigo ON dw.Mediciones_full ABORT;
   Ojo: un índice resumable pausado retiene espacio y bloquea algunas
   operaciones sobre la tabla. No lo dejes pausado por días. */


/* ===========================================================================
   3. CON ÍNDICE
   =========================================================================== */

DECLARE @j tinyint = 1;
WHILE @j <= 2
BEGIN
    DECLARE @t3 datetime2 = SYSUTCDATETIME();
    DECLARE @total2 bigint;

    SELECT @total2 = COUNT(*)
    FROM dw.Mediciones_full
    WHERE CODIGO = 'OB-0202-591';

    INSERT #tiempos VALUES
        ('CON', 'resumen', @j, DATEDIFF_BIG(millisecond, @t3, SYSUTCDATETIME()), @total2);

    DECLARE @t4 datetime2 = SYSUTCDATETIME();
    DECLARE @sink2 float;

    SELECT @sink2 = MAX(TRY_CAST(CAUDAL AS float))
    FROM dw.Mediciones_full
    WHERE CODIGO = 'OB-0202-591';

    INSERT #tiempos VALUES
        ('CON', 'lectura', @j, DATEDIFF_BIG(millisecond, @t4, SYSUTCDATETIME()), NULL);

    SET @j += 1;
END
GO


/* ===========================================================================
   4. RESULTADO
   =========================================================================== */

-- 4.a Detalle de todas las corridas
SELECT etapa, consulta, corrida, ms, filas
FROM #tiempos
ORDER BY CASE etapa WHEN 'SIN' THEN 1 WHEN 'INDICE' THEN 2 ELSE 3 END,
         consulta, corrida;

-- 4.b La comparación: sólo la 2ª corrida de cada una, que es la que no paga
--     arranque en frío.
SELECT
    s.consulta,
    s.ms                                   AS ms_sin_indice,
    c.ms                                   AS ms_con_indice,
    s.ms - c.ms                            AS ms_ahorrados,
    CAST(100.0 * (s.ms - c.ms) / NULLIF(s.ms, 0) AS decimal(5,1)) AS pct_mejora,
    CAST(1.0 * s.ms / NULLIF(c.ms, 0) AS decimal(8,1))            AS veces_mas_rapido
FROM #tiempos s
JOIN #tiempos c ON c.consulta = s.consulta AND c.etapa = 'CON' AND c.corrida = 2
WHERE s.etapa = 'SIN' AND s.corrida = 2;

-- 4.c Cuánto tardó construirlo
SELECT ms / 1000.0 AS segundos_construccion
FROM #tiempos WHERE etapa = 'INDICE';

-- 4.d El índice quedó creado
SELECT i.name AS indice, i.type_desc AS tipo, c.name AS columna
FROM sys.indexes i
JOIN sys.index_columns ic ON ic.object_id = i.object_id AND ic.index_id = i.index_id
JOIN sys.columns c ON c.object_id = i.object_id AND c.column_id = ic.column_id
WHERE i.object_id = OBJECT_ID('dw.Mediciones_full')
  AND i.name = 'IX_Mediciones_full_Codigo';


/* ===========================================================================
   5. APÉNDICE — columnas que proyecta el backend
   ---------------------------------------------------------------------------
   Los alias de api/routers/descargas.py se dedujeron de pipeline.sql y
   generar_top_usuarios.py, sin poder ejecutar nada contra la base. Deberían
   volver 24 filas; si falta alguna, hay un alias que corregir antes de
   desplegar.
   =========================================================================== */
SELECT c.name AS columna, t.name AS tipo
FROM sys.columns c
JOIN sys.types t ON t.user_type_id = c.user_type_id
WHERE c.object_id = OBJECT_ID('dw.Mediciones_full')
  AND c.name IN (
      'CODIGO', 'FECHA_MEDICION', 'NOMBRE_COMPLETO_USUARIO',
      'CAUDAL', 'TOTALIZADOR', 'ALTURA_LIMNIMETRICA', 'NIVEL_FREATICO',
      'NATURALEZA', 'CANAL_TRANSMISION', 'APR',
      'UTM_NORTE', 'UTM_ESTE', 'HUSO',
      'REGION', 'PROVINCIA', 'COMUNA',
      'NOM_CUENCA', 'NOM_SUBCUENCA', 'NOM_SUBSUBCUENCA',
      'SECTOR_SHA', 'COD_SECTOR_SHA', 'ID_JUNTA',
      'TIPO_DERECHO', 'VOLUMEN_ANUAL'
  )
ORDER BY c.name;


-- Deshacer, si hiciera falta:
-- DROP INDEX IX_Mediciones_full_Codigo ON dw.Mediciones_full;
