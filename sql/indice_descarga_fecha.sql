/* ===========================================================================
   Arreglo del preview lento: IX_Mediciones_full_Codigo_Fecha
   ---------------------------------------------------------------------------
   /api/mediciones/preview tarda ~22 s en producción pese al índice sobre
   CODIGO, y el frontend corta a los 30 s.

   POR QUÉ. El preview corre dos consultas. La de las 10 filas es un seek y
   vuela. La del resumen pide COUNT(*), MIN(FECHA_MEDICION) y
   MAX(FECHA_MEDICION); el índice actual sólo tiene CODIGO, así que localiza
   las ~45.511 filas de la obra pero tiene que ir al heap una por una para leer
   la fecha. Eso es el scan que quedaba.

   (El benchmark de indice_descarga_diagnostico.sql medía sólo COUNT(*), sin
   MIN/MAX, y por eso dio 9 ms: medía una consulta más barata que la real.)

   CÓMO. Con FECHA_MEDICION como segunda clave, MIN y MAX son el primer y el
   último registro del rango del índice: dos seeks, sin tocar la tabla.

   ESTRATEGIA. Se crea el índice nuevo ANTES de borrar el viejo, en vez de usar
   DROP_EXISTING, por dos razones: el índice actual sigue sirviendo consultas
   mientras el nuevo se construye, y DROP_EXISTING no admite RESUMABLE = ON.
   El costo es tener los dos índices conviviendo unos minutos.

   Corré el archivo entero en una sola conexión (F5).
   =========================================================================== */

SET NOCOUNT ON;

IF OBJECT_ID('tempdb..#t') IS NOT NULL DROP TABLE #t;
CREATE TABLE #t (etapa varchar(10), ms bigint, filas bigint NULL);
GO


/* ---------------------------------------------------------------------------
   1. ANTES — la consulta EXACTA que hace el endpoint (con MIN/MAX)
   --------------------------------------------------------------------------- */
DECLARE @i tinyint = 1;
WHILE @i <= 2
BEGIN
    DECLARE @t0 datetime2 = SYSUTCDATETIME();
    DECLARE @total bigint, @pri datetime2, @ult datetime2;

    SELECT @total = COUNT(*),
           @pri   = MIN(FECHA_MEDICION),
           @ult   = MAX(FECHA_MEDICION)
    FROM dw.Mediciones_full
    WHERE CODIGO = 'OB-0202-591';

    INSERT #t VALUES ('ANTES', DATEDIFF_BIG(millisecond, @t0, SYSUTCDATETIME()), @total);
    SET @i += 1;
END
GO


/* ---------------------------------------------------------------------------
   2. Crear el índice nuevo (el viejo sigue en pie y sirviendo)
   --------------------------------------------------------------------------- */
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = 'IX_Mediciones_full_Codigo_Fecha'
      AND object_id = OBJECT_ID('dw.Mediciones_full')
)
CREATE NONCLUSTERED INDEX IX_Mediciones_full_Codigo_Fecha
ON dw.Mediciones_full (CODIGO, FECHA_MEDICION)
WITH (ONLINE = ON, RESUMABLE = ON, MAXDOP = 2);
GO

/* Si se corta:  ALTER INDEX IX_Mediciones_full_Codigo_Fecha ON dw.Mediciones_full RESUME;
   Ver avance:   SELECT name, state_desc, percent_complete FROM sys.index_resumable_operations; */


/* ---------------------------------------------------------------------------
   3. DESPUÉS
   --------------------------------------------------------------------------- */
DECLARE @j tinyint = 1;
WHILE @j <= 2
BEGIN
    DECLARE @t1 datetime2 = SYSUTCDATETIME();
    DECLARE @total2 bigint, @pri2 datetime2, @ult2 datetime2;

    SELECT @total2 = COUNT(*),
           @pri2   = MIN(FECHA_MEDICION),
           @ult2   = MAX(FECHA_MEDICION)
    FROM dw.Mediciones_full
    WHERE CODIGO = 'OB-0202-591';

    INSERT #t VALUES ('DESPUES', DATEDIFF_BIG(millisecond, @t1, SYSUTCDATETIME()), @total2);
    SET @j += 1;
END
GO


/* ---------------------------------------------------------------------------
   4. Comparación (2ª corrida de cada una, sin arranque en frío)
   --------------------------------------------------------------------------- */
SELECT * FROM #t;

SELECT a.ms AS ms_antes, d.ms AS ms_despues,
       CAST(1.0 * a.ms / NULLIF(d.ms, 0) AS decimal(8,1)) AS veces_mas_rapido
FROM (SELECT TOP 1 ms FROM #t WHERE etapa = 'ANTES'   ORDER BY ms) a
CROSS JOIN (SELECT TOP 1 ms FROM #t WHERE etapa = 'DESPUES' ORDER BY ms) d;


/* ---------------------------------------------------------------------------
   5. Sólo si el paso 4 confirma la mejora: borrar el índice viejo
   ---------------------------------------------------------------------------
   El nuevo lo reemplaza por completo — (CODIGO, FECHA_MEDICION) sirve para
   todo lo que servía (CODIGO), porque CODIGO es la primera clave. Dejar los
   dos duplica el costo de escritura sin ganar nada.
   --------------------------------------------------------------------------- */
-- DROP INDEX IX_Mediciones_full_Codigo ON dw.Mediciones_full;

-- Verificación final
SELECT i.name AS indice, c.name AS columna, ic.key_ordinal AS orden
FROM sys.indexes i
JOIN sys.index_columns ic ON ic.object_id = i.object_id AND ic.index_id = i.index_id
JOIN sys.columns c ON c.object_id = i.object_id AND c.column_id = ic.column_id
WHERE i.object_id = OBJECT_ID('dw.Mediciones_full')
ORDER BY i.name, ic.key_ordinal;
