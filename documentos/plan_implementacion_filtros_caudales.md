# Plan de Implementación de Nuevas Funcionalidades Hidrológicas

El siguiente plan de implementación abarca todas las funcionalidades especificadas para expandir los filtros, campos de datos de los puntos y las métricas en los gráficos de la aplicación.

> **Última revisión**: 2026-04-13  
> **Estado general**: ~85% implementado. Quedan correcciones puntuales, índices y endpoints complementarios.

### Ubicación de repositorios
- **Backend**: `/home/javier/Github/Backend_aguas_cloud`
- **Frontend**: `/home/javier/Github/frontend-aguas` (Astro + React + Leaflet)
- **Plan frontend complementario**: [plan_implementacion_frontend.md](./plan_implementacion_frontend.md)

---

## Resumen de Estado

| Componente | Estado |
|---|---|
| Schemas Pydantic (`PuntoResponse`, `PuntoInfoResponse`, `TimeSeriesPoint`) | ✅ Completado |
| Filtros SHAC/APR/Junta en `/puntos` y `/puntos/count` | ✅ Completado |
| `/puntos/info` con JOIN a `Mediciones_full` | ✅ Completado |
| `SUM(caudal)` y totalizador en series por cuenca/subcuenca/subsubcuenca | ✅ Completado |
| Endpoint SHAC series temporales (caudal) | ✅ Completado |
| SQL de creación de `Puntos_Mapa` y `Series_Tiempo` | ✅ Completado |
| Índices para nuevos filtros en BD | ❌ Pendiente |
| Bug: `cod_subsubcuenca` no retornado en `/puntos` | ❌ Pendiente |
| `filtros_aplicados` incompleto en `/puntos/count` | ❌ Pendiente |
| Endpoints de listado SHAC / Juntas | ❌ Pendiente |
| Series temporales SHAC (altura/nivel freático) | ❌ Pendiente |
| Documentación de `Mediciones_full` | ❌ Pendiente |

---

## Cambios Completados

### `models/schemas.py`

#### [MODIFY] schemas.py — ✅ COMPLETADO
- **`PuntoResponse`**: Incluye `sector_sha`, `apr`, `id_junta` y `cod_subsubcuenca`.
- **`PuntoInfoResponse`**: Incluye `canal_transmision`, `parte_junta`, `representa_junta`, `sector_sha`, `apr`, `id_junta`, `cod_subsubcuenca`.
- **`TimeSeriesPoint`**: Incluye `caudal_sumado`, `totalizador_sumado`, `totalizador_max`.

---

### `api/routers/puntos_de_medicion.py`

#### [MODIFY] puntos_de_medicion.py — ✅ COMPLETADO
- **`/puntos/count`** y **`/puntos`**: Aceptan `cod_subsubcuenca`, `shac` (COD_SECTOR_SHA), `apr`, `id_junta` como query params.
- **`/puntos/info`**: Obtiene `CANAL_TRANSMISION`, `SECTOR_SHA`, `APR`, `ID_JUNTA`, `PARTE_JUNTA`, `REPRESENTA_JUNTA` mediante JOIN a `dw.Mediciones_full` ordenado por `FECHA_MEDICION DESC`.

---

### `api/routers/series_temporales.py`

#### [MODIFY] series_temporales.py — ✅ COMPLETADO
- Consultas SQL de caudal por **cuenca, subcuenca, subsubcuenca y SHAC** incluyen:
  - `SUM(CAST(s.CAUDAL AS FLOAT)) AS caudal_sumado`
  - `SUM(CAST(s.TOTALIZADOR AS BIGINT)) AS totalizador_sumado`
  - `MAX(CAST(s.TOTALIZADOR AS BIGINT)) AS totalizador_max`
- Endpoint **`/cuencas/shac/series_de_tiempo/caudal`** implementado con JOIN a `dw.Puntos_Mapa`.

---

### `sql/Actualizar_tablas.sql`

#### [MODIFY] Actualizar_tablas.sql — ✅ COMPLETADO
- `dw.Puntos_Mapa` se regenera con columnas: `COD_SECTOR_SHA`, `SECTOR_SHA`, `APR`, `ID_JUNTA`, `PARTE_JUNTA`, `REPRESENTA_JUNTA`, `CANAL_TRANSMISION`.
- `dw.Series_Tiempo` se regenera con columna `TOTALIZADOR`.

---

## Cambios Pendientes

### 1. Bug: `cod_subsubcuenca` no retornado en `/puntos`

**Archivo**: `api/routers/puntos_de_medicion.py`  
**Problema**: El schema `PuntoResponse` define `cod_subsubcuenca`, pero la query SQL del endpoint `/puntos` no selecciona `Cod_Subsubcuenca` de la BD, y el mapeo de respuesta tampoco lo incluye. Siempre devuelve `null`.

**Solución**: Agregar `Cod_Subsubcuenca` al SELECT de `puntos_query` y mapearlo en la respuesta:
```python
# En el SELECT agregar:
#     Cod_Subsubcuenca,

# En el mapeo agregar:
#     "cod_subsubcuenca": punto.get("Cod_Subsubcuenca"),
```

---

### 2. `filtros_aplicados` incompleto en `/puntos/count`

**Archivo**: `api/routers/puntos_de_medicion.py`  
**Problema**: La respuesta de `/puntos/count` incluye un diccionario `filtros_aplicados` que no refleja los nuevos filtros (`shac`, `apr`, `id_junta`, `cod_subsubcuenca`, `codigo_obra`).

**Solución**: Actualizar el diccionario de respuesta:
```python
"filtros_aplicados": {
    "region": region,
    "cod_cuenca": cod_cuenca,
    "cod_subcuenca": cod_subcuenca,
    "cod_subsubcuenca": cod_subsubcuenca,
    "filtro_null_subcuenca": filtro_null_subcuenca,
    "caudal_minimo": caudal_minimo,
    "caudal_maximo": caudal_maximo,
    "pozo": pozo,
    "codigo_obra": codigo_obra,
    "shac": shac,
    "apr": apr,
    "id_junta": id_junta
}
```

---

### 3. Índices para nuevos filtros en BD

**Archivo**: `sql/indices.sql` (o nuevo `sql/indicesv2.sql`)  
**Problema**: Los índices existentes en `dw.Puntos_Mapa` solo cubren `Region`, `Cod_Cuenca`, `Cod_Subcuenca`. Los nuevos filtros por `COD_SECTOR_SHA`, `APR` e `ID_JUNTA` harán table scans.

**Solución**: Crear índices adicionales:
```sql
-- Índice para filtro por SHAC
CREATE NONCLUSTERED INDEX IX_Puntos_Mapa_SHAC 
ON dw.Puntos_Mapa (COD_SECTOR_SHA)
INCLUDE (UTM_Norte, UTM_Este, caudal_promedio);

-- Índice para filtro por Junta
CREATE NONCLUSTERED INDEX IX_Puntos_Mapa_Junta 
ON dw.Puntos_Mapa (ID_JUNTA)
INCLUDE (UTM_Norte, UTM_Este, caudal_promedio);

-- Índice compuesto para filtros combinados frecuentes
CREATE NONCLUSTERED INDEX IX_Puntos_Mapa_Filtros_v2 
ON dw.Puntos_Mapa (Region, Cod_Cuenca, Cod_Subcuenca, Cod_Subsubcuenca, COD_SECTOR_SHA)
INCLUDE (UTM_Norte, UTM_Este, Huso, es_pozo_subterraneo, SECTOR_SHA, APR, ID_JUNTA, caudal_promedio);
```

> **Nota**: APR es bit y tiene baja cardinalidad (solo 0/1), por lo que un índice dedicado no aporta mucho. Se incluye como columna INCLUDE en el índice compuesto.

---

### 4. Endpoints de listado SHAC y Juntas

**Archivo**: `api/routers/cuencas_hidrograficas.py` (o nuevo router)  
**Problema**: El frontend necesita poblar dropdowns de filtro para SHAC y Juntas. Existe `/cuencas` para cuencas, pero no hay equivalente para estos nuevos filtros.

**Solución propuesta**:

#### [NEW] Endpoint `/shacs`
```
GET /shacs
Retorna: lista de { cod_sector_sha, sector_sha, total_puntos }
Fuente: SELECT DISTINCT COD_SECTOR_SHA, SECTOR_SHA, COUNT(*) FROM dw.Puntos_Mapa GROUP BY ...
```

#### [NEW] Endpoint `/juntas`
```
GET /juntas
Retorna: lista de { id_junta, total_puntos }
Fuente: SELECT DISTINCT ID_JUNTA, COUNT(*) FROM dw.Puntos_Mapa WHERE ID_JUNTA IS NOT NULL GROUP BY ...
```

---

### 5. Series temporales SHAC: altura limnimétrica y nivel freático

**Archivo**: `api/routers/series_temporales.py`  
**Problema**: Para cuenca, subcuenca y subsubcuenca existen endpoints de caudal, altura limnimétrica y nivel freático. Para SHAC solo existe el de caudal.

**Solución propuesta**:

#### [NEW] Endpoint `/cuencas/shac/series_de_tiempo/altura_linimetrica`
Misma estructura que el endpoint SHAC de caudal pero consultando `ALTURA_LIMNIMETRICA`.

#### [NEW] Endpoint `/cuencas/shac/series_de_tiempo/nivel_freatico`
Misma estructura pero consultando `NIVEL_FREATICO`.

---

### 6. Documentación de `dw.Mediciones_full`

**Archivo**: `CLAUDE.md`  
**Problema**: El plan y el código referencian `dw.Mediciones_full` como fuente de datos principal, pero no está documentada en ninguna parte. `CLAUDE.md` solo lista `DIM_Geografia`, `DIM_Cuenca` y `FACT_Mediciones_Caudal`.

**Solución**: Actualizar la sección "Key Database Tables" de `CLAUDE.md`:
```markdown
### Key Database Tables
- `dw.Mediciones_full`: Vista/tabla desnormalizada con todos los datos de mediciones, 
  geografía, cuencas, SHAC, APR, Junta y canal de transmisión. Fuente primaria.
- `dw.Puntos_Mapa`: Tabla pre-agregada por punto (UTM_Norte, UTM_Este) con estadísticas 
  de caudal y columnas de filtro. Generada desde Mediciones_full.
- `dw.Series_Tiempo`: Tabla pre-agregada por punto y fecha con promedios de caudal, 
  altura limnimétrica, nivel freático y totalizador. Generada desde Mediciones_full.
- `dw.Cuencas_Regiones`: Catálogo de cuencas/subcuencas/subsubcuencas por región.
- `dw.Cuenca_Stats`: Estadísticas pre-agregadas por cuenca.
- `dw.Filtros_Reactivos_Stats`: Estadísticas para filtros reactivos del frontend.
```

---

## Verification Plan

### Automated Tests
- Validar schemas Pydantic consumiendo el Swagger (`/docs`) y verificar que los tipos (Ej: enteros para ids, booleanos para APR) se procesen sin caídas 500.
- Ejecutar `uv run pytest` para confirmar que los tests existentes no se rompen con los cambios.

### Manual Verification
- Ejecutar el servidor con `uv run uvicorn main:app --reload`
- **Bug `cod_subsubcuenca`**: Llamar a `/puntos?cod_subsubcuenca=XXXX` y verificar que el campo aparezca en la respuesta JSON (actualmente es `null`).
- **Filtros aplicados**: Llamar a `/puntos/count?shac=121&apr=true` y verificar que `filtros_aplicados` incluya esos parámetros.
- **Endpoints nuevos**: Verificar `/shacs` y `/juntas` retornen datos correctos.
- Consumir el endpoint de series de tiempo SHAC de caudal y observar `caudal_sumado` y `totalizador_max`.
- Verificar rendimiento de queries con nuevos filtros SHAC/Junta (evaluar si los índices son necesarios según volumen de datos).
