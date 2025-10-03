# Changelog

All notable changes to the Aguas Transparentes API will be documented in this file.

## [1.2.0] - 2025-01-03

### 🚨 Breaking Changes

#### GET /puntos
- **REMOVED** fields from response:
  - `huso`
  - `region`
  - `provincia`
  - `comuna`
- **Response now only includes:**
  - `utm_norte`
  - `utm_este`
  - `es_pozo_subterraneo`

**Migration:** If you need geographic divisions (region, provincia, comuna), use the `/atlas` endpoint instead.

#### GET /puntos/info
- **REMOVED** fields:
  - `caudal_minimo`
  - `caudal_maximo`
- **ADDED** fields:
  - `utm_norte`
  - `utm_este`
  - `es_pozo_subterraneo`
  - `cod_cuenca`
  - `cod_subcuenca`
- **FIXED:** Now correctly queries cuenca based on UTM coordinates instead of returning random data

**Migration:** Use the `cod_cuenca` and `cod_subcuenca` fields to identify the watershed. Min/max values are available in `/puntos/estadisticas`.

### ✨ New Features

#### POST /puntos/estadisticas
- **ADDED** statistics for `altura_limnimetrica` (water level)
- **ADDED** statistics for `nivel_freatico` (water table level)
- **ADDED** date range fields for all metrics:
  - `primera_fecha` - first measurement date
  - `ultima_fecha` - last measurement date
- **Response structure changed** to nested format with separate objects for each metric type

**New Response Format:**
```json
[{
  "utm_norte": 123456,
  "utm_este": 789012,
  "caudal": {
    "total_registros": 1250,
    "promedio": 15.5,
    "minimo": 2.3,
    "maximo": 45.8,
    "desviacion_estandar": 8.2,
    "primera_fecha": "2020-01-15",
    "ultima_fecha": "2024-12-30"
  },
  "altura_limnimetrica": { ... },
  "nivel_freatico": { ... }
}]
```

#### GET /cuencas
- **ADDED** `cod_region` field to identify the region where the watershed is located

#### GET /cuencas/stats
- **ADDED** `cod_region` field
- **ADDED** global statistics (across entire database):
  - `global_promedio` - average flow across all measurements
  - `global_minimo` - minimum flow across all measurements
  - `global_maximo` - maximum flow across all measurements

**New Response Format:**
```json
{
  "estadisticas": [{
    "cod_cuenca": 101,
    "nom_cuenca": "Río Elqui",
    "cod_region": 4,
    "cod_subcuenca": 10101,
    "nom_subcuenca": "Río Turbio",
    "caudal_promedio": 15.5,
    "caudal_minimo": 2.3,
    "caudal_maximo": 45.8,
    "total_puntos_unicos": 12,
    "total_mediciones": 5800,
    "global_promedio": 22.3,
    "global_minimo": 0.1,
    "global_maximo": 150.5
  }]
}
```

### 🔧 Improvements
- Improved query performance for `/puntos/info` by using proper UTM-based joins
- Better data accuracy by removing cross joins and using proper foreign key relationships

---

## [1.1.0] - 2025-01-02

### ✨ New Features
- Added time series endpoints for cuenca, subcuenca, and subsubcuenca:
  - `/cuencas/cuenca/series_de_tiempo/caudal`
  - `/cuencas/cuenca/series_de_tiempo/altura_linimetrica`
  - `/cuencas/cuenca/series_de_tiempo/nivel_freatico`
  - `/cuencas/subcuenca/series_de_tiempo/*`
  - `/cuencas/subsubcuenca/series_de_tiempo/*`
- All time series endpoints return complete datasets without limits

### 🔄 Changed
- Renamed endpoint paths:
  - `/cuencas/series_de_tiempo/*` → `/cuencas/cuenca/*`
  - `/cuencas/subcuencas/*` → `/cuencas/subcuenca/*`
  - `/cuencas/subsubcuencas/*` → `/cuencas/subsubcuenca/*`

---

## [1.0.0] - 2025-01-01

Initial API release with basic endpoints for water resource data queries.
