# Plan de Implementación Frontend — Nuevas Funcionalidades Hidrológicas

> **Repositorio frontend**: `/home/javier/Github/frontend-aguas`  
> **Stack**: Astro + React + Leaflet + Tailwind CSS + Material-UI + Recharts  
> **Documento complementario**: [Plan backend](./plan_implementacion_filtros_caudales.md)

Este plan detalla los cambios necesarios en el frontend para consumir las nuevas funcionalidades del backend: filtros SHAC/APR/Junta, campos de subsubcuenca, datos de canal de transmisión, y métricas de caudal sumado y totalizador en gráficos.

---

## Resumen de Cambios

| Área | Archivos afectados | Complejidad |
|---|---|---|
| Nuevos filtros en sidebar (SHAC, APR, Junta) | 5 archivos | 🟡 Media |
| Nuevos campos en popup/info de punto | 2 archivos | 🟢 Baja |
| Métricas caudal_sumado/totalizador en gráficos | 3 archivos | 🟡 Media |
| API Service — nuevos endpoints y parámetros | 2 archivos | 🟢 Baja |
| Endpoint constants | 1 archivo | 🟢 Baja |

---

## Cambios Propuestos

### 1. Constantes de API — Nuevos endpoints

#### [MODIFY] `src/constants/apiEndpoints.js`

Agregar los nuevos endpoints que provee el backend y los nuevos filtros por defecto:

```javascript
// Dentro de API_ENDPOINTS, agregar:

// Endpoints de SHAC
SHACS: '/api/shacs',                           // Listado de SHACs (nuevo backend)
SHAC_SERIES_TIEMPO_CAUDAL: '/api/cuencas/shac/series_de_tiempo/caudal',

// Endpoints de juntas
JUNTAS: '/api/juntas',                         // Listado de Juntas (nuevo backend)

// Endpoints de subsubcuencas
SUBSUBCUENCAS_SERIES_TIEMPO_CAUDAL: '/api/cuencas/subsubcuenca/series_de_tiempo/caudal',
SUBSUBCUENCAS_SERIES_TIEMPO_ALTURA_LINIMETRICA: '/api/cuencas/subsubcuenca/series_de_tiempo/altura_linimetrica',
SUBSUBCUENCAS_SERIES_TIEMPO_NIVEL_FREATICO: '/api/cuencas/subsubcuenca/series_de_tiempo/nivel_freatico',
```

Agregar nuevos defaults al `FILTER_CONFIG`:
```javascript
DEFAULT_FILTERS: {
  region: '',
  cuenca: '',
  subcuenca: '',
  shac: '',        // NUEVO
  apr: '',         // NUEVO
  id_junta: '',    // NUEVO
  limit: 10,
  pozo: ''
},
```

---

### 2. API Service — Nuevos métodos

#### [MODIFY] `src/services/apiService.js`

Agregar métodos para consumir los nuevos endpoints:

```javascript
// Listado de SHACs para dropdown
async getShacs() {
  return this.request(API_ENDPOINTS.SHACS);
}

// Listado de Juntas para dropdown
async getJuntas() {
  return this.request(API_ENDPOINTS.JUNTAS);
}

// Series de tiempo SHAC
async getShacSeriesTiempoCaudal(shacIdentificador, pozo = null) {
  const pozoParam = pozo !== null ? `&pozo=${pozo}` : '';
  return this.request(
    `${API_ENDPOINTS.SHAC_SERIES_TIEMPO_CAUDAL}?shac_identificador=${shacIdentificador}${pozoParam}`
  );
}
```

Además, asegurar que `getPuntos` envíe los nuevos query params (`shac`, `apr`, `id_junta`). Actualmente ya lo hace de forma genérica al recibir un `URLSearchParams`, pero hay que verificar que `buildQueryParams` los incluya.

---

### 3. Lógica de filtros — Enviar nuevos parámetros a la API

#### [MODIFY] `src/utils/filterUtils.js`

En la función `buildQueryParams`, agregar la lógica para los nuevos filtros:

```javascript
// Después del bloque de filtro de código de obra:

// Filtro SHAC
if (filtros.shac) {
  queryParams.append("shac", String(filtros.shac));
}

// Filtro APR
if (filtros.apr !== undefined && filtros.apr !== "") {
  queryParams.append("apr", String(filtros.apr));
}

// Filtro Junta
if (filtros.id_junta) {
  queryParams.append("id_junta", String(filtros.id_junta));
}
```

#### [MODIFY] `src/hooks/useFilterLogic.js`

Agregar los nuevos filtros a la lista de dependencias que limpian puntos al cambiar:

```javascript
// Línea ~68: agregar filtros.shac, filtros.apr, filtros.id_junta al useEffect
useEffect(() => {
  setPuntos([]);
  setQueryCompleted(false);
}, [filtros.region, filtros.cuenca, filtros.subcuenca, filtros.tipoPunto,
    filtros.shac, filtros.apr, filtros.id_junta,   // NUEVO
    filtros.fechaInicio, filtros.fechaFin, filtros.fechaPredefinida]);
```

---

### 4. Nuevos componentes de filtro en la sidebar

#### [MODIFY] `src/components/sidebars/FilterSection.jsx`

Agregar 3 nuevos componentes de filtro:

```jsx
// Filtro SHAC — dropdown que se podrá alimentar desde /api/shacs
export const ShacFilter = ({ filtros, handleFiltroChange, shacsDisponibles = [] }) => (
  <SelectFilter
    label="Sector SHAC:"
    name="shac"
    value={filtros.shac}
    onChange={handleFiltroChange}
    options={shacsDisponibles}
    placeholder="-- Todos --"
  />
);

// Filtro APR — selector booleano
export const AprFilter = ({ filtros, handleFiltroChange }) => (
  <SelectFilter
    label="Agua Potable Rural:"
    name="apr"
    value={filtros.apr ?? ""}
    onChange={handleFiltroChange}
    options={[
      { value: "true", label: "Sí (APR)" },
      { value: "false", label: "No" }
    ]}
    placeholder="-- Todos --"
  />
);

// Filtro Junta — dropdown alimentado desde /api/juntas
export const JuntaFilter = ({ filtros, handleFiltroChange, juntasDisponibles = [] }) => (
  <SelectFilter
    label="Junta de Vigilancia:"
    name="id_junta"
    value={filtros.id_junta}
    onChange={handleFiltroChange}
    options={juntasDisponibles}
    placeholder="-- Todas --"
  />
);
```

#### [MODIFY] `src/components/sidebars/SidebarFiltros.jsx`

Importar y renderizar los nuevos filtros en el sidebar, entre `SubcuencaFilter` y `LimitFilter`:

```jsx
import {
  // ... imports existentes
  ShacFilter,
  AprFilter,
  JuntaFilter
} from './FilterSection.jsx';

// Dentro del JSX, después de <SubcuencaFilter /> y antes de <LimitFilter />:

<ShacFilter
  filtros={filtros}
  handleFiltroChange={handleFiltroChange}
  shacsDisponibles={shacsDisponibles}
/>

<AprFilter
  filtros={filtros}
  handleFiltroChange={handleFiltroChange}
/>

<JuntaFilter
  filtros={filtros}
  handleFiltroChange={handleFiltroChange}
  juntasDisponibles={juntasDisponibles}
/>
```

> **Nota**: `shacsDisponibles` y `juntasDisponibles` deben cargarse al inicio de la app desde los endpoints `/api/shacs` y `/api/juntas`. Esto puede hacerse en `useMapData.js` junto a la carga de cuencas.

---

### 5. Datos iniciales — Cargar SHACs y Juntas al inicio

#### [MODIFY] `src/hooks/useMapData.js`

Agregar la carga de SHACs y Juntas al hook de datos iniciales:

```javascript
const [shacsDisponibles, setShacsDisponibles] = useState([]);
const [juntasDisponibles, setJuntasDisponibles] = useState([]);

// En el useEffect de carga inicial, agregar:
apiService.getShacs()
  .then(data => {
    const opciones = (data || []).map(s => ({
      value: s.cod_sector_sha,
      label: s.sector_sha || `SHAC ${s.cod_sector_sha}`
    }));
    setShacsDisponibles(opciones);
  })
  .catch(err => console.error("Error cargando SHACs:", err));

apiService.getJuntas()
  .then(data => {
    const opciones = (data || []).map(j => ({
      value: j.id_junta,
      label: `Junta ${j.id_junta}`
    }));
    setJuntasDisponibles(opciones);
  })
  .catch(err => console.error("Error cargando Juntas:", err));
```

Los valores deben exponerse a través del `MapContext` para que lleguen al sidebar de filtros.

---

### 6. Popup/Info de punto — Mostrar nuevos campos

#### [MODIFY] `src/components/sidebars/SidebarPunto.jsx`

Cuando se muestra la info del punto, agregar los nuevos campos que ya retorna `/puntos/info`:

```jsx
{/* Después de la sección de código de obra */}

{punto.sector_sha && (
  <p className="text-sm text-gray-600">
    <strong>Sector SHAC:</strong> {punto.sector_sha}
  </p>
)}

{punto.apr !== null && punto.apr !== undefined && (
  <p className="text-sm text-gray-600">
    <strong>APR:</strong> {punto.apr ? 'Sí' : 'No'}
  </p>
)}

{punto.id_junta && (
  <p className="text-sm text-gray-600">
    <strong>Junta de Vigilancia:</strong> ID {punto.id_junta}
    {punto.parte_junta !== null && (
      <span> | Participa: {punto.parte_junta ? 'Sí' : 'No'}</span>
    )}
    {punto.representa_junta !== null && (
      <span> | Representa: {punto.representa_junta ? 'Sí' : 'No'}</span>
    )}
  </p>
)}

{punto.canal_transmision !== null && punto.canal_transmision !== undefined && (
  <p className="text-sm text-gray-600">
    <strong>Canal de Transmisión:</strong> {punto.canal_transmision}
  </p>
)}

{punto.nombre_subsubcuenca && (
  <p className="text-sm text-gray-600">
    <strong>Subsubcuenca:</strong> {punto.nombre_subsubcuenca}
  </p>
)}
```

---

### 7. Gráficos — Métricas de caudal sumado y totalizador

#### [MODIFY] `src/components/charts/TimeSeriesChartPair.jsx`

El componente actualmente muestra líneas de `avg`, `min` y `max`. Para las métricas de caudal sumado y totalizador, hay dos opciones:

**Opción A — Agregar toggles de líneas adicionales** (recomendada):
Agregar checkboxes para `caudal_sumado`, `totalizador_sumado` y `totalizador_max` en los controles del gráfico. Estas líneas se dibujarían con colores diferenciados.

**Opción B — Mostrar en un gráfico separado**:
Crear un segundo chart debajo del principal que muestre las métricas de sumatorio/totalizador.

La opción A requiere modificar `processSeriesTiempoData` en `useAnalysisData.js` para que propague los campos `caudal_sumado`, `totalizador_sumado` y `totalizador_max` de la respuesta de la API.

#### [MODIFY] `src/hooks/useAnalysisData.js`

En `processSeriesTiempoData`, actualmente se procesa solo un `valueKey`. Para soportar métricas adicionales, se necesita propagar los campos extra:

```javascript
// Al iterar seriesData, propagar campos adicionales:
const valor_sumado = Number(item['caudal_sumado']) || 0;
const valor_totalizador = Number(item['totalizador_max']) || 0;

// En el agrupado mensual/diario, agregar:
mensualMap[mesClave].valores_sumados.push(valor_sumado);
mensualMap[mesClave].valores_totalizador.push(valor_totalizador);
```

---

## Dependencias del Backend

Estos cambios del frontend **dependen** de que el backend tenga implementados:

| Endpoint backend requerido | Estado |
|---|---|
| `GET /api/shacs` — Listado de SHACs | ❌ Pendiente |
| `GET /api/juntas` — Listado de Juntas | ❌ Pendiente |
| Filtros `shac`, `apr`, `id_junta` en `/puntos` y `/puntos/count` | ✅ Implementado |
| Campos SHAC/APR/Junta/Canal en `/puntos/info` | ✅ Implementado |
| `caudal_sumado` y totalizador en series de tiempo | ✅ Implementado |
| `cod_subsubcuenca` en respuesta de `/puntos` | ❌ Bug pendiente |

---

## Orden de Implementación Recomendado

1. **Backend primero**: Implementar endpoints `/shacs`, `/juntas` y corregir bug de `cod_subsubcuenca`
2. **Frontend: API Service y Constants** — Agregar endpoints y métodos nuevos
3. **Frontend: Filtros** — `filterUtils.js` + `FilterSection.jsx` + `SidebarFiltros.jsx`
4. **Frontend: Datos iniciales** — `useMapData.js` para cargar listas de SHAC/Junta
5. **Frontend: Info de punto** — `SidebarPunto.jsx` con nuevos campos
6. **Frontend: Gráficos** — `useAnalysisData.js` + `TimeSeriesChartPair.jsx` para métricas nuevas

---

## Verification Plan

### Manual Verification

- Ejecutar el frontend con `npm run dev` y el backend con `uv run uvicorn main:app --reload`
- Verificar que los dropdowns de SHAC y Junta se pueblen correctamente
- Usar los filtros SHAC/APR/Junta y verificar que los puntos se filtren en el mapa
- Abrir un punto y verificar que aparezcan los nuevos campos (SHAC, APR, Junta, Canal, Subsubcuenca)
- Cargar gráficos de cuenca y verificar que las líneas de caudal sumado/totalizador se muestren
- Verificar en mobile (responsive) que los nuevos filtros no rompan el layout del sidebar
