import os
import json
import logging
import pyodbc
import math
import asyncio
import time
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from datetime import date
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
from asyncio_pool import AioPool
import threading
from queue import Queue, Empty

load_dotenv()

# Global connection pool and cache
connection_pool: Optional[Queue] = None
memory_cache: Dict[str, Dict] = {}
cache_timestamps: Dict[str, float] = {}
CACHE_TTL = 300  # 5 minutes
POOL_SIZE = 10
pool_lock = threading.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global connection_pool
    connection_pool = Queue(maxsize=POOL_SIZE)

    # Pre-populate connection pool
    for _ in range(POOL_SIZE):
        try:
            conn = create_db_connection()
            connection_pool.put(conn)
        except Exception as e:
            logging.error(f"Failed to create initial connection: {e}")

    logging.info(f"Connection pool initialized with {connection_pool.qsize()} connections")
    yield

    # Shutdown
    while not connection_pool.empty():
        try:
            conn = connection_pool.get_nowait()
            conn.close()
        except Empty:
            break
        except Exception as e:
            logging.error(f"Error closing connection: {e}")

app = FastAPI(
    title="Aguas Transparentes API",
    description="Backend API for water resource data from Azure Synapse Analytics",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    root_path="/api"  # This tells FastAPI it's behind /api prefix
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400
)

# Pydantic Models for Request/Response validation
class UTMLocation(BaseModel):
    utm_norte: int = Field(..., ge=0, le=10000000, description="UTM Norte coordinate")
    utm_este: int = Field(..., ge=0, le=1000000, description="UTM Este coordinate")

class PuntoData(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None
    utm_norte: int
    utm_este: int
    huso: int
    region: int
    provincia: int
    comuna: int
    nombre_cuenca: str
    nombre_subcuenca: Optional[str]
    cod_cuenca: int
    cod_subcuenca: Optional[int]
    caudal_promedio: Optional[float]
    n_mediciones: int
    tipoPunto: Dict[str, Any]

class CuencaStats(BaseModel):
    nom_cuenca: str
    cod_cuenca: int
    nom_subcuenca: Optional[str]
    cod_subcuenca: Optional[int]
    cod_region: int

class TimeSeriesPoint(BaseModel):
    fecha_medicion: str
    caudal: Optional[float]

class CaudalAnalysis(BaseModel):
    cuenca_identificador: str
    total_registros_con_caudal: int
    caudal_promedio: Optional[float]
    caudal_minimo: Optional[float]
    caudal_maximo: Optional[float]
    desviacion_estandar_caudal: Optional[float]

# Utility functions for data processing
def build_full_name(nomb_inf: Optional[str], a_pat_inf: Optional[str], a_mat_inf: Optional[str]) -> str:
    """Build full name efficiently"""
    parts = [part.strip() for part in [nomb_inf, a_pat_inf, a_mat_inf] if part and part.strip()]
    return " ".join(parts) if parts else "Desconocido"

def safe_round(value: Optional[float], decimals: int = 2) -> Optional[float]:
    """Safely round a value, handling None"""
    return round(value, decimals) if value is not None else None

def create_db_connection():
    """Create a new database connection"""
    server = os.getenv('SYNAPSE_SERVER')
    database = os.getenv('SYNAPSE_DATABASE')
    username = os.getenv('SYNAPSE_USERNAME')
    password = os.getenv('SYNAPSE_PASSWORD')

    connection_string = f"""
    DRIVER={{ODBC Driver 18 for SQL Server}};
    SERVER={server};
    DATABASE={database};
    UID={username};
    PWD={password};
    Encrypt=yes;
    TrustServerCertificate=no;
    Connection Timeout=30;
    """

    return pyodbc.connect(connection_string)

def get_db_connection():
    """Get connection from pool"""
    global connection_pool
    if connection_pool is None:
        return create_db_connection()

    try:
        # Try to get from pool with timeout
        conn = connection_pool.get(timeout=5.0)

        # Test connection is still alive
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return conn
        except:
            # Connection is dead, create new one
            conn.close()
            return create_db_connection()

    except Empty:
        # Pool is empty, create new connection
        return create_db_connection()

def return_db_connection(conn):
    """Return connection to pool"""
    global connection_pool
    if connection_pool is None:
        conn.close()
        return

    try:
        if not connection_pool.full():
            connection_pool.put_nowait(conn)
        else:
            conn.close()
    except:
        conn.close()

def get_cache_key(query: str, params: List = None) -> str:
    """Generate cache key from query and parameters"""
    key = query
    if params:
        key += str(params)
    return str(hash(key))

def is_cache_valid(cache_key: str) -> bool:
    """Check if cache entry is still valid"""
    if cache_key not in cache_timestamps:
        return False
    return time.time() - cache_timestamps[cache_key] < CACHE_TTL

def execute_query(query: str, params: List = None, use_cache: bool = True) -> List[Dict]:
    """Execute query with connection pooling and caching"""
    # Check cache first
    if use_cache:
        cache_key = get_cache_key(query, params)
        if cache_key in memory_cache and is_cache_valid(cache_key):
            logging.info(f"Cache hit for query: {query[:50]}...")
            return memory_cache[cache_key]

    # Execute query
    conn = get_db_connection()
    try:
        start_time = time.time()
        cursor = conn.cursor()

        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        columns = [column[0] for column in cursor.description]
        results = cursor.fetchall()
        result_list = [dict(zip(columns, row)) for row in results]

        execution_time = time.time() - start_time
        logging.info(f"Query executed in {execution_time:.3f}s, returned {len(result_list)} rows")

        # Cache the result
        if use_cache and len(result_list) > 0:
            memory_cache[cache_key] = result_list
            cache_timestamps[cache_key] = time.time()
            logging.info(f"Cached {len(result_list)} rows for query")

        return result_list

    finally:
        return_db_connection(conn)

def utm_to_latlon(utm_este: float, utm_norte: float, huso: int = 19) -> tuple:
    """Convertir coordenadas UTM a lat/lon"""
    try:
        central_meridian = -69.0  # Para zona 19

        lat = (utm_norte - 10000000) / 111320
        lon = central_meridian + (utm_este - 500000) / (111320 * math.cos(math.radians(lat)))

        return lon, lat
    except:
        return None, None

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint with database connectivity test"""
    try:
        results = execute_query("SELECT 1 as test")
        return {
            "status": "healthy",
            "message": "Water Data API is running",
            "database": "connected"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "status": "unhealthy",
            "message": "Water Data API is running but database connection failed",
            "database": "disconnected",
            "error": str(e)
        })

@app.get("/test-db", tags=["System"])
async def test_database_connection():
    """Test database connection with record count"""
    try:
        results = execute_query("SELECT COUNT(*) as total FROM dw.FACT_Mediciones_Caudal")
        return {
            "status": "success",
            "message": "Database connection successful",
            "total_records": results[0]['total']
        }
    except Exception as e:
        logging.error(f"Database connection failed: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "status": "error",
            "message": f"Database connection failed: {str(e)}"
        })

@app.get("/count", tags=["System"])
async def get_obras_count():
    """Obtiene el número total de registros en la tabla de mediciones"""
    try:
        results = execute_query("SELECT COUNT(*) as total FROM dw.FACT_Mediciones_Caudal")
        return {"total_records": results[0]['total']}
    except Exception as e:
        logging.error(f"Error in get_obras_count: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/puntos/count", tags=["puntos"])
async def get_puntos_count(region: Optional[int] = Query(None)):
    """Obtiene el número de puntos únicos desde Puntos_Mapa"""
    try:
        logging.info(f"Contando puntos con region: {region}")

        count_query = """
        SELECT COUNT(*) as total_puntos_unicos
        FROM dw.Puntos_Mapa
        WHERE 1=1
        """

        query_params = []
        if region:
            count_query += " AND Region = ?"
            query_params.append(region)

        logging.info(f"Ejecutando query count: {count_query}")
        results = execute_query(count_query, query_params)

        total_puntos = results[0]['total_puntos_unicos'] if results else 0

        response = {
            "total_puntos_unicos": total_puntos,
            "filtros_aplicados": {
                "region": region
            }
        }

        logging.info(f"Total puntos únicos encontrados: {total_puntos}")
        return response

    except Exception as e:
        logging.error(f"Error in get_puntos_count: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/cache/stats", tags=["cache"])
async def get_cache_stats():
    """Get cache statistics"""
    return {
        "cached_queries": len(memory_cache),
        "cache_keys": list(memory_cache.keys()),
        "cache_sizes": {k: len(v) for k, v in memory_cache.items()},
        "pool_connections": connection_pool.qsize() if connection_pool else 0
    }

@app.post("/cache/clear", tags=["cache"])
async def clear_cache():
    """Clear all cached data"""
    global memory_cache, cache_timestamps
    memory_cache.clear()
    cache_timestamps.clear()
    return {"message": "Cache cleared successfully"}

@app.get("/performance/warm-up", tags=["performance"])
async def warm_up_cache():
    """Pre-warm frequently accessed data"""
    try:
        logging.info("Starting cache warm-up...")

        # Warm up common queries
        queries = [
            ("SELECT COUNT(*) as total FROM dw.FACT_Mediciones_Caudal", None),
            ("SELECT COUNT(DISTINCT CONCAT(UTM_Norte, '-', UTM_Este)) as total_puntos_unicos FROM dw.DIM_Geografia g WHERE g.UTM_Norte IS NOT NULL AND g.UTM_Este IS NOT NULL", None),
            ("SELECT DISTINCT Region FROM dw.DIM_Geografia WHERE Region IS NOT NULL ORDER BY Region", None),
        ]

        warmed_queries = 0
        for query, params in queries:
            try:
                execute_query(query, params, use_cache=True)
                warmed_queries += 1
            except Exception as e:
                logging.error(f"Failed to warm query: {e}")

        return {
            "message": f"Cache warm-up completed. Warmed {warmed_queries} queries.",
            "cached_queries": len(memory_cache)
        }
    except Exception as e:
        logging.error(f"Cache warm-up failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/puntos", tags=["puntos"])
async def get_puntos(
    region: Optional[int] = Query(None)
):
    """Obtiene puntos desde la tabla pre-agregada Puntos_Mapa"""
    try:
        logging.info(f"Parametros recibidos en /puntos: region={region}")

        query_params = []
        region_filter = ""

        if region:
            region_filter = " AND Region = ?"
            query_params.append(region)

        # Query the pre-aggregated table
        puntos_query = f"""
        SELECT
            UTM_Norte,
            UTM_Este,
            Huso,
            Region,
            Provincia,
            Comuna,
            es_pozo_subterraneo
        FROM dw.Puntos_Mapa
        WHERE UTM_Norte IS NOT NULL
          AND UTM_Este IS NOT NULL
          {region_filter}
        """

        logging.info(f"Ejecutando query desde Puntos_Mapa: {puntos_query}")
        puntos = execute_query(puntos_query, query_params)

        logging.info(f"Se obtuvieron {len(puntos)} puntos desde Puntos_Mapa")

        # Build response
        puntos_out = []

        for punto in puntos:
            puntos_out.append({
                "utm_norte": punto["UTM_Norte"],
                "utm_este": punto["UTM_Este"],
                "huso": punto.get("Huso", 19),
                "region": punto.get("Region"),
                "provincia": punto.get("Provincia"),
                "comuna": punto.get("Comuna"),
                "es_pozo_subterraneo": bool(punto.get("es_pozo_subterraneo", 0))
            })

        logging.info(f"Retornando {len(puntos_out)} puntos")
        return puntos_out

    except Exception as e:
        logging.error(f"Error en get_puntos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/puntos/info", tags=["puntos"])
async def get_punto_info(
    utm_norte: int = Query(..., description="Coordenada UTM Norte del punto"),
    utm_este: int = Query(..., description="Coordenada UTM Este del punto")
):
    """Obtiene información detallada de un punto específico incluyendo cuenca y caudal"""
    try:
        logging.info(f"Obteniendo info detallada para punto: UTM_Norte={utm_norte}, UTM_Este={utm_este}")

        # Get basic geographic info for the point from Puntos_Mapa
        punto_query = """
        SELECT
            UTM_Norte,
            UTM_Este,
            Huso,
            Region,
            Provincia,
            Comuna,
            es_pozo_subterraneo
        FROM dw.Puntos_Mapa
        WHERE UTM_Norte = ?
          AND UTM_Este = ?
        """

        punto_result = execute_query(punto_query, [utm_norte, utm_este])

        if not punto_result:
            raise HTTPException(status_code=404, detail="Punto no encontrado")

        punto = punto_result[0]

        # Get cuenca info (simplified - get one cuenca)
        cuenca_query = """
        SELECT TOP 1
            Cod_Cuenca,
            Nom_Cuenca,
            Cod_Subcuenca,
            Nom_Subcuenca
        FROM dw.DIM_Cuenca
        WHERE Nom_Cuenca IS NOT NULL
        ORDER BY Cod_Cuenca
        """
        cuenca_result = execute_query(cuenca_query)
        cuenca = cuenca_result[0] if cuenca_result else {}

        # Get caudal statistics for this specific point
        caudal_query = """
        SELECT
            AVG(CAST(Caudal AS FLOAT)) as caudal_promedio,
            MIN(CAST(Caudal AS FLOAT)) as caudal_minimo,
            MAX(CAST(Caudal AS FLOAT)) as caudal_maximo,
            COUNT(*) as n_mediciones
        FROM dw.FACT_Mediciones_Caudal
        WHERE UTM_Norte = ?
          AND UTM_Este = ?
          AND Caudal IS NOT NULL
        """
        caudal_result = execute_query(caudal_query, [utm_norte, utm_este])
        caudal_stats = caudal_result[0] if caudal_result else {}

        # Build detailed response
        response = {
            "nombre_cuenca": cuenca.get('Nom_Cuenca'),
            "nombre_subcuenca": cuenca.get('Nom_Subcuenca'),
            "caudal_promedio": safe_round(caudal_stats.get('caudal_promedio')),
            "caudal_minimo": safe_round(caudal_stats.get('caudal_minimo')),
            "caudal_maximo": safe_round(caudal_stats.get('caudal_maximo')),
            "n_mediciones": caudal_stats.get('n_mediciones', 0)
        }

        logging.info(f"Info detallada obtenida para punto {utm_norte}/{utm_este}")
        return response

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error en get_punto_info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/cuencas", tags=["cuencas"])
async def get_unique_cuencas():
    """Obtiene cuencas, subcuencas y subsubcuencas únicas"""
    try:
        # Query cuencas directly - get unique combinations
        cuencas_query = """
        SELECT DISTINCT
            Cod_Cuenca as cod_cuenca,
            Nom_Cuenca as nom_cuenca,
            Cod_Subcuenca as cod_subcuenca,
            Nom_Subcuenca as nom_subcuenca,
            Cod_Subsubcuenca as cod_subsubcuenca,
            Nom_Subsubcuenca as nom_subsubcuenca
        FROM dw.DIM_Cuenca
        WHERE Nom_Cuenca IS NOT NULL
        ORDER BY Cod_Cuenca, Cod_Subcuenca, Cod_Subsubcuenca
        """

        results = execute_query(cuencas_query)

        return {
            "cuencas": [
                {
                    "cod_cuenca": r.get('cod_cuenca'),
                    "nom_cuenca": r.get('nom_cuenca'),
                    "cod_subcuenca": r.get('cod_subcuenca'),
                    "nom_subcuenca": r.get('nom_subcuenca'),
                    "cod_subsubcuenca": r.get('cod_subsubcuenca'),
                    "nom_subsubcuenca": r.get('nom_subsubcuenca')
                } for r in results
            ]
        }
    except Exception as e:
        logging.error(f"Error in get_unique_cuencas: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/atlas", tags=["atlas"])
async def get_atlas():
    """Obtiene regiones, provincias y comunas únicas"""
    try:
        # Query unique geographic divisions from Puntos_Mapa
        atlas_query = """
        SELECT DISTINCT
            Region as region,
            Provincia as provincia,
            Comuna as comuna
        FROM dw.Puntos_Mapa
        WHERE Region IS NOT NULL
        ORDER BY Region, Provincia, Comuna
        """

        results = execute_query(atlas_query)

        return {
            "divisiones": [
                {
                    "region": r.get('region'),
                    "provincia": r.get('provincia'),
                    "comuna": r.get('comuna')
                } for r in results
            ]
        }
    except Exception as e:
        logging.error(f"Error in get_atlas: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/cuencas/stats", tags=["cuencas"])
async def get_cuencas_stats(
    cod_cuenca: Optional[int] = Query(None, description="Código de cuenca"),
    cod_subcuenca: Optional[int] = Query(None, description="Código de subcuenca"),
    cod_subsubcuenca: Optional[int] = Query(None, description="Código de subsubcuenca")
):
    """Obtiene estadísticas de caudal por cuenca, subcuenca o subsubcuenca desde tabla pre-agregada"""
    try:
        # Build filter conditions
        filters = []
        params = []

        if cod_cuenca is not None:
            filters.append("Cod_Cuenca = ?")
            params.append(cod_cuenca)
        if cod_subcuenca is not None:
            filters.append("Cod_Subcuenca = ?")
            params.append(cod_subcuenca)
        if cod_subsubcuenca is not None:
            filters.append("Cod_Subsubcuenca = ?")
            params.append(cod_subsubcuenca)

        # Build WHERE clause (if no filters, return all)
        where_clause = "WHERE " + " AND ".join(filters) if filters else ""

        # Query pre-aggregated table for instant results
        stats_query = f"""
        SELECT
            Cod_Cuenca,
            Nom_Cuenca,
            Cod_Subcuenca,
            Nom_Subcuenca,
            Cod_Subsubcuenca,
            Nom_Subsubcuenca,
            caudal_promedio,
            caudal_minimo,
            caudal_maximo,
            total_puntos_unicos,
            total_mediciones
        FROM dw.Cuenca_Stats
        {where_clause}
        """

        results = execute_query(stats_query, params)

        if not results:
            return {"estadisticas": []}

        # Return all results (can be multiple cuencas if no filter applied)
        return {
            "estadisticas": [
                {
                    "cod_cuenca": r.get('Cod_Cuenca'),
                    "nom_cuenca": r.get('Nom_Cuenca'),
                    "cod_subcuenca": r.get('Cod_Subcuenca'),
                    "nom_subcuenca": r.get('Nom_Subcuenca'),
                    "cod_subsubcuenca": r.get('Cod_Subsubcuenca'),
                    "nom_subsubcuenca": r.get('Nom_Subsubcuenca'),
                    "caudal_promedio": safe_round(r.get('caudal_promedio')),
                    "caudal_minimo": safe_round(r.get('caudal_minimo')),
                    "caudal_maximo": safe_round(r.get('caudal_maximo')),
                    "total_puntos_unicos": r.get('total_puntos_unicos', 0),
                    "total_mediciones": r.get('total_mediciones', 0)
                } for r in results
            ]
        }

    except Exception as e:
        logging.error(f"Error in get_cuencas_stats: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/cuencas/series_de_tiempo/caudal", tags=["cuencas"])
async def get_caudal_por_tiempo_por_cuenca(
    cuenca_identificador: str = Query(..., description="Código o nombre de la cuenca"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio (YYYY-MM-DD)"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin (YYYY-MM-DD)")
):
    """Obtiene el caudal extraído a lo largo del tiempo para una cuenca específica"""
    try:
        # Determine if identifier is numeric (code) or text (name)
        if cuenca_identificador.isdigit():
            filter_condition = "Cod_Cuenca = ?"
            params = [int(cuenca_identificador)]
        else:
            filter_condition = "Nom_Cuenca = ?"
            params = [cuenca_identificador]

        # Get UTM coordinates for this cuenca
        utm_query = f"""
        SELECT DISTINCT UTM_Norte, UTM_Este
        FROM dw.DIM_Cuenca
        WHERE {filter_condition}
          AND UTM_Norte IS NOT NULL
          AND UTM_Este IS NOT NULL
        """

        utm_results = execute_query(utm_query, params)

        if not utm_results:
            raise HTTPException(status_code=404, detail="No se encontró la cuenca especificada.")

        # Build OR conditions for all UTM coordinates in this cuenca
        utm_conditions = " OR ".join([
            f"(UTM_Norte = {r['UTM_Norte']} AND UTM_Este = {r['UTM_Este']})"
            for r in utm_results
        ])

        # Build date filters
        date_filter = ""
        if fecha_inicio:
            date_filter += f" AND Fecha_Medicion >= '{fecha_inicio}'"
        if fecha_fin:
            date_filter += f" AND Fecha_Medicion <= '{fecha_fin}'"

        # Query time series data (Note: adjust field name if different)
        time_series_query = f"""
        SELECT TOP 1000
            Fecha_Medicion as fecha_medicion,
            Caudal as caudal
        FROM dw.FACT_Mediciones_Caudal
        WHERE ({utm_conditions})
          AND Caudal IS NOT NULL
          {date_filter}
        ORDER BY Fecha_Medicion DESC
        """

        results = execute_query(time_series_query)

        if not results:
            raise HTTPException(status_code=404, detail="No se encontraron datos de caudal para el período o cuenca especificada.")

        caudal_por_tiempo = [
            {
                "fecha_medicion": str(r.get('fecha_medicion')) if r.get('fecha_medicion') else None,
                "caudal": r.get('caudal')
            } for r in results
        ]

        return {
            "cuenca_identificador": cuenca_identificador,
            "total_registros": len(caudal_por_tiempo),
            "caudal_por_tiempo": caudal_por_tiempo
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error in get_caudal_por_tiempo_por_cuenca: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/cuencas/subcuencas/series_de_tiempo/caudal", tags=["cuencas"])
async def get_caudal_por_tiempo_por_subcuenca(
    cuenca_identificador: str = Query(..., description="Código o nombre de la subcuenca"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio (YYYY-MM-DD)"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin (YYYY-MM-DD)")
):
    """Obtiene el caudal extraído a lo largo del tiempo para una subcuenca específica"""
    try:
        # Determine if identifier is numeric (code) or text (name)
        if cuenca_identificador.isdigit():
            filter_condition = "Cod_Subcuenca = ?"
            params = [int(cuenca_identificador)]
        else:
            filter_condition = "Nom_Subcuenca = ?"
            params = [cuenca_identificador]

        # Get UTM coordinates for this subcuenca
        utm_query = f"""
        SELECT DISTINCT UTM_Norte, UTM_Este
        FROM dw.DIM_Cuenca
        WHERE {filter_condition}
          AND UTM_Norte IS NOT NULL
          AND UTM_Este IS NOT NULL
        """

        utm_results = execute_query(utm_query, params)

        if not utm_results:
            raise HTTPException(status_code=404, detail="No se encontró la subcuenca especificada.")

        # Build OR conditions for all UTM coordinates in this subcuenca
        utm_conditions = " OR ".join([
            f"(UTM_Norte = {r['UTM_Norte']} AND UTM_Este = {r['UTM_Este']})"
            for r in utm_results
        ])

        # Build date filters
        date_filter = ""
        if fecha_inicio:
            date_filter += f" AND Fecha_Medicion >= '{fecha_inicio}'"
        if fecha_fin:
            date_filter += f" AND Fecha_Medicion <= '{fecha_fin}'"

        # Query time series data
        time_series_query = f"""
        SELECT TOP 1000
            Fecha_Medicion as fecha_medicion,
            Caudal as caudal
        FROM dw.FACT_Mediciones_Caudal
        WHERE ({utm_conditions})
          AND Caudal IS NOT NULL
          {date_filter}
        ORDER BY Fecha_Medicion DESC
        """

        results = execute_query(time_series_query)

        if not results:
            raise HTTPException(status_code=404, detail="No se encontraron datos de caudal para el período o subcuenca especificada.")

        caudal_por_tiempo = [
            {
                "fecha_medicion": str(r.get('fecha_medicion')) if r.get('fecha_medicion') else None,
                "caudal": r.get('caudal')
            } for r in results
        ]

        return {
            "subcuenca_identificador": cuenca_identificador,
            "total_registros": len(caudal_por_tiempo),
            "caudal_por_tiempo": caudal_por_tiempo
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error in get_caudal_por_tiempo_por_subcuenca: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/cuencas/subsubcuencas/series_de_tiempo/caudal", tags=["cuencas"])
async def get_caudal_por_tiempo_por_subsubcuenca(
    cuenca_identificador: str = Query(..., description="Código o nombre de la subsubcuenca"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio (YYYY-MM-DD)"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin (YYYY-MM-DD)")
):
    """Obtiene el caudal extraído a lo largo del tiempo para una subsubcuenca específica"""
    try:
        # Determine if identifier is numeric (code) or text (name)
        if cuenca_identificador.isdigit():
            filter_condition = "Cod_Subsubcuenca = ?"
            params = [int(cuenca_identificador)]
        else:
            filter_condition = "Nom_Subsubcuenca = ?"
            params = [cuenca_identificador]

        # Get UTM coordinates for this subsubcuenca
        utm_query = f"""
        SELECT DISTINCT UTM_Norte, UTM_Este
        FROM dw.DIM_Cuenca
        WHERE {filter_condition}
          AND UTM_Norte IS NOT NULL
          AND UTM_Este IS NOT NULL
        """

        utm_results = execute_query(utm_query, params)

        if not utm_results:
            raise HTTPException(status_code=404, detail="No se encontró la subsubcuenca especificada.")

        # Build OR conditions for all UTM coordinates in this subsubcuenca
        utm_conditions = " OR ".join([
            f"(UTM_Norte = {r['UTM_Norte']} AND UTM_Este = {r['UTM_Este']})"
            for r in utm_results
        ])

        # Build date filters
        date_filter = ""
        if fecha_inicio:
            date_filter += f" AND Fecha_Medicion >= '{fecha_inicio}'"
        if fecha_fin:
            date_filter += f" AND Fecha_Medicion <= '{fecha_fin}'"

        # Query time series data
        time_series_query = f"""
        SELECT TOP 1000
            Fecha_Medicion as fecha_medicion,
            Caudal as caudal
        FROM dw.FACT_Mediciones_Caudal
        WHERE ({utm_conditions})
          AND Caudal IS NOT NULL
          {date_filter}
        ORDER BY Fecha_Medicion DESC
        """

        results = execute_query(time_series_query)

        if not results:
            raise HTTPException(status_code=404, detail="No se encontraron datos de caudal para el período o subsubcuenca especificada.")

        caudal_por_tiempo = [
            {
                "fecha_medicion": str(r.get('fecha_medicion')) if r.get('fecha_medicion') else None,
                "caudal": r.get('caudal')
            } for r in results
        ]

        return {
            "subsubcuenca_identificador": cuenca_identificador,
            "total_registros": len(caudal_por_tiempo),
            "caudal_por_tiempo": caudal_por_tiempo
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error in get_caudal_por_tiempo_por_subsubcuenca: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.post("/puntos/estadisticas", tags=["puntos"])
async def get_point_statistics(locations: List[UTMLocation]):
    """Obtiene estadísticas de caudal para uno o varios puntos UTM específicos"""
    try:
        if not locations:
            raise HTTPException(status_code=400, detail="Debe proporcionar al menos una coordenada UTM")

        if len(locations) == 1:
            # Single location analysis
            loc = locations[0]

            stats_query = """
            SELECT
                COUNT(*) as count,
                AVG(CAST(Caudal AS FLOAT)) as avg_caudal,
                MIN(CAST(Caudal AS FLOAT)) as min_caudal,
                MAX(CAST(Caudal AS FLOAT)) as max_caudal,
                STDEV(CAST(Caudal AS FLOAT)) as std_caudal
            FROM dw.FACT_Mediciones_Caudal
            WHERE UTM_Norte = ?
            AND UTM_Este = ?
            AND Caudal IS NOT NULL
            """

            results = execute_query(stats_query, [loc.utm_norte, loc.utm_este])
            result = results[0] if results else {}

            if result.get('count', 0) == 0:
                return [{
                    "utm_norte": loc.utm_norte,
                    "utm_este": loc.utm_este,
                    "message": "No se encontraron datos de caudal para las coordenadas UTM especificadas."
                }]
            else:
                return [{
                    "utm_norte": loc.utm_norte,
                    "utm_este": loc.utm_este,
                    "total_registros_con_caudal": result.get('count'),
                    "caudal_promedio": round(result.get('avg_caudal', 0), 2) if result.get('avg_caudal') else None,
                    "caudal_minimo": round(result.get('min_caudal', 0), 2) if result.get('min_caudal') else None,
                    "caudal_maximo": round(result.get('max_caudal', 0), 2) if result.get('max_caudal') else None,
                    "desviacion_estandar_caudal": round(result.get('std_caudal', 0), 2) if result.get('std_caudal') else None
                }]
        else:
            # Multiple locations analysis
            coords_conditions = " OR ".join([
                f"(UTM_Norte = {loc.utm_norte} AND UTM_Este = {loc.utm_este})"
                for loc in locations
            ])

            multi_stats_query = f"""
            SELECT
                COUNT(*) as count,
                AVG(CAST(Caudal AS FLOAT)) as avg_caudal,
                MIN(CAST(Caudal AS FLOAT)) as min_caudal,
                MAX(CAST(Caudal AS FLOAT)) as max_caudal,
                STDEV(CAST(Caudal AS FLOAT)) as std_caudal
            FROM dw.FACT_Mediciones_Caudal
            WHERE ({coords_conditions})
            AND Caudal IS NOT NULL
            """

            results = execute_query(multi_stats_query)
            result = results[0] if results else {}

            return [{
                "puntos_consultados": len(locations),
                "total_registros_con_caudal": result.get('count', 0),
                "caudal_promedio": round(result.get('avg_caudal', 0), 2) if result.get('avg_caudal') else None,
                "caudal_minimo": round(result.get('min_caudal', 0), 2) if result.get('min_caudal') else None,
                "caudal_maximo": round(result.get('max_caudal', 0), 2) if result.get('max_caudal') else None,
                "desviacion_estandar_caudal": round(result.get('std_caudal', 0), 2) if result.get('std_caudal') else None
            }]

    except Exception as e:
        logging.error(f"Error in get_point_statistics: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/puntos/series_de_tiempo/caudal", tags=["puntos"])
async def get_caudal_por_tiempo_por_punto(
    utm_norte: int = Query(..., description="Coordenada UTM Norte del punto"),
    utm_este: int = Query(..., description="Coordenada UTM Este del punto"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio (YYYY-MM-DD)"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin (YYYY-MM-DD)")
):
    """Obtiene el caudal extraído a lo largo del tiempo para un punto UTM específico"""
    try:
        time_series_query = """
        SELECT TOP 1000
            '2023-01-01' as fecha_medicion,  -- Simulated date - replace with actual date field
            Caudal as caudal
        FROM dw.FACT_Mediciones_Caudal
        WHERE UTM_Norte = ?
        AND UTM_Este = ?
        AND Caudal IS NOT NULL
        ORDER BY Caudal DESC
        """

        results = execute_query(time_series_query, [utm_norte, utm_este])

        if not results:
            raise HTTPException(status_code=404, detail="No se encontraron datos de caudal para el punto UTM o período especificado.")

        caudal_por_tiempo = [
            {
                "fecha_medicion": r.get('fecha_medicion'),
                "caudal": r.get('caudal')
            } for r in results
        ]

        return {
            "utm_norte": utm_norte,
            "utm_este": utm_este,
            "caudal_por_tiempo": caudal_por_tiempo
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error in get_caudal_por_tiempo_por_punto: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/puntos/series_de_tiempo/altura_linimetrica", tags=["puntos"])
async def get_altura_linimetrica_por_tiempo_por_punto(
    utm_norte: int = Query(..., description="Coordenada UTM Norte del punto"),
    utm_este: int = Query(..., description="Coordenada UTM Este del punto"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio (YYYY-MM-DD)"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin (YYYY-MM-DD)")
):
    """Obtiene la altura limnimétrica a lo largo del tiempo para un punto UTM específico"""
    try:
        # Build date filters
        date_filter = ""
        if fecha_inicio:
            date_filter += f" AND Fecha_Medicion >= '{fecha_inicio}'"
        if fecha_fin:
            date_filter += f" AND Fecha_Medicion <= '{fecha_fin}'"

        # Get total count
        count_query = f"""
        SELECT COUNT(*) as total
        FROM dw.FACT_Mediciones_Caudal
        WHERE UTM_Norte = ?
          AND UTM_Este = ?
          AND Altura_Limnimetrica IS NOT NULL
          {date_filter}
        """

        count_result = execute_query(count_query, [utm_norte, utm_este])
        total_count = count_result[0]['total'] if count_result else 0

        # Query time series data for altura limnimétrica
        time_series_query = f"""
        SELECT
            Fecha_Medicion as fecha_medicion,
            Altura_Limnimetrica as altura_linimetrica
        FROM dw.FACT_Mediciones_Caudal
        WHERE UTM_Norte = ?
          AND UTM_Este = ?
          AND Altura_Limnimetrica IS NOT NULL
          {date_filter}
        ORDER BY Fecha_Medicion DESC
        """

        results = execute_query(time_series_query, [utm_norte, utm_este])

        if not results:
            raise HTTPException(status_code=404, detail="No se encontraron datos de altura limnimétrica para el punto UTM o período especificado.")

        altura_por_tiempo = [
            {
                "fecha_medicion": str(r.get('fecha_medicion')) if r.get('fecha_medicion') else None,
                "altura_linimetrica": r.get('altura_linimetrica')
            } for r in results
        ]

        return {
            "utm_norte": utm_norte,
            "utm_este": utm_este,
            "total_registros": total_count,
            "registros_retornados": len(altura_por_tiempo),
            "altura_por_tiempo": altura_por_tiempo
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error in get_altura_linimetrica_por_tiempo_por_punto: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/puntos/series_de_tiempo/nivel_freatico", tags=["puntos"])
async def get_nivel_freatico_por_tiempo_por_punto(
    utm_norte: int = Query(..., description="Coordenada UTM Norte del punto"),
    utm_este: int = Query(..., description="Coordenada UTM Este del punto"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio (YYYY-MM-DD)"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin (YYYY-MM-DD)")
):
    """Obtiene el nivel freático a lo largo del tiempo para un punto UTM específico"""
    try:
        # Build date filters
        date_filter = ""
        if fecha_inicio:
            date_filter += f" AND Fecha_Medicion >= '{fecha_inicio}'"
        if fecha_fin:
            date_filter += f" AND Fecha_Medicion <= '{fecha_fin}'"

        # Get total count
        count_query = f"""
        SELECT COUNT(*) as total
        FROM dw.FACT_Mediciones_Caudal
        WHERE UTM_Norte = ?
          AND UTM_Este = ?
          AND Nivel_Freatico IS NOT NULL
          {date_filter}
        """

        count_result = execute_query(count_query, [utm_norte, utm_este])
        total_count = count_result[0]['total'] if count_result else 0

        # Query time series data for nivel freático
        time_series_query = f"""
        SELECT
            Fecha_Medicion as fecha_medicion,
            Nivel_Freatico as nivel_freatico
        FROM dw.FACT_Mediciones_Caudal
        WHERE UTM_Norte = ?
          AND UTM_Este = ?
          AND Nivel_Freatico IS NOT NULL
          {date_filter}
        ORDER BY Fecha_Medicion DESC
        """

        results = execute_query(time_series_query, [utm_norte, utm_este])

        if not results:
            raise HTTPException(status_code=404, detail="No se encontraron datos de nivel freático para el punto UTM o período especificado.")

        nivel_por_tiempo = [
            {
                "fecha_medicion": str(r.get('fecha_medicion')) if r.get('fecha_medicion') else None,
                "nivel_freatico": r.get('nivel_freatico')
            } for r in results
        ]

        return {
            "utm_norte": utm_norte,
            "utm_este": utm_este,
            "total_registros": total_count,
            "registros_retornados": len(nivel_por_tiempo),
            "nivel_por_tiempo": nivel_por_tiempo
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error in get_nivel_freatico_por_tiempo_por_punto: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

