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

# Load .env only in development (Azure provides env vars automatically)
# Azure sets WEBSITE_INSTANCE_ID when running in App Service/Functions
if not os.getenv('WEBSITE_INSTANCE_ID'):
    load_dotenv()
    logging.info("Running in development mode - loaded .env file")
else:
    logging.info("Running in Azure - using Application Settings")

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

tags_metadata = [
    {
        "name": "System",
        "description": "Endpoints de sistema y diagnóstico para verificar el estado del servicio y la base de datos.",
    },
    {
        "name": "Puntos de Medición",
        "description": "Consulta de puntos de medición con coordenadas UTM, información de cuenca y estadísticas de caudal. Soporta filtros por región, cuenca, subcuenca y rangos de caudal.",
    },
    {
        "name": "Cuencas Hidrográficas",
        "description": "Información sobre cuencas, subcuencas y subsubcuencas hidrográficas, incluyendo estadísticas agregadas.",
    },
    {
        "name": "Series Temporales",
        "description": "Series temporales de caudal, altura limnimétrica y nivel freático por cuenca, subcuenca o punto específico.",
    },
    {
        "name": "Atlas",
        "description": "Divisiones administrativas (regiones, provincias y comunas) disponibles en el sistema.",
    },
    {
        "name": "Cache y Rendimiento",
        "description": "Gestión de caché y optimización del rendimiento de consultas.",
    },
]

app = FastAPI(
    title="Aguas Transparentes API",
    description="API de Recursos Hídricos de Chile. Proporciona acceso a datos de mediciones de caudal, cuencas hidrográficas y series temporales almacenados en Azure Synapse Analytics. Sistema UTM Zona 19S.",
    version="1.5.3",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    root_path="/api",
    openapi_tags=tags_metadata,
    contact={
        "name": "Aguas Transparentes",
        "url": "https://github.com/JavierFuenza/Backend_aguas_cloud",
    }
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
    utm_norte: int = Field(..., ge=0, le=10000000, description="Coordenada UTM Norte (metros)")
    utm_este: int = Field(..., ge=0, le=1000000, description="Coordenada UTM Este (metros)")

    class Config:
        json_schema_extra = {
            "example": {
                "utm_norte": 6300000,
                "utm_este": 350000
            }
        }

class PuntoResponse(BaseModel):
    utm_norte: int = Field(..., description="Coordenada UTM Norte")
    utm_este: int = Field(..., description="Coordenada UTM Este")
    huso: int = Field(..., description="Huso UTM (zona)")
    es_pozo_subterraneo: bool = Field(..., description="Indica si es un pozo subterráneo")

    class Config:
        json_schema_extra = {
            "example": {
                "utm_norte": 6300000,
                "utm_este": 350000,
                "huso": 19,
                "es_pozo_subterraneo": False
            }
        }

class PuntoInfoResponse(BaseModel):
    utm_norte: int
    utm_este: int
    huso: int
    es_pozo_subterraneo: bool
    cod_cuenca: Optional[int] = None
    cod_subcuenca: Optional[int] = None
    nombre_cuenca: Optional[str] = None
    nombre_subcuenca: Optional[str] = None
    caudal_promedio: Optional[float] = Field(None, description="Caudal promedio en l/s")
    n_mediciones: int = Field(..., description="Número de mediciones registradas")

    class Config:
        json_schema_extra = {
            "example": {
                "utm_norte": 6300000,
                "utm_este": 350000,
                "huso": 19,
                "es_pozo_subterraneo": False,
                "cod_cuenca": 101,
                "cod_subcuenca": 10101,
                "nombre_cuenca": "Río Lluta",
                "nombre_subcuenca": "Río Lluta Alto",
                "caudal_promedio": 25.5,
                "n_mediciones": 120
            }
        }

class CuencaData(BaseModel):
    cod_cuenca: Optional[int] = Field(None, description="Código de cuenca")
    nom_cuenca: Optional[str] = Field(None, description="Nombre de cuenca")
    cod_subcuenca: Optional[int] = Field(None, description="Código de subcuenca")
    nom_subcuenca: Optional[str] = Field(None, description="Nombre de subcuenca")
    cod_subsubcuenca: Optional[int] = Field(None, description="Código de subsubcuenca")
    nom_subsubcuenca: Optional[str] = Field(None, description="Nombre de subsubcuenca")
    cod_region: Optional[int] = Field(None, description="Código de región")

    class Config:
        json_schema_extra = {
            "example": {
                "cod_cuenca": 101,
                "nom_cuenca": "Río Lluta",
                "cod_subcuenca": 10101,
                "nom_subcuenca": "Río Lluta Alto",
                "cod_subsubcuenca": None,
                "nom_subsubcuenca": None,
                "cod_region": 15
            }
        }

class CuencaStatsResponse(BaseModel):
    cod_cuenca: Optional[int]
    nom_cuenca: Optional[str]
    cod_region: Optional[int]
    cod_subcuenca: Optional[int] = None
    nom_subcuenca: Optional[str] = None
    cod_subsubcuenca: Optional[int] = None
    nom_subsubcuenca: Optional[str] = None
    caudal_promedio: Optional[float] = Field(None, description="Caudal promedio en l/s")
    caudal_minimo: Optional[float] = Field(None, description="Caudal mínimo registrado en l/s")
    caudal_maximo: Optional[float] = Field(None, description="Caudal máximo registrado en l/s")
    total_puntos_unicos: int = Field(..., description="Número de puntos únicos de medición")
    total_mediciones: int = Field(..., description="Número total de mediciones")

    class Config:
        json_schema_extra = {
            "example": {
                "cod_cuenca": 101,
                "nom_cuenca": "Río Lluta",
                "cod_region": 15,
                "cod_subcuenca": None,
                "nom_subcuenca": None,
                "cod_subsubcuenca": None,
                "nom_subsubcuenca": None,
                "caudal_promedio": 45.3,
                "caudal_minimo": 5.2,
                "caudal_maximo": 120.5,
                "total_puntos_unicos": 15,
                "total_mediciones": 1850
            }
        }

class TimeSeriesPoint(BaseModel):
    fecha_medicion: str = Field(..., description="Fecha de medición (ISO format)")
    caudal: Optional[float] = Field(None, description="Caudal medido en l/s")

    class Config:
        json_schema_extra = {
            "example": {
                "fecha_medicion": "2023-06-15",
                "caudal": 35.2
            }
        }

class AlturaTimeSeriesPoint(BaseModel):
    fecha_medicion: str = Field(..., description="Fecha de medición (ISO format)")
    altura_linimetrica: Optional[float] = Field(None, description="Altura limnimétrica en metros")

    class Config:
        json_schema_extra = {
            "example": {
                "fecha_medicion": "2023-06-15",
                "altura_linimetrica": 2.5
            }
        }

class NivelFreaticoTimeSeriesPoint(BaseModel):
    fecha_medicion: str = Field(..., description="Fecha de medición (ISO format)")
    nivel_freatico: Optional[float] = Field(None, description="Nivel freático en metros")

    class Config:
        json_schema_extra = {
            "example": {
                "fecha_medicion": "2023-06-15",
                "nivel_freatico": 15.3
            }
        }

class HealthResponse(BaseModel):
    status: str = Field(..., description="Estado del servicio")
    message: str = Field(..., description="Mensaje descriptivo")
    database: str = Field(..., description="Estado de la conexión a base de datos")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "message": "Water Data API is running",
                "database": "connected"
            }
        }

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

@app.get(
    "/health",
    tags=["System"],
    response_model=HealthResponse,
    summary="Verificación de estado del servicio",
    description="Verifica el estado del servicio API y la conectividad con la base de datos Azure Synapse. Retorna el estado del servicio y de la conexión a base de datos."
)
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
        results = execute_query("SELECT COUNT(*) as total FROM dw.Mediciones_full")
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
        results = execute_query("SELECT COUNT(*) as total FROM dw.Mediciones_full")
        return {"total_records": results[0]['total']}
    except Exception as e:
        logging.error(f"Error in get_obras_count: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/puntos/count", tags=["Puntos de Medición"])
async def get_puntos_count(
    region: Optional[int] = Query(None),
    cod_cuenca: Optional[int] = Query(None),
    cod_subcuenca: Optional[int] = Query(None),
    filtro_null_subcuenca: Optional[bool] = Query(None, description="Si es True, filtra por subcuenca nula"),
    caudal_minimo: Optional[float] = Query(None),
    caudal_maximo: Optional[float] = Query(None)
):
    """Obtiene el número de puntos únicos desde Puntos_Mapa con filtros"""
    try:
        logging.info(f"Contando puntos con filtros")

        count_query = """
        SELECT COUNT(*) as total_puntos_unicos
        FROM dw.Puntos_Mapa
        WHERE 1=1
        """

        query_params = []

        if region is not None:
            count_query += " AND Region = ?"
            query_params.append(region)

        if cod_cuenca is not None:
            count_query += " AND Cod_Cuenca = ?"
            query_params.append(cod_cuenca)

        # Handle subcuenca filtering logic
        if filtro_null_subcuenca:
            count_query += " AND Cod_Subcuenca IS NULL"
        elif cod_subcuenca is not None:
            count_query += " AND Cod_Subcuenca = ?"
            query_params.append(cod_subcuenca)

        if caudal_minimo is not None:
            count_query += " AND caudal_promedio >= ?"
            query_params.append(caudal_minimo)

        if caudal_maximo is not None:
            count_query += " AND caudal_promedio <= ?"
            query_params.append(caudal_maximo)

        logging.info(f"Ejecutando query count: {count_query}")
        results = execute_query(count_query, query_params)

        total_puntos = results[0]['total_puntos_unicos'] if results else 0

        response = {
            "total_puntos_unicos": total_puntos,
            "filtros_aplicados": {
                "region": region,
                "cod_cuenca": cod_cuenca,
                "cod_subcuenca": cod_subcuenca,
                "filtro_null_subcuenca": filtro_null_subcuenca,
                "caudal_minimo": caudal_minimo,
                "caudal_maximo": caudal_maximo
            }
        }

        logging.info(f"Total puntos únicos encontrados: {total_puntos}")
        return response

    except Exception as e:
        logging.error(f"Error in get_puntos_count: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get(
    "/cache/stats",
    tags=["Cache y Rendimiento"],
    summary="Estadísticas del sistema de caché",
    description="Obtiene estadísticas del sistema de caché en memoria. Incluye número de consultas cacheadas, claves activas, tamaño de cada caché y conexiones disponibles en el pool."
)
async def get_cache_stats():
    """Get cache statistics"""
    return {
        "cached_queries": len(memory_cache),
        "cache_keys": list(memory_cache.keys()),
        "cache_sizes": {k: len(v) for k, v in memory_cache.items()},
        "pool_connections": connection_pool.qsize() if connection_pool else 0
    }

@app.post(
    "/cache/clear",
    tags=["Cache y Rendimiento"],
    summary="Limpiar caché del sistema",
    description="Elimina todos los datos almacenados en la caché del sistema. Las próximas consultas consultarán directamente la base de datos y el caché se reconstruirá automáticamente."
)
async def clear_cache():
    """Clear all cached data"""
    global memory_cache, cache_timestamps
    memory_cache.clear()
    cache_timestamps.clear()
    return {"message": "Cache cleared successfully"}

@app.get(
    "/performance/warm-up",
    tags=["Cache y Rendimiento"],
    summary="Pre-calentar caché del sistema",
    description="Pre-carga las consultas más frecuentes en el caché para mejorar el rendimiento. Útil al iniciar el sistema o después de limpiar el caché."
)
async def warm_up_cache():
    """Pre-warm frequently accessed data"""
    try:
        logging.info("Starting cache warm-up...")

        # Warm up common queries
        queries = [
            ("SELECT COUNT(*) as total FROM dw.Mediciones_full", None),
            ("SELECT COUNT(DISTINCT CONCAT(UTM_Norte, '-', UTM_Este)) as total_puntos_unicos FROM dw.Mediciones_full g WHERE g.UTM_Norte IS NOT NULL AND g.UTM_Este IS NOT NULL", None),
            ("SELECT DISTINCT Region FROM dw.Mediciones_full WHERE Region IS NOT NULL ORDER BY Region", None),
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

@app.get(
    "/puntos",
    tags=["Puntos de Medición"],
    response_model=List[PuntoResponse],
    summary="Obtener puntos de medición",
    description="Obtiene la lista de puntos de medición con coordenadas UTM e indicador de pozo subterráneo. Soporta filtros por región, cuenca, subcuenca, caudal y más."
)
async def get_puntos(
    region: Optional[int] = Query(None, description="Código de región (ej: 15 para Arica y Parinacota)"),
    cod_cuenca: Optional[int] = Query(None, description="Código de cuenca"),
    cod_subcuenca: Optional[int] = Query(None, description="Código de subcuenca"),
    filtro_null_subcuenca: Optional[bool] = Query(None, description="Si es True, filtra por subcuenca nula. Ignora 'cod_subcuenca' si es True."),
    caudal_minimo: Optional[float] = Query(None, description="Caudal promedio mínimo (l/s)"),
    caudal_maximo: Optional[float] = Query(None, description="Caudal promedio máximo (l/s)"),
    limit: Optional[int] = Query(120, description="Número máximo de puntos a retornar")
):
    """Obtiene puntos desde la tabla pre-agregada Puntos_Mapa con filtros"""
    try:
        logging.info(f"Parametros recibidos en /puntos: region={region}, cod_cuenca={cod_cuenca}, cod_subcuenca={cod_subcuenca}")

        puntos_query = """
        SELECT
            UTM_Norte,
            UTM_Este,
            Huso,
            es_pozo_subterraneo
        FROM dw.Puntos_Mapa
        WHERE UTM_Norte IS NOT NULL
          AND UTM_Este IS NOT NULL
        """

        query_params = []

        if region is not None:
            puntos_query += " AND Region = ?"
            query_params.append(region)

        if cod_cuenca is not None:
            puntos_query += " AND Cod_Cuenca = ?"
            query_params.append(cod_cuenca)

        # Handle subcuenca filtering logic
        if filtro_null_subcuenca:
            puntos_query += " AND Cod_Subcuenca IS NULL"
        elif cod_subcuenca is not None:
            puntos_query += " AND Cod_Subcuenca = ?"
            query_params.append(cod_subcuenca)

        if caudal_minimo is not None:
            puntos_query += " AND caudal_promedio >= ?"
            query_params.append(caudal_minimo)

        if caudal_maximo is not None:
            puntos_query += " AND caudal_promedio <= ?"
            query_params.append(caudal_maximo)

        # Apply limit
        if limit is not None:
            puntos_query = f"SELECT TOP {limit} * FROM ({puntos_query}) AS filtered_puntos"

        logging.info(f"Ejecutando query desde Puntos_Mapa: {puntos_query}")
        puntos = execute_query(puntos_query, query_params)

        logging.info(f"Se obtuvieron {len(puntos)} puntos desde Puntos_Mapa")

        # Build response
        puntos_out = []

        for punto in puntos:
            puntos_out.append({
                "utm_norte": punto["UTM_Norte"],
                "utm_este": punto["UTM_Este"],
                "huso": punto["Huso"],
                "es_pozo_subterraneo": bool(punto.get("es_pozo_subterraneo", 0))
            })

        logging.info(f"Retornando {len(puntos_out)} puntos")
        return puntos_out

    except Exception as e:
        logging.error(f"Error en get_puntos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get(
    "/puntos/info",
    tags=["Puntos de Medición"],
    response_model=PuntoInfoResponse,
    summary="Información detallada de un punto",
    description="Obtiene información detallada de un punto de medición específico incluyendo cuenca, subcuenca y estadísticas de caudal. Requiere coordenadas UTM Norte y Este."
)
async def get_punto_info(
    utm_norte: int = Query(..., description="Coordenada UTM Norte en metros", example=6300000),
    utm_este: int = Query(..., description="Coordenada UTM Este en metros", example=350000)
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
            es_pozo_subterraneo
        FROM dw.Puntos_Mapa
        WHERE UTM_Norte = ?
          AND UTM_Este = ?
        """

        punto_result = execute_query(punto_query, [utm_norte, utm_este])

        if not punto_result:
            raise HTTPException(status_code=404, detail="Punto no encontrado")

        punto = punto_result[0]

        # Get cuenca info based on UTM coordinates
        cuenca_query = """
        SELECT TOP 1
            Cod_Cuenca,
            Nom_Cuenca,
            Cod_Subcuenca,
            Nom_Subcuenca
        FROM dw.Mediciones_full
        WHERE UTM_Norte = ?
          AND UTM_Este = ?
        """
        cuenca_result = execute_query(cuenca_query, [utm_norte, utm_este])
        cuenca = cuenca_result[0] if cuenca_result else {}

        # Get caudal statistics for this specific point
        caudal_query = """
        SELECT
            AVG(CAST(Caudal AS FLOAT)) as caudal_promedio,
            MIN(CAST(Caudal AS FLOAT)) as caudal_minimo,
            MAX(CAST(Caudal AS FLOAT)) as caudal_maximo,
            COUNT(*) as n_mediciones
        FROM dw.Datos
        WHERE UTM_Norte = ?
          AND UTM_Este = ?
          AND Caudal IS NOT NULL
        """
        caudal_result = execute_query(caudal_query, [utm_norte, utm_este])
        caudal_stats = caudal_result[0] if caudal_result else {}

        # Build detailed response
        response = {
            "utm_norte": utm_norte,
            "utm_este": utm_este,
            "huso": punto.get('Huso'),
            "es_pozo_subterraneo": bool(punto.get('es_pozo_subterraneo', 0)),
            "cod_cuenca": cuenca.get('Cod_Cuenca'),
            "cod_subcuenca": cuenca.get('Cod_Subcuenca'),
            "nombre_cuenca": cuenca.get('Nom_Cuenca'),
            "nombre_subcuenca": cuenca.get('Nom_Subcuenca'),
            "caudal_promedio": safe_round(caudal_stats.get('caudal_promedio')),
            "n_mediciones": caudal_stats.get('n_mediciones', 0)
        }

        logging.info(f"Info detallada obtenida para punto {utm_norte}/{utm_este}")
        return response

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error en get_punto_info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get(
    "/cuencas",
    tags=["Cuencas Hidrográficas"],
    summary="Listado de cuencas hidrográficas",
    description="Obtiene el listado completo de cuencas, subcuencas y subsubcuencas hidrográficas con sus códigos, nombres y región asociada."
)
async def get_unique_cuencas():
    """Obtiene cuencas, subcuencas y subsubcuencas únicas"""
    try:
        # Query from pre-aggregated table
        cuencas_query = """
        SELECT
            Cod_Cuenca as cod_cuenca,
            Nom_Cuenca as nom_cuenca,
            Cod_Subcuenca as cod_subcuenca,
            Nom_Subcuenca as nom_subcuenca,
            Cod_Subsubcuenca as cod_subsubcuenca,
            Nom_Subsubcuenca as nom_subsubcuenca,
            Cod_Region as cod_region
        FROM dw.Cuencas_Regiones
        ORDER BY Cod_Cuenca, Cod_Subcuenca, Cod_Subsubcuenca
        """

        results = execute_query(cuencas_query)

        return {
            "cuencas": [
                {
                    "cod_cuenca": r.get('cod_cuenca'),
                    "nom_cuenca": r.get('nom_cuenca'),
                    "cod_region": r.get('cod_region'),
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

@app.get(
    "/atlas",
    tags=["Atlas"],
    summary="Divisiones administrativas de Chile",
    description="Obtiene el listado de divisiones administrativas disponibles: regiones, provincias y comunas."
)
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

@app.get(
    "/filtrosreactivos",
    tags=["Cuencas Hidrográficas"],
    summary="Estadísticas de caudal para filtros reactivos",
    description="Obtiene estadísticas de caudal mínimo y máximo agregadas globalmente, por cuenca y por subcuenca. Usado para configurar filtros reactivos en el frontend."
)
async def get_filtros_reactivos():
    """Obtiene estadísticas de caudal para filtros reactivos desde tabla pre-agregada"""
    try:
        # Query the pre-aggregated table
        stats_query = """
        SELECT
            nivel,
            nom_cuenca,
            nom_subcuenca,
            avgMin,
            avgMax,
            total_puntos
        FROM dw.Filtros_Reactivos_Stats
        ORDER BY
            CASE nivel
                WHEN 'global' THEN 1
                WHEN 'cuenca' THEN 2
                WHEN 'subcuenca' THEN 3
            END,
            nom_cuenca,
            nom_subcuenca
        """
        results = execute_query(stats_query)

        # Separate results by nivel
        global_stats = {}
        cuenca_stats = []
        subcuenca_stats = []

        for r in results:
            nivel = r.get('nivel')
            if nivel == 'global':
                global_stats = {
                    "avgMin": safe_round(r.get('avgMin')),
                    "avgMax": safe_round(r.get('avgMax')),
                    "total_puntos_unicos": r.get('total_puntos', 0)
                }
            elif nivel == 'cuenca':
                cuenca_stats.append({
                    "nom_cuenca": r.get('nom_cuenca'),
                    "avgMin": safe_round(r.get('avgMin')),
                    "avgMax": safe_round(r.get('avgMax')),
                    "total_puntos": r.get('total_puntos', 0)
                })
            elif nivel == 'subcuenca':
                subcuenca_stats.append({
                    "nom_cuenca": r.get('nom_cuenca'),
                    "nom_subcuenca": r.get('nom_subcuenca'),
                    "avgMin": safe_round(r.get('avgMin')),
                    "avgMax": safe_round(r.get('avgMax')),
                    "total_puntos": r.get('total_puntos', 0)
                })

        return {
            "estadisticas": {
                "caudal_global": global_stats,
                "caudal_por_cuenca": cuenca_stats,
                "caudal_por_subcuenca": subcuenca_stats
            }
        }
    except Exception as e:
        logging.error(f"Error in get_filtros_reactivos: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get(
    "/cuencas/stats",
    tags=["Cuencas Hidrográficas"],
    summary="Estadísticas de caudal por cuenca",
    description="Obtiene estadísticas de caudal agregadas por cuenca, subcuenca o subsubcuenca. Incluye caudal promedio, mínimo, máximo, total de puntos y mediciones. Opcionalmente incluye estadísticas globales del sistema."
)
async def get_cuencas_stats(
    cod_cuenca: Optional[int] = Query(None, description="Código de cuenca", example=101),
    cod_subcuenca: Optional[int] = Query(None, description="Código de subcuenca", example=10101),
    cod_subsubcuenca: Optional[int] = Query(None, description="Código de subsubcuenca"),
    include_global: bool = Query(False, description="Incluir estadísticas globales del sistema completo")
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

        # Query pre-aggregated table (Cod_Region is now included in the table)
        stats_query = f"""
        SELECT
            Cod_Cuenca,
            Nom_Cuenca,
            Cod_Subcuenca,
            Nom_Subcuenca,
            Cod_Subsubcuenca,
            Nom_Subsubcuenca,
            Cod_Region,
            caudal_promedio,
            caudal_minimo,
            caudal_maximo,
            total_puntos_unicos,
            total_mediciones
        FROM dw.Cuenca_Stats
        {where_clause}
        ORDER BY Cod_Cuenca, Cod_Subcuenca, Cod_Subsubcuenca
        """

        results = execute_query(stats_query, params)

        if not results:
            return {"estadisticas": []}

        # Get global statistics only if requested
        global_stats = {}
        if include_global:
            global_stats_query = """
            SELECT
                AVG(CAST(Caudal AS FLOAT)) as global_promedio,
                MIN(CAST(Caudal AS FLOAT)) as global_minimo,
                MAX(CAST(Caudal AS FLOAT)) as global_maximo
            FROM dw.Datos
            WHERE Caudal IS NOT NULL
            """
            global_result = execute_query(global_stats_query)
            global_stats = global_result[0] if global_result else {}

        # Build response
        estadisticas = []
        for r in results:
            stat = {
                "cod_cuenca": r.get('Cod_Cuenca'),
                "nom_cuenca": r.get('Nom_Cuenca'),
                "cod_region": r.get('Cod_Region'),
                "cod_subcuenca": r.get('Cod_Subcuenca'),
                "nom_subcuenca": r.get('Nom_Subcuenca'),
                "cod_subsubcuenca": r.get('Cod_Subsubcuenca'),
                "nom_subsubcuenca": r.get('Nom_Subsubcuenca'),
                "caudal_promedio": safe_round(r.get('caudal_promedio')),
                "caudal_minimo": safe_round(r.get('caudal_minimo')),
                "caudal_maximo": safe_round(r.get('caudal_maximo')),
                "total_puntos_unicos": r.get('total_puntos_unicos', 0),
                "total_mediciones": r.get('total_mediciones', 0)
            }

            # Add global stats only if requested
            if include_global:
                stat["global_promedio"] = safe_round(global_stats.get('global_promedio'))
                stat["global_minimo"] = safe_round(global_stats.get('global_minimo'))
                stat["global_maximo"] = safe_round(global_stats.get('global_maximo'))

            estadisticas.append(stat)

        return {"estadisticas": estadisticas}

    except Exception as e:
        logging.error(f"Error in get_cuencas_stats: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/cuencas/cuenca/series_de_tiempo/caudal",
    tags=["Series Temporales"],
    summary="Serie temporal de caudal por cuenca",
    description="Obtiene la serie temporal de caudal para una cuenca específica (máximo 1000 registros más recientes). Acepta código numérico o nombre de cuenca. Opcionalmente filtra por rango de fechas (YYYY-MM-DD)."
)
async def get_caudal_por_tiempo_por_cuenca(
    cuenca_identificador: str = Query(..., description="Código numérico o nombre de la cuenca", example="101"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio en formato YYYY-MM-DD", example="2023-01-01"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin en formato YYYY-MM-DD", example="2023-12-31")
):
    try:
        filters = []
        params = []

        # Filtro por cuenca (código o nombre)
        if cuenca_identificador.isdigit():
            filters.append("COD_CUENCA = ?")
            params.append(int(cuenca_identificador))
        else:
            filters.append("NOM_CUENCA = ?")
            params.append(cuenca_identificador)

        # Filtros de fecha
        if fecha_inicio:
            filters.append("FECHA_MEDICION >= ?")
            params.append(fecha_inicio)

        if fecha_fin:
            filters.append("FECHA_MEDICION <= ?")
            params.append(fecha_fin)

        where_clause = " AND ".join(filters)

        query = f"""
        SELECT TOP (1000)
            FECHA_MEDICION AS fecha_medicion,
            CAUDAL AS caudal
        FROM dw.Series_Tiempo
        WHERE {where_clause}
          AND CAUDAL IS NOT NULL
        ORDER BY FECHA_MEDICION DESC
        """

        results = execute_query(query, params)

        if not results:
            raise HTTPException(status_code=404, detail="No se encontraron datos de caudal para el período o cuenca especificada.")

        caudal_por_tiempo = [
            {
                "fecha_medicion": str(r["fecha_medicion"]) if r.get("fecha_medicion") else None,
                "caudal": r["caudal"]
            }
            for r in results
        ]

        return {
            "cuenca_identificador": cuenca_identificador,
            "total_registros": len(caudal_por_tiempo),
            "caudal_por_tiempo": caudal_por_tiempo
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_caudal_por_tiempo_por_cuenca: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/cuencas/subcuenca/series_de_tiempo/caudal",
    tags=["Series Temporales"],
    summary="Serie temporal de caudal por subcuenca",
    description="Obtiene la serie temporal de caudal para una subcuenca específica (máximo 1000 registros más recientes). Acepta código numérico o nombre de subcuenca. Opcionalmente filtra por rango de fechas (YYYY-MM-DD)."
)
async def get_caudal_por_tiempo_por_subcuenca(
    cuenca_identificador: str = Query(..., description="Código numérico o nombre de la subcuenca", example="205"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio en formato YYYY-MM-DD", example="2023-01-01"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin en formato YYYY-MM-DD", example="2023-12-31")
):
    try:
        filters = []
        params = []

        # Filtro por subcuenca (código o nombre)
        if cuenca_identificador.isdigit():
            filters.append("Cod_Subcuenca = ?")
            params.append(int(cuenca_identificador))
        else:
            filters.append("Nom_Subcuenca = ?")
            params.append(cuenca_identificador)

        # Filtros de fecha
        if fecha_inicio:
            filters.append("Fecha_Medicion >= ?")
            params.append(fecha_inicio)

        if fecha_fin:
            filters.append("Fecha_Medicion <= ?")
            params.append(fecha_fin)

        where_clause = " AND ".join(filters)

        query = f"""
        SELECT TOP (1000)
            Fecha_Medicion AS fecha_medicion,
            Caudal AS caudal
        FROM dw.Datos
        WHERE {where_clause}
          AND Caudal IS NOT NULL
        ORDER BY Fecha_Medicion DESC
        """

        results = execute_query(query, params)

        if not results:
            raise HTTPException(
                status_code=404,
                detail="No se encontraron datos de caudal para el período o subcuenca especificada."
            )

        caudal_por_tiempo = [
            {
                "fecha_medicion": str(r["fecha_medicion"]) if r.get("fecha_medicion") else None,
                "caudal": r["caudal"]
            }
            for r in results
        ]

        return {
            "subcuenca_identificador": cuenca_identificador,
            "total_registros": len(caudal_por_tiempo),
            "caudal_por_tiempo": caudal_por_tiempo
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_caudal_por_tiempo_por_subcuenca: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/cuencas/subsubcuenca/series_de_tiempo/caudal",
    tags=["Series Temporales"],
    summary="Serie temporal de caudal por subsubcuenca",
    description="Obtiene la serie temporal de caudal para una subsubcuenca específica (máximo 1000 registros más recientes). Acepta código numérico o nombre de subsubcuenca. Opcionalmente filtra por rango de fechas (YYYY-MM-DD)."
)
async def get_caudal_por_tiempo_por_subsubcuenca(
    cuenca_identificador: str = Query(..., description="Código numérico o nombre de la subsubcuenca", example="305"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio en formato YYYY-MM-DD", example="2023-01-01"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin en formato YYYY-MM-DD", example="2023-12-31")
):
    try:
        filters = []
        params = []

        # Filtro por subsubcuenca (código o nombre)
        if cuenca_identificador.isdigit():
            filters.append("Cod_Subsubcuenca = ?")
            params.append(int(cuenca_identificador))
        else:
            filters.append("Nom_Subsubcuenca = ?")
            params.append(cuenca_identificador)

        # Filtros de fecha
        if fecha_inicio:
            filters.append("Fecha_Medicion >= ?")
            params.append(fecha_inicio)

        if fecha_fin:
            filters.append("Fecha_Medicion <= ?")
            params.append(fecha_fin)

        where_clause = " AND ".join(filters)

        query = f"""
        SELECT TOP (1000)
            Fecha_Medicion AS fecha_medicion,
            Caudal AS caudal
        FROM dw.Datos
        WHERE {where_clause}
          AND Caudal IS NOT NULL
        ORDER BY Fecha_Medicion DESC
        """

        results = execute_query(query, params)

        if not results:
            raise HTTPException(
                status_code=404,
                detail="No se encontraron datos de caudal para el período o subsubcuenca especificada."
            )

        caudal_por_tiempo = [
            {
                "fecha_medicion": str(r["fecha_medicion"]) if r.get("fecha_medicion") else None,
                "caudal": r["caudal"]
            }
            for r in results
        ]

        return {
            "subsubcuenca_identificador": cuenca_identificador,
            "total_registros": len(caudal_por_tiempo),
            "caudal_por_tiempo": caudal_por_tiempo
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_caudal_por_tiempo_por_subsubcuenca: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/cuencas/cuenca/series_de_tiempo/altura_linimetrica",
    tags=["Series Temporales"],
    summary="Serie temporal de altura limnimétrica por cuenca",
    description="Obtiene la serie temporal de altura limnimétrica para una cuenca específica (máximo 1000 registros más recientes). Acepta código numérico o nombre de cuenca. Opcionalmente filtra por rango de fechas (YYYY-MM-DD)."
)
async def get_altura_linimetrica_por_tiempo_por_cuenca(
    cuenca_identificador: str = Query(..., description="Código numérico o nombre de la cuenca", example="101"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio en formato YYYY-MM-DD", example="2023-01-01"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin en formato YYYY-MM-DD", example="2023-12-31")
):
    try:
        filters = []
        params = []

        # Filtro por cuenca (código o nombre)
        if cuenca_identificador.isdigit():
            filters.append("COD_CUENCA = ?")
            params.append(int(cuenca_identificador))
        else:
            filters.append("NOM_CUENCA = ?")
            params.append(cuenca_identificador)

        # Filtros de fecha
        if fecha_inicio:
            filters.append("FECHA_MEDICION >= ?")
            params.append(fecha_inicio)

        if fecha_fin:
            filters.append("FECHA_MEDICION <= ?")
            params.append(fecha_fin)

        where_clause = " AND ".join(filters)

        query = f"""
        SELECT TOP (1000)
            FECHA_MEDICION AS fecha_medicion,
            ALTURA_LIMNIMETRICA AS altura_linimetrica
        FROM dw.Series_Tiempo
        WHERE {where_clause}
          AND ALTURA_LIMNIMETRICA IS NOT NULL
        ORDER BY FECHA_MEDICION DESC
        """

        results = execute_query(query, params)

        if not results:
            raise HTTPException(
                status_code=404,
                detail="No se encontraron datos de altura limnimétrica para el período o cuenca especificada."
            )

        altura_por_tiempo = [
            {
                "fecha_medicion": str(r["fecha_medicion"]) if r.get("fecha_medicion") else None,
                "altura_linimetrica": r["altura_linimetrica"]
            }
            for r in results
        ]

        return {
            "cuenca_identificador": cuenca_identificador,
            "total_registros": len(altura_por_tiempo),
            "altura_por_tiempo": altura_por_tiempo
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_altura_linimetrica_por_tiempo_por_cuenca: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/cuencas/cuenca/series_de_tiempo/nivel_freatico",
    tags=["Series Temporales"],
    summary="Serie temporal de nivel freático por cuenca",
    description="Obtiene la serie temporal de nivel freático para una cuenca específica (máximo 1000 registros más recientes). Acepta código numérico o nombre de cuenca. Opcionalmente filtra por rango de fechas (YYYY-MM-DD)."
)
async def get_nivel_freatico_por_tiempo_por_cuenca(
    cuenca_identificador: str = Query(..., description="Código numérico o nombre de la cuenca", example="101"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio en formato YYYY-MM-DD", example="2023-01-01"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin en formato YYYY-MM-DD", example="2023-12-31")
):
    try:
        filters = []
        params = []

        # Filtro por cuenca (código o nombre)
        if cuenca_identificador.isdigit():
            filters.append("COD_CUENCA = ?")
            params.append(int(cuenca_identificador))
        else:
            filters.append("NOM_CUENCA = ?")
            params.append(cuenca_identificador)

        # Filtros de fecha
        if fecha_inicio:
            filters.append("FECHA_MEDICION >= ?")
            params.append(fecha_inicio)

        if fecha_fin:
            filters.append("FECHA_MEDICION <= ?")
            params.append(fecha_fin)

        where_clause = " AND ".join(filters)

        query = f"""
        SELECT TOP (1000)
            FECHA_MEDICION AS fecha_medicion,
            NIVEL_FREATICO AS nivel_freatico
        FROM dw.Series_Tiempo
        WHERE {where_clause}
          AND NIVEL_FREATICO IS NOT NULL
        ORDER BY FECHA_MEDICION DESC
        """

        results = execute_query(query, params)

        if not results:
            raise HTTPException(
                status_code=404,
                detail="No se encontraron datos de nivel freático para el período o cuenca especificada."
            )

        nivel_por_tiempo = [
            {
                "fecha_medicion": str(r["fecha_medicion"]) if r.get("fecha_medicion") else None,
                "nivel_freatico": r["nivel_freatico"]
            }
            for r in results
        ]

        return {
            "cuenca_identificador": cuenca_identificador,
            "total_registros": len(nivel_por_tiempo),
            "nivel_por_tiempo": nivel_por_tiempo
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_nivel_freatico_por_tiempo_por_cuenca: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/cuencas/subcuenca/series_de_tiempo/nivel_freatico",
    tags=["Series Temporales"],
    summary="Serie temporal de nivel freático por subcuenca",
    description="Obtiene la serie temporal de nivel freático para una subcuenca específica (máximo 1000 registros más recientes). Acepta código numérico o nombre de subcuenca. Opcionalmente filtra por rango de fechas (YYYY-MM-DD)."
)
async def get_nivel_freatico_por_tiempo_por_subcuenca(
    cuenca_identificador: str = Query(..., description="Código numérico o nombre de la subcuenca", example="205"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio en formato YYYY-MM-DD", example="2023-01-01"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin en formato YYYY-MM-DD", example="2023-12-31")
):
    try:
        filters = []
        params = []

        if cuenca_identificador.isdigit():
            filters.append("Cod_Subcuenca = ?")
            params.append(int(cuenca_identificador))
        else:
            filters.append("Nom_Subcuenca = ?")
            params.append(cuenca_identificador)

        if fecha_inicio:
            filters.append("Fecha_Medicion >= ?")
            params.append(fecha_inicio)

        if fecha_fin:
            filters.append("Fecha_Medicion <= ?")
            params.append(fecha_fin)

        where_clause = " AND ".join(filters)

        query = f"""
        SELECT TOP (1000)
            Fecha_Medicion AS fecha_medicion,
            Nivel_Freatico AS nivel_freatico
        FROM dw.Datos
        WHERE {where_clause}
          AND Nivel_Freatico IS NOT NULL
        ORDER BY Fecha_Medicion DESC
        """

        results = execute_query(query, params)

        if not results:
            raise HTTPException(
                status_code=404,
                detail="No se encontraron datos de nivel freático para el período o subcuenca especificada."
            )

        nivel_por_tiempo = [
            {
                "fecha_medicion": str(r["fecha_medicion"]) if r.get("fecha_medicion") else None,
                "nivel_freatico": r["nivel_freatico"]
            }
            for r in results
        ]

        return {
            "subcuenca_identificador": cuenca_identificador,
            "total_registros": len(nivel_por_tiempo),
            "nivel_por_tiempo": nivel_por_tiempo
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_nivel_freatico_por_tiempo_por_subcuenca: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/cuencas/subsubcuenca/series_de_tiempo/altura_linimetrica",
    tags=["Series Temporales"],
    summary="Serie temporal de altura limnimétrica por subsubcuenca",
    description="Obtiene la serie temporal de altura limnimétrica para una subsubcuenca específica (máximo 1000 registros más recientes). Acepta código numérico o nombre de subsubcuenca. Opcionalmente filtra por rango de fechas (YYYY-MM-DD)."
)
async def get_altura_linimetrica_por_tiempo_por_subsubcuenca(
    cuenca_identificador: str = Query(..., description="Código numérico o nombre de la subsubcuenca", example="305"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio en formato YYYY-MM-DD", example="2023-01-01"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin en formato YYYY-MM-DD", example="2023-12-31")
):
    try:
        filters = []
        params = []

        if cuenca_identificador.isdigit():
            filters.append("Cod_Subsubcuenca = ?")
            params.append(int(cuenca_identificador))
        else:
            filters.append("Nom_Subsubcuenca = ?")
            params.append(cuenca_identificador)

        if fecha_inicio:
            filters.append("Fecha_Medicion >= ?")
            params.append(fecha_inicio)

        if fecha_fin:
            filters.append("Fecha_Medicion <= ?")
            params.append(fecha_fin)

        where_clause = " AND ".join(filters)

        query = f"""
        SELECT TOP (1000)
            Fecha_Medicion AS fecha_medicion,
            Altura_Limnimetrica AS altura_linimetrica
        FROM dw.Datos
        WHERE {where_clause}
          AND Altura_Limnimetrica IS NOT NULL
        ORDER BY Fecha_Medicion DESC
        """

        results = execute_query(query, params)

        if not results:
            raise HTTPException(
                status_code=404,
                detail="No se encontraron datos de altura limnimétrica para el período o subsubcuenca especificada."
            )

        altura_por_tiempo = [
            {
                "fecha_medicion": str(r["fecha_medicion"]) if r.get("fecha_medicion") else None,
                "altura_linimetrica": r["altura_linimetrica"]
            }
            for r in results
        ]

        return {
            "subsubcuenca_identificador": cuenca_identificador,
            "total_registros": len(altura_por_tiempo),
            "altura_por_tiempo": altura_por_tiempo
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_altura_linimetrica_por_tiempo_por_subsubcuenca: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/cuencas/subsubcuenca/series_de_tiempo/nivel_freatico",
    tags=["Series Temporales"],
    summary="Serie temporal de nivel freático por subsubcuenca",
    description="Obtiene la serie temporal de nivel freático para una subsubcuenca específica (máximo 1000 registros más recientes). Acepta código numérico o nombre de subsubcuenca. Opcionalmente filtra por rango de fechas (YYYY-MM-DD)."
)
async def get_nivel_freatico_por_tiempo_por_subsubcuenca(
    cuenca_identificador: str = Query(..., description="Código numérico o nombre de la subsubcuenca", example="305"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio en formato YYYY-MM-DD", example="2023-01-01"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin en formato YYYY-MM-DD", example="2023-12-31")
):
    try:
        filters = []
        params = []

        if cuenca_identificador.isdigit():
            filters.append("Cod_Subsubcuenca = ?")
            params.append(int(cuenca_identificador))
        else:
            filters.append("Nom_Subsubcuenca = ?")
            params.append(cuenca_identificador)

        if fecha_inicio:
            filters.append("Fecha_Medicion >= ?")
            params.append(fecha_inicio)

        if fecha_fin:
            filters.append("Fecha_Medicion <= ?")
            params.append(fecha_fin)

        where_clause = " AND ".join(filters)

        query = f"""
        SELECT TOP (1000)
            Fecha_Medicion AS fecha_medicion,
            Nivel_Freatico AS nivel_freatico
        FROM dw.Datos
        WHERE {where_clause}
          AND Nivel_Freatico IS NOT NULL
        ORDER BY Fecha_Medicion DESC
        """

        results = execute_query(query, params)

        if not results:
            raise HTTPException(
                status_code=404,
                detail="No se encontraron datos de nivel freático para el período o subsubcuenca especificada."
            )

        nivel_por_tiempo = [
            {
                "fecha_medicion": str(r["fecha_medicion"]) if r.get("fecha_medicion") else None,
                "nivel_freatico": r["nivel_freatico"]
            }
            for r in results
        ]

        return {
            "subsubcuenca_identificador": cuenca_identificador,
            "total_registros": len(nivel_por_tiempo),
            "nivel_por_tiempo": nivel_por_tiempo
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_nivel_freatico_por_tiempo_por_subsubcuenca: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.post("/puntos/estadisticas", tags=["Puntos de Medición"])
async def get_point_statistics(locations: List[UTMLocation]):
    """Obtiene estadísticas de caudal para uno o varios puntos UTM específicos"""
    try:
        if not locations:
            raise HTTPException(status_code=400, detail="Debe proporcionar al menos una coordenada UTM")

        if len(locations) == 1:
            # Single location analysis
            loc = locations[0]

            # Caudal statistics
            caudal_stats_query = """
            SELECT
                COUNT(*) as count,
                AVG(CAST(Caudal AS FLOAT)) as avg_val,
                MIN(CAST(Caudal AS FLOAT)) as min_val,
                MAX(CAST(Caudal AS FLOAT)) as max_val,
                STDEV(CAST(Caudal AS FLOAT)) as std_val,
                MIN(Fecha_Medicion) as primera_fecha,
                MAX(Fecha_Medicion) as ultima_fecha
            FROM dw.Datos
            WHERE UTM_Norte = ? AND UTM_Este = ? AND Caudal IS NOT NULL
            """
            caudal_result = execute_query(caudal_stats_query, [loc.utm_norte, loc.utm_este])
            caudal_stats = caudal_result[0] if caudal_result else {}

            # Altura Limnimetrica statistics
            altura_stats_query = """
            SELECT
                COUNT(*) as count,
                AVG(CAST(Altura_Limnimetrica AS FLOAT)) as avg_val,
                MIN(CAST(Altura_Limnimetrica AS FLOAT)) as min_val,
                MAX(CAST(Altura_Limnimetrica AS FLOAT)) as max_val,
                STDEV(CAST(Altura_Limnimetrica AS FLOAT)) as std_val,
                MIN(Fecha_Medicion) as primera_fecha,
                MAX(Fecha_Medicion) as ultima_fecha
            FROM dw.Datos
            WHERE UTM_Norte = ? AND UTM_Este = ? AND Altura_Limnimetrica IS NOT NULL
            """
            altura_result = execute_query(altura_stats_query, [loc.utm_norte, loc.utm_este])
            altura_stats = altura_result[0] if altura_result else {}

            # Nivel Freatico statistics
            nivel_stats_query = """
            SELECT
                COUNT(*) as count,
                AVG(CAST(Nivel_Freatico AS FLOAT)) as avg_val,
                MIN(CAST(Nivel_Freatico AS FLOAT)) as min_val,
                MAX(CAST(Nivel_Freatico AS FLOAT)) as max_val,
                STDEV(CAST(Nivel_Freatico AS FLOAT)) as std_val,
                MIN(Fecha_Medicion) as primera_fecha,
                MAX(Fecha_Medicion) as ultima_fecha
            FROM dw.Datos
            WHERE UTM_Norte = ? AND UTM_Este = ? AND Nivel_Freatico IS NOT NULL
            """
            nivel_result = execute_query(nivel_stats_query, [loc.utm_norte, loc.utm_este])
            nivel_stats = nivel_result[0] if nivel_result else {}

            response = {
                "utm_norte": loc.utm_norte,
                "utm_este": loc.utm_este
            }

            if caudal_stats.get('count', 0) > 0:
                response["caudal"] = {
                    "total_registros": caudal_stats.get('count'),
                    "promedio": round(caudal_stats.get('avg_val', 0), 2) if caudal_stats.get('avg_val') else None,
                    "minimo": round(caudal_stats.get('min_val', 0), 2) if caudal_stats.get('min_val') else None,
                    "maximo": round(caudal_stats.get('max_val', 0), 2) if caudal_stats.get('max_val') else None,
                    "desviacion_estandar": round(caudal_stats.get('std_val', 0), 2) if caudal_stats.get('std_val') else None,
                    "primera_fecha": str(caudal_stats.get('primera_fecha')) if caudal_stats.get('primera_fecha') else None,
                    "ultima_fecha": str(caudal_stats.get('ultima_fecha')) if caudal_stats.get('ultima_fecha') else None
                }

            if altura_stats.get('count', 0) > 0:
                response["altura_limnimetrica"] = {
                    "total_registros": altura_stats.get('count'),
                    "promedio": round(altura_stats.get('avg_val', 0), 2) if altura_stats.get('avg_val') else None,
                    "minimo": round(altura_stats.get('min_val', 0), 2) if altura_stats.get('min_val') else None,
                    "maximo": round(altura_stats.get('max_val', 0), 2) if altura_stats.get('max_val') else None,
                    "desviacion_estandar": round(altura_stats.get('std_val', 0), 2) if altura_stats.get('std_val') else None,
                    "primera_fecha": str(altura_stats.get('primera_fecha')) if altura_stats.get('primera_fecha') else None,
                    "ultima_fecha": str(altura_stats.get('ultima_fecha')) if altura_stats.get('ultima_fecha') else None
                }

            if nivel_stats.get('count', 0) > 0:
                response["nivel_freatico"] = {
                    "total_registros": nivel_stats.get('count'),
                    "promedio": round(nivel_stats.get('avg_val', 0), 2) if nivel_stats.get('avg_val') else None,
                    "minimo": round(nivel_stats.get('min_val', 0), 2) if nivel_stats.get('min_val') else None,
                    "maximo": round(nivel_stats.get('max_val', 0), 2) if nivel_stats.get('max_val') else None,
                    "desviacion_estandar": round(nivel_stats.get('std_val', 0), 2) if nivel_stats.get('std_val') else None,
                    "primera_fecha": str(nivel_stats.get('primera_fecha')) if nivel_stats.get('primera_fecha') else None,
                    "ultima_fecha": str(nivel_stats.get('ultima_fecha')) if nivel_stats.get('ultima_fecha') else None
                }

            return [response]
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
            FROM dw.Datos
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

@app.get("/puntos/series_de_tiempo/caudal", tags=["Series Temporales"])
async def get_caudal_por_tiempo_por_punto(
    utm_norte: int = Query(..., description="Coordenada UTM Norte del punto"),
    utm_este: int = Query(..., description="Coordenada UTM Este del punto"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio (YYYY-MM-DD)"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin (YYYY-MM-DD)")
):
    """Obtiene el caudal extraído a lo largo del tiempo para un punto UTM específico"""
    try:
        time_series_query = """
        SELECT
            Fecha_Medicion as fecha_medicion,
            Caudal as caudal
        FROM dw.Series_Tiempo
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

@app.get("/puntos/series_de_tiempo/altura_linimetrica", tags=["Series Temporales"])
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
        FROM dw.Series_Tiempo
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
        FROM dw.Series_Tiempo
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

@app.get("/puntos/series_de_tiempo/nivel_freatico", tags=["Series Temporales"])
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
        FROM dw.Series_Tiempo
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
        FROM dw.Series_Tiempo
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

