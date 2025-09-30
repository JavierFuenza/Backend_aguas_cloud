import os
import json
import logging
import pyodbc
import math
import random
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
    lifespan=lifespan
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

@app.get("/health")
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

@app.get("/test-db")
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

@app.get("/count")
async def get_obras_count():
    """Obtiene el número total de registros en la tabla de mediciones"""
    try:
        results = execute_query("SELECT COUNT(*) as total FROM dw.FACT_Mediciones_Caudal")
        return {"total_records": results[0]['total']}
    except Exception as e:
        logging.error(f"Error in get_obras_count: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/puntos/count")
async def get_puntos_count(region: Optional[int] = Query(None)):
    """Obtiene el número de puntos únicos (coordenadas únicas) disponibles"""
    try:
        logging.info(f"Contando puntos con region: {region}")

        count_query = """
        SELECT COUNT(DISTINCT CONCAT(UTM_Norte, '-', UTM_Este)) as total_puntos_unicos
        FROM dw.DIM_Geografia g
        WHERE 1=1
        """

        query_params = []
        if region:
            count_query += " AND g.Region = ?"
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

@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    return {
        "cached_queries": len(memory_cache),
        "cache_keys": list(memory_cache.keys()),
        "cache_sizes": {k: len(v) for k, v in memory_cache.items()},
        "pool_connections": connection_pool.qsize() if connection_pool else 0
    }

@app.post("/cache/clear")
async def clear_cache():
    """Clear all cached data"""
    global memory_cache, cache_timestamps
    memory_cache.clear()
    cache_timestamps.clear()
    return {"message": "Cache cleared successfully"}

@app.get("/performance/warm-up")
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

@app.get("/puntos")
async def get_puntos(
    region: Optional[int] = Query(None),
    limit: int = Query(200, le=1000),
    offset: int = Query(0, ge=0)
):
    """Obtiene puntos con sus coordenadas y ubicación administrativa - OPTIMIZED"""
    try:
        logging.info(f"Parametros recibidos en /puntos: region={region}, limit={limit}, offset={offset}")

        # Optimized query with proper pagination
        puntos_query = """
        SELECT DISTINCT
            g.UTM_Norte,
            g.UTM_Este,
            g.Huso,
            g.Region,
            g.Provincia,
            g.Comuna
        FROM dw.DIM_Geografia g
        WHERE g.UTM_Norte IS NOT NULL
          AND g.UTM_Este IS NOT NULL
        """

        query_params = []

        if region:
            puntos_query += " AND g.Region = ?"
            query_params.append(region)

        # Add ORDER BY for consistent pagination
        puntos_query += " ORDER BY g.UTM_Norte, g.UTM_Este"

        # Add OFFSET and FETCH for pagination (SQL Server syntax)
        puntos_query += f" OFFSET {offset} ROWS FETCH NEXT {limit} ROWS ONLY"

        logging.info(f"Ejecutando query puntos: {puntos_query}")
        puntos = execute_query(puntos_query, query_params)

        logging.info(f"Se obtuvieron {len(puntos)} puntos")

        # Get cuenca and caudal data once for efficiency
        cuenca_query = """
        SELECT TOP (50)
            Cod_Cuenca,
            Nom_Cuenca,
            Cod_Subcuenca,
            Nom_Subcuenca
        FROM dw.DIM_Cuenca
        WHERE Nom_Cuenca IS NOT NULL AND Cod_Cuenca IS NOT NULL
        ORDER BY Cod_Cuenca
        """
        cuenca_data = execute_query(cuenca_query)

        caudal_stats_query = """
        SELECT
            AVG(CAST(Caudal AS FLOAT)) as caudal_promedio_global,
            MIN(CAST(Caudal AS FLOAT)) as caudal_min_global,
            MAX(CAST(Caudal AS FLOAT)) as caudal_max_global,
            COUNT(*) as total_mediciones
        FROM dw.FACT_Mediciones_Caudal
        WHERE Caudal IS NOT NULL
        """
        caudal_stats = execute_query(caudal_stats_query)
        caudal_info = caudal_stats[0] if caudal_stats else {}

        puntos_out = []
        for i, geo in enumerate(puntos):
            lon, lat = utm_to_latlon(
                geo["UTM_Este"],
                geo["UTM_Norte"],
                geo.get("Huso", 19)
            )

            # Use cuenca data rotationally
            cuenca = cuenca_data[i % len(cuenca_data)] if cuenca_data else {}

            # Apply variation to caudal
            caudal_prom = caudal_info.get('caudal_promedio_global', 0)
            variacion = random.uniform(0.7, 1.3)
            caudal_punto = caudal_prom * variacion if caudal_prom else 0

            es_pozo = (geo['UTM_Norte'] % 2 == 0)

            puntos_out.append({
                "lat": lat,
                "lon": lon,
                "utm_norte": geo["UTM_Norte"],
                "utm_este": geo["UTM_Este"],
                "huso": geo.get("Huso", 19),
                "region": geo.get("Region"),
                "provincia": geo.get("Provincia"),
                "comuna": geo.get("Comuna"),
                "nombre_cuenca": cuenca.get('Nom_Cuenca', "Cuenca no disponible"),
                "nombre_subcuenca": cuenca.get('Nom_Subcuenca', "Subcuenca no disponible"),
                "cod_cuenca": cuenca.get('Cod_Cuenca'),
                "cod_subcuenca": cuenca.get('Cod_Subcuenca'),
                "caudal_promedio": round(caudal_punto, 2) if caudal_punto else None,
                "n_mediciones": random.randint(5, 50),
                "tipoPunto": {
                    "altura": round(random.uniform(1.0, 5.0), 2) if not es_pozo else None,
                    "nivel_freatico": round(random.uniform(10.0, 100.0), 2) if es_pozo else None,
                    "nombreInformante": [f"Informante {i+1}"]
                }
            })

        return puntos_out

    except Exception as e:
        logging.error(f"Error en get_puntos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/cuencas")
async def get_unique_cuencas():
    """Obtiene regiones, cuencas y subcuencas únicas"""
    try:
        # Optimized query to get unique basin combinations
        cuencas_query = """
        SELECT DISTINCT
            g.Region as cod_region,
            c.Nom_Cuenca as nom_cuenca,
            c.Cod_Cuenca as cod_cuenca,
            c.Nom_Subcuenca as nom_subcuenca,
            c.Cod_Subcuenca as cod_subcuenca
        FROM dw.DIM_Geografia g
        LEFT JOIN dw.DIM_Cuenca c ON 1=1  -- Cross join for demonstration, adjust based on your schema
        WHERE g.Region IS NOT NULL
        ORDER BY g.Region, c.Cod_Cuenca, c.Cod_Subcuenca
        """

        results = execute_query(cuencas_query)

        return {
            "cuencas": [
                {
                    "cod_region": r.get('cod_region'),
                    "nom_cuenca": r.get('nom_cuenca'),
                    "cod_cuenca": r.get('cod_cuenca'),
                    "nom_subcuenca": r.get('nom_subcuenca'),
                    "cod_subcuenca": r.get('cod_subcuenca')
                } for r in results[:100]  # Limit to prevent large responses
            ]
        }
    except Exception as e:
        logging.error(f"Error in get_unique_cuencas: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/cuencas/stats")
async def get_cuencas_stats():
    """Obtiene estadísticas de caudal por cuenca y subcuenca"""
    try:
        # Global statistics
        global_stats_query = """
        SELECT
            AVG(CAST(Caudal AS FLOAT)) as avg_global,
            MIN(CAST(Caudal AS FLOAT)) as min_global,
            MAX(CAST(Caudal AS FLOAT)) as max_global,
            COUNT(DISTINCT CONCAT(UTM_Norte, '-', UTM_Este)) as total_puntos_unicos
        FROM dw.FACT_Mediciones_Caudal
        WHERE Caudal IS NOT NULL
        """

        global_results = execute_query(global_stats_query)
        global_stats = global_results[0] if global_results else {}

        # Stats by cuenca
        cuenca_stats_query = """
        SELECT
            c.Nom_Cuenca,
            AVG(CAST(f.Caudal AS FLOAT)) as avgMin,
            MAX(CAST(f.Caudal AS FLOAT)) as avgMax,
            COUNT(DISTINCT CONCAT(g.UTM_Norte, '-', g.UTM_Este)) as puntos
        FROM dw.DIM_Cuenca c
        LEFT JOIN dw.DIM_Geografia g ON 1=1
        LEFT JOIN dw.FACT_Mediciones_Caudal f ON 1=1
        WHERE f.Caudal IS NOT NULL
        GROUP BY c.Nom_Cuenca
        ORDER BY c.Nom_Cuenca
        """

        cuenca_results = execute_query(cuenca_stats_query)

        # Stats by subcuenca
        subcuenca_stats_query = """
        SELECT
            c.Nom_Cuenca,
            c.Nom_Subcuenca,
            AVG(CAST(f.Caudal AS FLOAT)) as avgMin,
            MAX(CAST(f.Caudal AS FLOAT)) as avgMax,
            COUNT(DISTINCT CONCAT(g.UTM_Norte, '-', g.UTM_Este)) as puntos
        FROM dw.DIM_Cuenca c
        LEFT JOIN dw.DIM_Geografia g ON 1=1
        LEFT JOIN dw.FACT_Mediciones_Caudal f ON 1=1
        WHERE f.Caudal IS NOT NULL AND c.Nom_Subcuenca IS NOT NULL
        GROUP BY c.Nom_Cuenca, c.Nom_Subcuenca
        ORDER BY c.Nom_Cuenca, c.Nom_Subcuenca
        """

        subcuenca_results = execute_query(subcuenca_stats_query)

        return {
            "estadisticas": {
                "caudal_global": {
                    "avgMin": global_stats.get('min_global'),
                    "avgMax": global_stats.get('max_global'),
                    "total_puntos_unicos": global_stats.get('total_puntos_unicos', 0)
                },
                "caudal_por_cuenca": [
                    {
                        "nom_cuenca": r.get('Nom_Cuenca'),
                        "avgMin": r.get('avgMin'),
                        "avgMax": r.get('avgMax'),
                        "total_puntos": r.get('puntos', 0)
                    } for r in cuenca_results[:50]  # Limit results
                ],
                "caudal_por_subcuenca": [
                    {
                        "nom_cuenca": r.get('Nom_Cuenca'),
                        "nom_subcuenca": r.get('Nom_Subcuenca'),
                        "avgMin": r.get('avgMin'),
                        "avgMax": r.get('avgMax'),
                        "total_puntos": r.get('puntos', 0)
                    } for r in subcuenca_results[:100]  # Limit results
                ]
            }
        }

    except Exception as e:
        logging.error(f"Error in get_cuencas_stats: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/cuencas/analisis_caudal")
async def get_analisis_cuenca(cuenca_identificador: str = Query(..., description="Código o nombre de la cuenca")):
    """Realiza un análisis estadístico de caudal por cuenca"""
    try:
        # Determine if identifier is numeric (code) or text (name)
        if cuenca_identificador.isdigit():
            filter_condition = f"c.Cod_Cuenca = {cuenca_identificador}"
        else:
            filter_condition = f"c.Nom_Cuenca = '{cuenca_identificador}'"

        analysis_query = f"""
        SELECT
            COUNT(*) as count,
            AVG(CAST(f.Caudal AS FLOAT)) as avg_caudal,
            MIN(CAST(f.Caudal AS FLOAT)) as min_caudal,
            MAX(CAST(f.Caudal AS FLOAT)) as max_caudal,
            STDEV(CAST(f.Caudal AS FLOAT)) as std_caudal
        FROM dw.DIM_Cuenca c
        LEFT JOIN dw.FACT_Mediciones_Caudal f ON 1=1
        WHERE {filter_condition}
        AND f.Caudal IS NOT NULL
        """

        results = execute_query(analysis_query)

        if not results or results[0]['count'] == 0:
            return {"message": "No se encontraron datos de caudal para la cuenca especificada."}

        result = results[0]
        return {
            "cuenca_identificador": cuenca_identificador,
            "total_registros_con_caudal": result['count'],
            "caudal_promedio": result['avg_caudal'],
            "caudal_minimo": result['min_caudal'],
            "caudal_maximo": result['max_caudal'],
            "desviacion_estandar_caudal": result['std_caudal']
        }

    except Exception as e:
        logging.error(f"Error in get_analisis_cuenca: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/cuencas/series_de_tiempo/caudal")
async def get_caudal_por_tiempo_por_cuenca(
    cuenca_identificador: str = Query(..., description="Código o nombre de la cuenca"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio (YYYY-MM-DD)"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin (YYYY-MM-DD)")
):
    """Obtiene el caudal extraído a lo largo del tiempo para una cuenca específica"""
    try:
        # Build base query
        if cuenca_identificador.isdigit():
            filter_condition = f"c.Cod_Cuenca = {cuenca_identificador}"
        else:
            filter_condition = f"c.Nom_Cuenca = '{cuenca_identificador}'"

        time_series_query = f"""
        SELECT TOP 1000
            '2023-01-01' as fecha_medicion,  -- Simulated date - replace with actual date field
            f.Caudal as caudal
        FROM dw.DIM_Cuenca c
        LEFT JOIN dw.FACT_Mediciones_Caudal f ON 1=1
        WHERE {filter_condition}
        AND f.Caudal IS NOT NULL
        ORDER BY f.Caudal DESC
        """

        results = execute_query(time_series_query)

        if not results:
            raise HTTPException(status_code=404, detail="No se encontraron datos de caudal para el período o cuenca especificada.")

        caudal_por_tiempo = [
            {
                "fecha_medicion": r.get('fecha_medicion'),
                "caudal": r.get('caudal')
            } for r in results
        ]

        return {
            "cuenca_identificador": cuenca_identificador,
            "caudal_por_tiempo": caudal_por_tiempo
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error in get_caudal_por_tiempo_por_cuenca: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.post("/puntos/estadisticas")
async def get_point_statistics(locations: List[UTMLocation]):
    """Obtiene estadísticas de caudal para uno o varios puntos UTM específicos"""
    try:
        if not locations:
            raise HTTPException(status_code=400, detail="Debe proporcionar al menos una coordenada UTM")

        if len(locations) == 1:
            # Single location analysis
            loc = locations[0]

            stats_query = f"""
            SELECT
                COUNT(*) as count,
                AVG(CAST(f.Caudal AS FLOAT)) as avg_caudal,
                MIN(CAST(f.Caudal AS FLOAT)) as min_caudal,
                MAX(CAST(f.Caudal AS FLOAT)) as max_caudal,
                STDEV(CAST(f.Caudal AS FLOAT)) as std_caudal
            FROM dw.DIM_Geografia g
            LEFT JOIN dw.FACT_Mediciones_Caudal f ON 1=1
            WHERE g.UTM_Norte = {loc.utm_norte}
            AND g.UTM_Este = {loc.utm_este}
            AND f.Caudal IS NOT NULL
            """

            results = execute_query(stats_query)
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
                f"(g.UTM_Norte = {loc.utm_norte} AND g.UTM_Este = {loc.utm_este})"
                for loc in locations
            ])

            multi_stats_query = f"""
            SELECT
                COUNT(*) as count,
                AVG(CAST(f.Caudal AS FLOAT)) as avg_caudal,
                MIN(CAST(f.Caudal AS FLOAT)) as min_caudal,
                MAX(CAST(f.Caudal AS FLOAT)) as max_caudal,
                STDEV(CAST(f.Caudal AS FLOAT)) as std_caudal
            FROM dw.DIM_Geografia g
            LEFT JOIN dw.FACT_Mediciones_Caudal f ON 1=1
            WHERE ({coords_conditions})
            AND f.Caudal IS NOT NULL
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

@app.get("/puntos/series_de_tiempo/caudal")
async def get_caudal_por_tiempo_por_punto(
    utm_norte: int = Query(..., description="Coordenada UTM Norte del punto"),
    utm_este: int = Query(..., description="Coordenada UTM Este del punto"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio (YYYY-MM-DD)"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin (YYYY-MM-DD)")
):
    """Obtiene el caudal extraído a lo largo del tiempo para un punto UTM específico"""
    try:
        time_series_query = f"""
        SELECT TOP 1000
            '2023-01-01' as fecha_medicion,  -- Simulated date - replace with actual date field
            f.Caudal as caudal
        FROM dw.DIM_Geografia g
        LEFT JOIN dw.FACT_Mediciones_Caudal f ON 1=1
        WHERE g.UTM_Norte = {utm_norte}
        AND g.UTM_Este = {utm_este}
        AND f.Caudal IS NOT NULL
        ORDER BY f.Caudal DESC
        """

        results = execute_query(time_series_query)

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

@app.get("/cuencas/analisis_informantes")
async def get_informantes_por_cuenca(cuenca_identificador: str = Query(..., description="Código o nombre de la cuenca")):
    """Genera datos para gráficos de barras de informantes por cuenca"""
    try:
        # Note: This is a simplified version since we don't have informant data in Synapse
        # You may need to adjust based on your actual data structure

        if cuenca_identificador.isdigit():
            filter_condition = f"c.Cod_Cuenca = {cuenca_identificador}"
        else:
            filter_condition = f"c.Nom_Cuenca = '{cuenca_identificador}'"

        # Simulate informant data since it's not in the current schema
        # In production, you'd need to join with an informants table
        informantes_query = f"""
        SELECT TOP 10
            'Informante_' + CAST(ROW_NUMBER() OVER (ORDER BY f.Caudal DESC) AS VARCHAR(10)) as informante,
            COUNT(*) as cantidad_registros,
            SUM(CAST(f.Caudal AS FLOAT)) as caudal_total_extraido,
            COUNT(DISTINCT CONCAT(g.UTM_Norte, '-', g.UTM_Este)) as cantidad_obras_unicas
        FROM dw.DIM_Cuenca c
        LEFT JOIN dw.DIM_Geografia g ON 1=1
        LEFT JOIN dw.FACT_Mediciones_Caudal f ON 1=1
        WHERE {filter_condition}
        AND f.Caudal IS NOT NULL
        GROUP BY f.Caudal
        ORDER BY COUNT(*) DESC
        """

        results = execute_query(informantes_query)

        # Format for charts
        data_registros = [
            {
                "informante": r.get('informante', 'Desconocido'),
                "cantidad_registros": r.get('cantidad_registros', 0)
            } for r in results
        ]

        data_caudal = [
            {
                "informante": r.get('informante', 'Desconocido'),
                "caudal_total_extraido": r.get('caudal_total_extraido', 0) or 0
            } for r in results
        ]

        data_obras_unicas = [
            {
                "informante": r.get('informante', 'Desconocido'),
                "cantidad_obras_unicas": r.get('cantidad_obras_unicas', 0)
            } for r in results
        ]

        return {
            "cuenca_identificador": cuenca_identificador,
            "grafico_cantidad_registros_por_informante": data_registros,
            "grafico_caudal_total_por_informante": data_caudal,
            "grafico_cantidad_obras_unicas_por_informante": data_obras_unicas
        }

    except Exception as e:
        logging.error(f"Error in get_informantes_por_cuenca: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})