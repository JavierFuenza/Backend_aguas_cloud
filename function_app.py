import azure.functions as func
import os
import json
import logging
import pyodbc
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
import math

# Cargar variables de entorno
load_dotenv()

# Crear Azure Function App
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

def get_db_connection():
    """Crear conexión a Synapse"""
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

def execute_query(query: str, params: List = None) -> List[Dict]:
    """Ejecutar query y retornar resultados como lista de diccionarios"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        columns = [column[0] for column in cursor.description]
        results = cursor.fetchall()
        
        return [dict(zip(columns, row)) for row in results]
    
    finally:
        conn.close()

def utm_to_latlon(utm_este: float, utm_norte: float, huso: int = 19) -> tuple:
    """Convertir coordenadas UTM a lat/lon"""
    try:
        # Constantes para zona 19S (Chile)
        central_meridian = -69.0  # Para zona 19
        
        # Aproximación básica para Chile zona 19S
        lat = (utm_norte - 10000000) / 111320
        lon = central_meridian + (utm_este - 500000) / (111320 * math.cos(math.radians(lat)))
        
        return lon, lat
    except:
        return None, None

def json_response(data: dict, status_code: int = 200):
    """Helper para crear respuestas JSON con CORS"""
    return func.HttpResponse(
        json.dumps(data, ensure_ascii=False, default=str),
        status_code=status_code,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
            "Access-Control-Allow-Credentials": "false",
            "Access-Control-Max-Age": "86400"
        }
    )

def parse_query_params(req: func.HttpRequest) -> dict:
    """Helper para parsear parámetros de query"""
    params = {}
    for key, value in req.params.items():
        # Convertir a tipo apropiado
        if value.lower() in ['true', 'false']:
            params[key] = value.lower() == 'true'
        elif value.replace('.', '', 1).isdigit():
            params[key] = float(value) if '.' in value else int(value)
        else:
            params[key] = value
    return params

@app.route(route="health", methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    try:
        results = execute_query("SELECT 1 as test")
        return json_response({
            "status": "healthy",
            "message": "Water Data API is running",
            "database": "connected"
        })
    except Exception as e:
        return json_response({
            "status": "unhealthy", 
            "message": "Water Data API is running but database connection failed",
            "database": "disconnected",
            "error": str(e)
        }, 500)

@app.route(route="test-db", methods=["GET"])
def test_database_connection(req: func.HttpRequest) -> func.HttpResponse:
    try:
        results = execute_query("SELECT COUNT(*) as total FROM dw.FACT_Mediciones_Caudal")
        return json_response({
            "status": "success",
            "message": "Database connection successful",
            "total_records": results[0]['total']
        })
    except Exception as e:
        logging.error(f"Database connection failed: {str(e)}")
        return json_response({
            "status": "error",
            "message": f"Database connection failed: {str(e)}"
        }, 500)

@app.route(route="count", methods=["GET"])
def get_obras_count(req: func.HttpRequest) -> func.HttpResponse:
    """Obtiene el número total de registros en la tabla de mediciones"""
    try:
        results = execute_query("SELECT COUNT(*) as total FROM dw.FACT_Mediciones_Caudal")
        return json_response({"total_records": results[0]['total']})
    except Exception as e:
        logging.error(f"Error in get_obras_count: {e}")
        return json_response({"error": str(e)}, 500)

@app.route(route="puntos/count", methods=["GET"])
def get_puntos_count(req: func.HttpRequest) -> func.HttpResponse:
    """Obtiene el número de puntos únicos (coordenadas únicas) disponibles"""
    try:
        # Parsear parámetros para aplicar mismos filtros que /puntos
        params = parse_query_params(req)
        
        region = params.get('region')
        
        logging.info(f"Contando puntos con parametros: {params}")
        
        # Query para contar puntos únicos (coordenadas únicas)
        count_query = """
        SELECT COUNT(DISTINCT CONCAT(UTM_Norte, '-', UTM_Este)) as total_puntos_unicos
        FROM dw.DIM_Geografia g
        WHERE 1=1
        """
        
        # Aplicar mismos filtros que el endpoint /puntos
        if region:
            count_query += f" AND g.Region = {region}"
        
        logging.info(f"Ejecutando query count: {count_query}")
        results = execute_query(count_query)
        
        total_puntos = results[0]['total_puntos_unicos'] if results else 0
        
        response = {
            "total_puntos_unicos": total_puntos,
            "filtros_aplicados": {
                "region": region
            }
        }
        
        logging.info(f"Total puntos únicos encontrados: {total_puntos}")
        return json_response(response)
        
    except Exception as e:
        logging.error(f"Error in get_puntos_count: {e}", exc_info=True)
        return json_response({"error": str(e)}, 500)

@app.route(route="puntos", methods=["GET", "OPTIONS"]) 
def get_puntos(req: func.HttpRequest) -> func.HttpResponse:
    """Obtiene puntos con sus coordenadas y ubicación administrativa"""

    # Manejar preflight OPTIONS (CORS)
    if req.method == "OPTIONS":
        return func.HttpResponse(
            "",
            status_code=200,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
                "Access-Control-Max-Age": "86400"
            }
        )

    try:
        # Parsear parámetros
        params = parse_query_params(req)
        region = params.get("region")
        limit = min(params.get("limit", 200), 2000)  # límite de seguridad

        logging.info(f"Parametros recibidos en /puntos: region={region}, limit={limit}")

        # Query principal: puntos únicos con ubicación
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

        # Aplicar límite
        puntos_query = f"""
        SELECT TOP ({limit}) *
        FROM ({puntos_query}) AS UniquePoints
        ORDER BY UTM_Norte, UTM_Este
        """

        logging.info(f"Ejecutando query puntos: {puntos_query}")
        puntos = execute_query(puntos_query, query_params)

        logging.info(f"Se obtuvieron {len(puntos)} puntos")

        # Transformar a formato API (lat/lon)
        puntos_out = []
        for geo in puntos:
            lon, lat = utm_to_latlon(
                geo["UTM_Este"],
                geo["UTM_Norte"],
                geo.get("Huso", 19)
            )
            puntos_out.append({
                "lat": lat,
                "lon": lon,
                "utm_norte": geo["UTM_Norte"],
                "utm_este": geo["UTM_Este"],
                "huso": geo.get("Huso", 19),
                "region": geo.get("Region"),
                "provincia": geo.get("Provincia"),
                "comuna": geo.get("Comuna")
            })

        return json_response(puntos_out)

    except Exception as e:
        logging.error(f"Error en get_puntos: {e}", exc_info=True)
        return json_response({"error": str(e)}, 500)

    """Obtiene coordenadas únicas con datos de agua"""
    
    # Manejar preflight OPTIONS
    if req.method == "OPTIONS":
        return func.HttpResponse(
            "",
            status_code=200,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
                "Access-Control-Max-Age": "86400"
            }
        )
    
    try:
        # Parsear parámetros
        params = parse_query_params(req)
        region = params.get('region')
        limit = min(params.get('limit', 120), 1000)  # Limitar máximo a 1000
        
        logging.info(f"Parametros recibidos: region={region}, limit={limit}")
        
        # Construir query principal con coordenadas únicas (sintaxis correcta)
        geo_query = """
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
        
        # Aplicar filtros
        if region:
            geo_query += " AND g.Region = ?"
            query_params.append(region)
        
        geo_query += " ORDER BY g.UTM_Norte, g.UTM_Este"
        
        # Aplicar límite después de obtener los únicos
        geo_query = f"SELECT TOP ({limit}) * FROM ({geo_query}) AS UniqueCoords"
        
        logging.info(f"Ejecutando query geografia: {geo_query}")
        logging.info(f"Con parámetros: {query_params}")
        
        geo_data = execute_query(geo_query, query_params)
        logging.info(f"Geografia obtenida: {len(geo_data)} registros")
        
        # Debug: mostrar algunos registros
        if geo_data:
            logging.info(f"Primer registro: {geo_data[0]}")
            if len(geo_data) > 1:
                logging.info(f"Segundo registro: {geo_data[1]}")
        else:
            logging.warning("No se obtuvieron datos de geografía")
        
        # Obtener estadísticas globales de caudal
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
        logging.info(f"Estadisticas caudal: {caudal_info}")
        
        # Obtener muestra de cuencas
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
        logging.info(f"Cuencas obtenidas: {len(cuenca_data)} registros")
        
        # Combinar datos
        coordenadas = []
        import random
        
        for i, geo in enumerate(geo_data):
            # Seleccionar cuenca de forma rotativa
            cuenca = cuenca_data[i % len(cuenca_data)] if cuenca_data else {}
            
            # Usar estadísticas globales con variación
            caudal_prom = caudal_info.get('caudal_promedio_global', 0)
            variacion = random.uniform(0.7, 1.3)
            caudal_punto = caudal_prom * variacion if caudal_prom else 0
            
            # Convertir coordenadas UTM a lat/lon
            lon, lat = utm_to_latlon(geo['UTM_Este'], geo['UTM_Norte'], geo.get('Huso', 19))
            
            # Determinar tipo de punto
            es_pozo = (geo['UTM_Norte'] % 2 == 0)
            
            coordenadas.append({
                "lat": lat,
                "lon": lon,
                "utm_norte": geo['UTM_Norte'],
                "utm_este": geo['UTM_Este'],
                "huso": geo.get('Huso', 19),
                "region": geo.get('Region'),
                "provincia": geo.get('Provincia'),
                "comuna": geo.get('Comuna'),
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
        
        logging.info(f"Retornando {len(coordenadas)} coordenadas")
        return json_response(coordenadas)
        
    except Exception as e:
        logging.error(f"Error in get_coordenadas_unicas: {e}", exc_info=True)
        return json_response({"error": str(e)}, 500)
    """Obtiene coordenadas únicas con datos de agua"""
    
    # Manejar preflight OPTIONS
    if req.method == "OPTIONS":
        return func.HttpResponse(
            "",
            status_code=200,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
                "Access-Control-Max-Age": "86400"
            }
        )
    
    try:
        # Parsear parámetros
        params = parse_query_params(req)
        region = params.get('region')
        limit = min(params.get('limit', 120), 1000)  # Limitar máximo a 1000
        
        logging.info(f"Parametros recibidos: region={region}, limit={limit}")
        
        # Construir query principal con coordenadas únicas
        geo_query = """
        SELECT TOP (?)
            DISTINCT
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
        
        query_params = [limit]
        
        # Aplicar filtros
        if region:
            geo_query += " AND g.Region = ?"
            query_params.append(region)
        
        geo_query += " ORDER BY g.UTM_Norte, g.UTM_Este"
        
        logging.info(f"Ejecutando query geografia: {geo_query}")
        logging.info(f"Con parámetros: {query_params}")
        
        geo_data = execute_query(geo_query, query_params)
        logging.info(f"Geografia obtenida: {len(geo_data)} registros")
        
        # Debug: mostrar algunos registros
        if geo_data:
            logging.info(f"Primer registro: {geo_data[0]}")
            if len(geo_data) > 1:
                logging.info(f"Segundo registro: {geo_data[1]}")
        else:
            logging.warning("No se obtuvieron datos de geografía")
        
        # Obtener estadísticas globales de caudal
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
        logging.info(f"Estadisticas caudal: {caudal_info}")
        
        # Obtener muestra de cuencas
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
        logging.info(f"Cuencas obtenidas: {len(cuenca_data)} registros")
        
        # Combinar datos
        coordenadas = []
        import random
        
        for i, geo in enumerate(geo_data):
            # Seleccionar cuenca de forma rotativa
            cuenca = cuenca_data[i % len(cuenca_data)] if cuenca_data else {}
            
            # Usar estadísticas globales con variación
            caudal_prom = caudal_info.get('caudal_promedio_global', 0)
            variacion = random.uniform(0.7, 1.3)
            caudal_punto = caudal_prom * variacion if caudal_prom else 0
            
            # Convertir coordenadas UTM a lat/lon
            lon, lat = utm_to_latlon(geo['UTM_Este'], geo['UTM_Norte'], geo.get('Huso', 19))
            
            # Determinar tipo de punto
            es_pozo = (geo['UTM_Norte'] % 2 == 0)
            
            coordenadas.append({
                "lat": lat,
                "lon": lon,
                "utm_norte": geo['UTM_Norte'],
                "utm_este": geo['UTM_Este'],
                "huso": geo.get('Huso', 19),
                "region": geo.get('Region'),
                "provincia": geo.get('Provincia'),
                "comuna": geo.get('Comuna'),
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
        
        logging.info(f"Retornando {len(coordenadas)} coordenadas")
        return json_response(coordenadas)
        
    except Exception as e:
        logging.error(f"Error in get_coordenadas_unicas: {e}", exc_info=True)
        return json_response({"error": str(e)}, 500)
    """Obtiene coordenadas únicas con datos de agua"""
    
    # Manejar preflight OPTIONS directamente aquí
    if req.method == "OPTIONS":
        logging.info("Manejando preflight OPTIONS para /puntos")
        return func.HttpResponse(
            "",
            status_code=200,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
                "Access-Control-Max-Age": "86400"
            }
        )
    
    try:
        # Parsear parámetros
        params = parse_query_params(req)
        
        region = params.get('region')
        limit = params.get('limit', 120)
        
        logging.info(f"Parametros recibidos: {params}")
        
        # Construir query principal unificada
        # Obtener geografía con filtros aplicados
        geo_query = f"""
        SELECT TOP ({limit})
            g.UTM_Norte, 
            g.UTM_Este, 
            g.Huso, 
            g.Region, 
            g.Provincia, 
            g.Comuna
        FROM dw.DIM_Geografia g
        WHERE 1=1
        """
        
        # Aplicar filtros geográficos
        if region:
            geo_query += f" AND g.Region = {region}"
        
        geo_query += " ORDER BY g.UTM_Norte"
        
        logging.info(f"Ejecutando query geografia: {geo_query}")
        geo_data = execute_query(geo_query)
        logging.info(f"Geografia obtenida: {len(geo_data)} registros")
        
        # Debug: mostrar algunos registros
        if geo_data:
            logging.info(f"Primer registro: {geo_data[0]}")
            if len(geo_data) > 1:
                logging.info(f"Segundo registro: {geo_data[1]}")
        else:
            logging.warning("No se obtuvieron datos de geografía")
        
        # Obtener estadísticas globales de caudal una sola vez
        caudal_stats_query = """
        SELECT 
            AVG(CAST(Caudal AS FLOAT)) as caudal_promedio_global,
            MIN(CAST(Caudal AS FLOAT)) as caudal_min_global,
            MAX(CAST(Caudal AS FLOAT)) as caudal_max_global,
            COUNT(*) as total_mediciones
        FROM dw.FACT_Mediciones_Caudal
        WHERE Caudal IS NOT NULL
        """
        
        logging.info("Ejecutando query estadisticas caudal")
        caudal_stats = execute_query(caudal_stats_query)
        caudal_info = caudal_stats[0] if caudal_stats else {}
        logging.info(f"Estadisticas caudal: {caudal_info}")
        
        # Obtener muestra de cuencas para simular relación
        cuenca_query = """
        SELECT TOP (20)
            Cod_Cuenca, 
            Nom_Cuenca, 
            Cod_Subcuenca, 
            Nom_Subcuenca
        FROM dw.DIM_Cuenca
        WHERE Nom_Cuenca IS NOT NULL
        """
        
        logging.info("Ejecutando query cuencas")
        cuenca_data = execute_query(cuenca_query)
        logging.info(f"Cuencas obtenidas: {len(cuenca_data)} registros")
        
        # Combinar datos
        coordenadas = []
        coordenadas_vistas = set()  # Para evitar duplicados
        
        for i, geo in enumerate(geo_data):
            # Crear clave única para coordenadas
            coord_key = f"{geo['UTM_Norte']}-{geo['UTM_Este']}"
            
            # Saltar si ya procesamos estas coordenadas
            if coord_key in coordenadas_vistas:
                continue
            coordenadas_vistas.add(coord_key)
            
            # Simular relación con cuenca (por índice rotativo)
            cuenca = cuenca_data[i % len(cuenca_data)] if cuenca_data else {}
            
            # Usar estadísticas globales como base para este punto
            caudal_prom = caudal_info.get('caudal_promedio_global', 0)
            
            # Aplicar variación aleatoria simple para simular diferencias por punto
            import random
            variacion = random.uniform(0.7, 1.3)  # Variación del 70% al 130%
            caudal_punto = caudal_prom * variacion if caudal_prom else 0
            
            # Convertir coordenadas UTM a lat/lon
            lon, lat = utm_to_latlon(geo['UTM_Este'], geo['UTM_Norte'], geo.get('Huso', 19))
            
            # Simular tipo de punto basado en coordenadas
            es_pozo = (geo['UTM_Norte'] % 2 == 0)  # Criterio simple para simular
            
            coordenadas.append({
                "lat": lat,
                "lon": lon,
                "utm_norte": geo['UTM_Norte'],
                "utm_este": geo['UTM_Este'],
                "nombre_cuenca": cuenca.get('Nom_Cuenca', "Cuenca no disponible"),
                "nombre_subcuenca": cuenca.get('Nom_Subcuenca', "Subcuenca no disponible"),
                "comuna": geo.get('Comuna', "Comuna no disponible"),
                "cod_cuenca": cuenca.get('Cod_Cuenca'),
                "cod_subcuenca": cuenca.get('Cod_Subcuenca'),
                "caudal_promedio": round(caudal_punto, 2) if caudal_punto else None,
                "n_mediciones": random.randint(5, 50),  # Simulado
                "tipoPunto": {
                    "altura": round(random.uniform(1.0, 5.0), 2) if not es_pozo else None,
                    "nivel_freatico": round(random.uniform(10.0, 100.0), 2) if es_pozo else None,
                    "nombreInformante": [f"Informante {i+1}"]
                }
            })
        
        logging.info(f"Retornando {len(coordenadas)} coordenadas")
        return json_response(coordenadas[:limit])
        
    except Exception as e:
        logging.error(f"Error in get_coordenadas_unicas: {e}", exc_info=True)
        return json_response({"error": str(e)}, 500)