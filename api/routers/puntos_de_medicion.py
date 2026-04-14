import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from core.database import execute_query
from utils.helpers import safe_round, build_full_name
from models.schemas import PuntoResponse, PuntoInfoResponse, UTMLocation

router = APIRouter()

@router.get("/puntos/count", tags=["Puntos de Medición"])
async def get_puntos_count(
    region: Optional[int] = Query(None),
    cod_cuenca: Optional[int] = Query(None),
    cod_subcuenca: Optional[int] = Query(None),
    cod_subsubcuenca: Optional[int] = Query(None),
    filtro_null_subcuenca: Optional[bool] = Query(None, description="Si es True, filtra por subcuenca nula"),
    caudal_minimo: Optional[float] = Query(None),
    caudal_maximo: Optional[float] = Query(None),
    pozo: Optional[bool] = Query(None, description="Filtra por pozo subterráneo"),
    codigo_obra: Optional[str] = Query(None, description="Buscar por código de obra"),
    shac: Optional[int] = Query(None, description="Filtrar por código de SHAC"),
    apr: Optional[bool] = Query(None, description="Filtrar por Agua Potable Rural"),
    id_junta: Optional[float] = Query(None, description="Filtrar por ID de Junta")
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
            
        if cod_subsubcuenca is not None:
            count_query += " AND Cod_Subsubcuenca = ?"
            query_params.append(cod_subsubcuenca)

        if caudal_minimo is not None:
            count_query += " AND caudal_promedio >= ?"
            query_params.append(caudal_minimo)

        if caudal_maximo is not None:
            count_query += " AND caudal_promedio <= ?"
            query_params.append(caudal_maximo)

        if pozo is not None:
            count_query += " AND es_pozo_subterraneo = ?"
            query_params.append(1 if pozo else 0)

        if codigo_obra is not None:
            count_query += " AND codigo LIKE ?"
            query_params.append(f"%{codigo_obra}%")
            
        if shac is not None:
            count_query += " AND COD_SECTOR_SHA = ?"
            query_params.append(shac)
            
        if apr is not None:
            count_query += " AND APR = ?"
            query_params.append(1 if apr else 0)
            
        if id_junta is not None:
            count_query += " AND ID_JUNTA = ?"
            query_params.append(id_junta)

        logging.info(f"Ejecutando query count: {count_query}")
        results = execute_query(count_query, query_params)

        total_puntos = results[0]['total_puntos_unicos'] if results else 0

        response = {
            "total_puntos_unicos": total_puntos,
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
        }

        logging.info(f"Total puntos únicos encontrados: {total_puntos}")
        return response

    except Exception as e:
        logging.error(f"Error in get_puntos_count: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get(
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
    cod_subsubcuenca: Optional[int] = Query(None, description="Código de subsubcuenca"),
    filtro_null_subcuenca: Optional[bool] = Query(None, description="Si es True, filtra por subcuenca nula. Ignora 'cod_subcuenca' si es True."),
    caudal_minimo: Optional[float] = Query(None, description="Caudal promedio mínimo (l/s)"),
    caudal_maximo: Optional[float] = Query(None, description="Caudal promedio máximo (l/s)"),
    limit: Optional[int] = Query(120, description="Número máximo de puntos a retornar"),
    pozo: Optional[bool] = Query(None, description="Filtra por pozo subterráneo"),
    codigo_obra: Optional[str] = Query(None, description="Buscar por código de obra"),
    shac: Optional[int] = Query(None, description="Filtrar por código de SHAC"),
    apr: Optional[bool] = Query(None, description="Filtrar por Agua Potable Rural"),
    id_junta: Optional[float] = Query(None, description="Filtrar por ID de Junta")
):
    """Obtiene puntos desde la tabla pre-agregada Puntos_Mapa con filtros"""
    try:
        logging.info(f"Parametros recibidos en /puntos: region={region}, cod_cuenca={cod_cuenca}, cod_subcuenca={cod_subcuenca}")

        puntos_query = """
        SELECT
            UTM_Norte,
            UTM_Este,
            Huso,
            es_pozo_subterraneo,
            Cod_Subsubcuenca,
            SECTOR_SHA,
            APR,
            ID_JUNTA
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
            
        if cod_subsubcuenca is not None:
            puntos_query += " AND Cod_Subsubcuenca = ?"
            query_params.append(cod_subsubcuenca)

        if caudal_minimo is not None:
            puntos_query += " AND caudal_promedio >= ?"
            query_params.append(caudal_minimo)

        if caudal_maximo is not None:
            puntos_query += " AND caudal_promedio <= ?"
            query_params.append(caudal_maximo)

        if pozo is not None:
            puntos_query += " AND es_pozo_subterraneo = ?"
            query_params.append(1 if pozo else 0)

        if codigo_obra is not None:
            puntos_query += " AND codigo LIKE ?"
            query_params.append(f"%{codigo_obra}%")
            
        if shac is not None:
            puntos_query += " AND COD_SECTOR_SHA = ?"
            query_params.append(shac)
            
        if apr is not None:
            puntos_query += " AND APR = ?"
            query_params.append(1 if apr else 0)
            
        if id_junta is not None:
            puntos_query += " AND ID_JUNTA = ?"
            query_params.append(id_junta)

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
                "es_pozo_subterraneo": bool(punto.get("es_pozo_subterraneo", 0)),
                "cod_subsubcuenca": punto.get("Cod_Subsubcuenca"),
                "sector_sha": punto.get("SECTOR_SHA"),
                "apr": bool(punto.get("APR", 0)) if punto.get("APR") is not None else None,
                "id_junta": punto.get("ID_JUNTA")
            })

        logging.info(f"Retornando {len(puntos_out)} puntos")
        return puntos_out

    except Exception as e:
        logging.error(f"Error en get_puntos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})

@router.get(
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
            es_pozo_subterraneo,
            codigo
        FROM dw.Puntos_Mapa
        WHERE UTM_Norte = ?
          AND UTM_Este = ?
        """

        punto_result = execute_query(punto_query, [utm_norte, utm_este])

        if not punto_result:
            raise HTTPException(status_code=404, detail="Punto no encontrado")

        punto = punto_result[0]

        # Get cuenca and last dynamic info based on UTM coordinates
        cuenca_query = """
        SELECT TOP 1
            Cod_Cuenca,
            Nom_Cuenca,
            Cod_Subcuenca,
            Nom_Subcuenca,
            Cod_Subsubcuenca,
            Nom_Subsubcuenca,
            SECTOR_SHA, 
            APR, 
            ID_JUNTA, 
            PARTE_JUNTA, 
            REPRESENTA_JUNTA, 
            CANAL_TRANSMISION
        FROM dw.Mediciones_full
        WHERE UTM_Norte = ?
          AND UTM_Este = ?
        ORDER BY FECHA_MEDICION DESC
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
            "codigo": punto.get('codigo'),
            "cod_cuenca": cuenca.get('Cod_Cuenca'),
            "cod_subcuenca": cuenca.get('Cod_Subcuenca'),
            "cod_subsubcuenca": cuenca.get('Cod_Subsubcuenca'),
            "nombre_cuenca": cuenca.get('Nom_Cuenca'),
            "nombre_subcuenca": cuenca.get('Nom_Subcuenca'),
            "nombre_subsubcuenca": cuenca.get('Nom_Subsubcuenca'),
            "caudal_promedio": safe_round(caudal_stats.get('caudal_promedio')),
            "n_mediciones": caudal_stats.get('n_mediciones', 0),
            "sector_sha": cuenca.get('SECTOR_SHA'),
            "apr": bool(cuenca.get('APR', 0)) if cuenca.get('APR') is not None else None,
            "id_junta": cuenca.get('ID_JUNTA'),
            "parte_junta": bool(cuenca.get('PARTE_JUNTA', 0)) if cuenca.get('PARTE_JUNTA') is not None else None,
            "representa_junta": bool(cuenca.get('REPRESENTA_JUNTA', 0)) if cuenca.get('REPRESENTA_JUNTA') is not None else None,
            "canal_transmision": cuenca.get('CANAL_TRANSMISION')
        }

        logging.info(f"Info detallada obtenida para punto {utm_norte}/{utm_este}")
        return response

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error en get_punto_info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.post("/puntos/estadisticas", tags=["Puntos de Medición"])
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

