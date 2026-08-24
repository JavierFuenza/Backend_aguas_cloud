import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from core.database import execute_query
from utils.helpers import safe_round
from models.schemas import PuntoResponse, PuntoInfoResponse, UTMLocation

router = APIRouter()

MAX_LIMIT = 10000


@router.get("/puntos/count", tags=["Puntos de Medición"])
async def get_puntos_count(
    region: Optional[int] = Query(None),
    cod_cuenca: Optional[int] = Query(None),
    cod_subcuenca: Optional[int] = Query(None),
    cod_subsubcuenca: Optional[int] = Query(None),
    filtro_null_subcuenca: Optional[bool] = Query(
        None, description="Si es True, filtra por subcuenca nula"
    ),
    caudal_minimo: Optional[float] = Query(None),
    caudal_maximo: Optional[float] = Query(None),
    pozo: Optional[bool] = Query(None, description="Filtra por pozo subterráneo"),
    codigo_obra: Optional[str] = Query(None, description="Buscar por código de obra"),
    shac: Optional[int] = Query(None, description="Filtrar por código de SHAC"),
    apr: Optional[bool] = Query(None, description="Filtrar por Agua Potable Rural"),
    id_junta: Optional[float] = Query(None, description="Filtrar por ID de Junta"),
):
    """Obtiene el número de puntos únicos desde Puntos_Mapa con filtros"""
    try:
        logging.info("Contando puntos con filtros")

        # dw.Puntos_Mapa trae filas repetidas para un mismo punto (ver /puntos),
        # así que se cuentan pares de coordenadas distintos y no filas.
        # Mismo filtro de coordenadas que /puntos para que ambos totales coincidan.
        count_query = """
        SELECT DISTINCT UTM_Norte, UTM_Este
        FROM dw.Puntos_Mapa
        WHERE UTM_Norte IS NOT NULL
          AND UTM_Este IS NOT NULL
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
            # Búsqueda exacta: con LIKE '%x%' "OB-0202-1" traía también
            # OB-0202-1x, OB-0202-1xx, etc.
            count_query += " AND codigo = ?"
            query_params.append(codigo_obra.strip())

        if shac is not None:
            count_query += " AND COD_SECTOR_SHA = ?"
            query_params.append(shac)

        if apr is not None:
            count_query += " AND APR = ?"
            query_params.append(1 if apr else 0)

        if id_junta is not None:
            count_query += " AND ID_JUNTA = ?"
            query_params.append(id_junta)

        count_query = f"SELECT COUNT(*) as total_puntos_unicos FROM ({count_query}) AS puntos_unicos"

        logging.info(f"Ejecutando query count: {count_query}")
        results = await execute_query(count_query, query_params)

        total_puntos = results[0]["total_puntos_unicos"] if results else 0

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
                "id_junta": id_junta,
            },
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
    description="Obtiene la lista de puntos de medición con coordenadas UTM e indicador de pozo subterráneo. Soporta filtros por región, cuenca, subcuenca, caudal y más.",
)
async def get_puntos(
    region: Optional[int] = Query(
        None, description="Código de región (ej: 15 para Arica y Parinacota)"
    ),
    cod_cuenca: Optional[int] = Query(None, description="Código de cuenca"),
    cod_subcuenca: Optional[int] = Query(None, description="Código de subcuenca"),
    cod_subsubcuenca: Optional[int] = Query(None, description="Código de subsubcuenca"),
    filtro_null_subcuenca: Optional[bool] = Query(
        None,
        description="Si es True, filtra por subcuenca nula. Ignora 'cod_subcuenca' si es True.",
    ),
    caudal_minimo: Optional[float] = Query(
        None, description="Caudal promedio mínimo (l/s)"
    ),
    caudal_maximo: Optional[float] = Query(
        None, description="Caudal promedio máximo (l/s)"
    ),
    limit: Optional[int] = Query(120, description="Número máximo de puntos a retornar"),
    pozo: Optional[bool] = Query(None, description="Filtra por pozo subterráneo"),
    codigo_obra: Optional[str] = Query(None, description="Buscar por código de obra"),
    shac: Optional[int] = Query(None, description="Filtrar por código de SHAC"),
    apr: Optional[bool] = Query(None, description="Filtrar por Agua Potable Rural"),
    id_junta: Optional[float] = Query(None, description="Filtrar por ID de Junta"),
):
    """Obtiene puntos desde la tabla pre-agregada Puntos_Mapa con filtros"""
    try:
        logging.info(
            f"Parametros recibidos en /puntos: region={region}, cod_cuenca={cod_cuenca}, cod_subcuenca={cod_subcuenca}"
        )

        puntos_query = """
        SELECT
            UTM_Norte,
            UTM_Este,
            Huso,
            es_pozo_subterraneo,
            Cod_Subsubcuenca,
            SECTOR_SHA,
            APR,
            ID_JUNTA,
            -- dw.Puntos_Mapa está agregada por (punto, CANAL_TRANSMISION), no por
            -- punto: el 21% de los puntos tiene más de una fila y el mapa dibujaba
            -- un marcador por canal. Los campos que devuelve este endpoint son
            -- idénticos entre canales, así que basta con quedarse con una fila;
            -- se elige la del canal con más mediciones para que sea determinista.
            ROW_NUMBER() OVER (
                PARTITION BY UTM_Norte, UTM_Este
                ORDER BY n_mediciones DESC, CANAL_TRANSMISION
            ) AS rn
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
            # Búsqueda exacta: con LIKE '%x%' "OB-0202-1" traía 15 puntos
            # (OB-0202-1, -10, -100, -304...). La DGA pide coincidencia exacta.
            puntos_query += " AND codigo = ?"
            query_params.append(codigo_obra.strip())

        if shac is not None:
            puntos_query += " AND COD_SECTOR_SHA = ?"
            query_params.append(shac)

        if apr is not None:
            puntos_query += " AND APR = ?"
            query_params.append(1 if apr else 0)

        if id_junta is not None:
            puntos_query += " AND ID_JUNTA = ?"
            query_params.append(id_junta)

        # Se queda una sola fila por par de coordenadas (rn = 1) y recién ahí
        # se aplica el límite, para no gastar cupo en filas repetidas.
        columnas = (
            "UTM_Norte, UTM_Este, Huso, es_pozo_subterraneo, "
            "Cod_Subsubcuenca, SECTOR_SHA, APR, ID_JUNTA"
        )

        # Apply limit (clamped to safe range; int coerced by FastAPI)
        top = ""
        if limit is not None:
            safe_limit = max(1, min(int(limit), MAX_LIMIT))
            top = f"TOP {safe_limit} "

        puntos_query = (
            f"SELECT {top}{columnas} "
            f"FROM ({puntos_query}) AS puntos_numerados "
            f"WHERE rn = 1"
        )

        logging.info(f"Ejecutando query desde Puntos_Mapa: {puntos_query}")
        puntos = await execute_query(puntos_query, query_params)

        logging.info(f"Se obtuvieron {len(puntos)} puntos desde Puntos_Mapa")

        # Build response
        puntos_out = []

        for punto in puntos:
            puntos_out.append(
                {
                    "utm_norte": punto["UTM_Norte"],
                    "utm_este": punto["UTM_Este"],
                    "huso": punto["Huso"],
                    "es_pozo_subterraneo": bool(punto.get("es_pozo_subterraneo", 0)),
                    "cod_subsubcuenca": punto.get("Cod_Subsubcuenca"),
                    "sector_sha": punto.get("SECTOR_SHA"),
                    "apr": (
                        bool(punto.get("APR", 0))
                        if punto.get("APR") is not None
                        else None
                    ),
                    "id_junta": punto.get("ID_JUNTA"),
                }
            )

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
    description="Obtiene información detallada de un punto de medición específico incluyendo cuenca, subcuenca y estadísticas de caudal. Requiere coordenadas UTM Norte y Este.",
)
async def get_punto_info(
    utm_norte: int = Query(
        ..., description="Coordenada UTM Norte en metros", example=6300000
    ),
    utm_este: int = Query(
        ..., description="Coordenada UTM Este en metros", example=350000
    ),
):
    """Obtiene información detallada de un punto específico incluyendo cuenca y caudal"""
    try:
        logging.info(
            f"Obteniendo info detallada para punto: UTM_Norte={utm_norte}, UTM_Este={utm_este}"
        )

        # Single query against pre-aggregated Puntos_Mapa — avoids full scan of Mediciones_full
        punto_query = """
        SELECT
            UTM_Norte,
            UTM_Este,
            Huso,
            es_pozo_subterraneo,
            codigo,
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
            CANAL_TRANSMISION,
            caudal_promedio,
            n_mediciones
        FROM dw.Puntos_Mapa
        WHERE UTM_Norte = ?
          AND UTM_Este = ?
        """

        punto_result = await execute_query(punto_query, [utm_norte, utm_este])

        if not punto_result:
            raise HTTPException(status_code=404, detail="Punto no encontrado")

        # dw.Puntos_Mapa está agregada por (punto, CANAL_TRANSMISION), no por punto:
        # el 21% de los puntos tiene más de una fila (hasta 8). Con TOP 1 se mostraba
        # el caudal de un canal cualquiera — ej. OB-0202-591 informaba 4.701 mediciones
        # cuando el punto tiene 45.511 repartidas en dos canales.
        # Los campos descriptivos (cuenca, junta, APR…) son idénticos entre filas;
        # solo hay que recomponer las estadísticas de caudal y la lista de canales.
        p = max(punto_result, key=lambda r: r.get("n_mediciones") or 0)

        total_mediciones = sum((r.get("n_mediciones") or 0) for r in punto_result)

        # Promedio ponderado por número de mediciones de cada canal
        caudal_promedio = None
        if total_mediciones > 0:
            suma = sum(
                (r.get("caudal_promedio") or 0) * (r.get("n_mediciones") or 0)
                for r in punto_result
                if r.get("caudal_promedio") is not None
            )
            caudal_promedio = suma / total_mediciones

        canales = sorted(
            {
                r.get("CANAL_TRANSMISION")
                for r in punto_result
                if r.get("CANAL_TRANSMISION") is not None
            }
        )

        # Build detailed response
        response = {
            "utm_norte": utm_norte,
            "utm_este": utm_este,
            "huso": p.get("Huso"),
            "es_pozo_subterraneo": bool(p.get("es_pozo_subterraneo", 0)),
            "codigo": p.get("codigo"),
            "cod_cuenca": p.get("Cod_Cuenca"),
            "cod_subcuenca": p.get("Cod_Subcuenca"),
            "cod_subsubcuenca": p.get("Cod_Subsubcuenca"),
            "nombre_cuenca": p.get("Nom_Cuenca"),
            "nombre_subcuenca": p.get("Nom_Subcuenca"),
            "nombre_subsubcuenca": p.get("Nom_Subsubcuenca"),
            "caudal_promedio": safe_round(caudal_promedio),
            "n_mediciones": total_mediciones,
            "sector_sha": p.get("SECTOR_SHA"),
            "apr": bool(p.get("APR", 0)) if p.get("APR") is not None else None,
            "id_junta": p.get("ID_JUNTA"),
            "parte_junta": (
                bool(p.get("PARTE_JUNTA", 0))
                if p.get("PARTE_JUNTA") is not None
                else None
            ),
            "representa_junta": (
                bool(p.get("REPRESENTA_JUNTA", 0))
                if p.get("REPRESENTA_JUNTA") is not None
                else None
            ),
            # Se mantiene el canal con más mediciones por compatibilidad, y se
            # agrega la lista completa: un punto puede transmitir por varias vías.
            "canal_transmision": p.get("CANAL_TRANSMISION"),
            "canales_transmision": canales,
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
            raise HTTPException(
                status_code=400, detail="Debe proporcionar al menos una coordenada UTM"
            )

        if len(locations) == 1:
            # Single location analysis
            loc = locations[0]

            # Caudal statistics
            caudal_stats_query = """
            SELECT
                COUNT(*) as count,
                AVG(CAST(CAUDAL AS FLOAT)) as avg_val,
                MIN(CAST(CAUDAL AS FLOAT)) as min_val,
                MAX(CAST(CAUDAL AS FLOAT)) as max_val,
                STDEV(CAST(CAUDAL AS FLOAT)) as std_val,
                MIN(FECHA_MEDICION) as primera_fecha,
                MAX(FECHA_MEDICION) as ultima_fecha
            FROM dw.Series_Tiempo
            WHERE UTM_NORTE = ? AND UTM_ESTE = ? AND CAUDAL IS NOT NULL
            """
            caudal_result = await execute_query(
                caudal_stats_query, [loc.utm_norte, loc.utm_este]
            )
            caudal_stats = caudal_result[0] if caudal_result else {}

            # Altura Limnimetrica statistics
            altura_stats_query = """
            SELECT
                COUNT(*) as count,
                AVG(CAST(ALTURA_LIMNIMETRICA AS FLOAT)) as avg_val,
                MIN(CAST(ALTURA_LIMNIMETRICA AS FLOAT)) as min_val,
                MAX(CAST(ALTURA_LIMNIMETRICA AS FLOAT)) as max_val,
                STDEV(CAST(ALTURA_LIMNIMETRICA AS FLOAT)) as std_val,
                MIN(FECHA_MEDICION) as primera_fecha,
                MAX(FECHA_MEDICION) as ultima_fecha
            FROM dw.Series_Tiempo
            WHERE UTM_NORTE = ? AND UTM_ESTE = ? AND ALTURA_LIMNIMETRICA IS NOT NULL
            """
            altura_result = await execute_query(
                altura_stats_query, [loc.utm_norte, loc.utm_este]
            )
            altura_stats = altura_result[0] if altura_result else {}

            # Nivel Freatico statistics
            nivel_stats_query = """
            SELECT
                COUNT(*) as count,
                AVG(CAST(NIVEL_FREATICO AS FLOAT)) as avg_val,
                MIN(CAST(NIVEL_FREATICO AS FLOAT)) as min_val,
                MAX(CAST(NIVEL_FREATICO AS FLOAT)) as max_val,
                STDEV(CAST(NIVEL_FREATICO AS FLOAT)) as std_val,
                MIN(FECHA_MEDICION) as primera_fecha,
                MAX(FECHA_MEDICION) as ultima_fecha
            FROM dw.Series_Tiempo
            WHERE UTM_NORTE = ? AND UTM_ESTE = ? AND NIVEL_FREATICO IS NOT NULL
            """
            nivel_result = await execute_query(
                nivel_stats_query, [loc.utm_norte, loc.utm_este]
            )
            nivel_stats = nivel_result[0] if nivel_result else {}

            response = {"utm_norte": loc.utm_norte, "utm_este": loc.utm_este}

            if caudal_stats.get("count", 0) > 0:
                # safe_round distingue 0 de None. Con `if valor else None` una obra
                # con caudal cero (ej. OB-0202-304) devolvía null y el visualizador
                # mostraba '-' en vez de 0.
                response["caudal"] = {
                    "total_registros": caudal_stats.get("count"),
                    "promedio": safe_round(caudal_stats.get("avg_val")),
                    "minimo": safe_round(caudal_stats.get("min_val")),
                    "maximo": safe_round(caudal_stats.get("max_val")),
                    "desviacion_estandar": safe_round(caudal_stats.get("std_val")),
                    "primera_fecha": (
                        str(caudal_stats.get("primera_fecha"))
                        if caudal_stats.get("primera_fecha")
                        else None
                    ),
                    "ultima_fecha": (
                        str(caudal_stats.get("ultima_fecha"))
                        if caudal_stats.get("ultima_fecha")
                        else None
                    ),
                }

            if altura_stats.get("count", 0) > 0:
                response["altura_limnimetrica"] = {
                    "total_registros": altura_stats.get("count"),
                    "promedio": safe_round(altura_stats.get("avg_val")),
                    "minimo": safe_round(altura_stats.get("min_val")),
                    "maximo": safe_round(altura_stats.get("max_val")),
                    "desviacion_estandar": safe_round(altura_stats.get("std_val")),
                    "primera_fecha": (
                        str(altura_stats.get("primera_fecha"))
                        if altura_stats.get("primera_fecha")
                        else None
                    ),
                    "ultima_fecha": (
                        str(altura_stats.get("ultima_fecha"))
                        if altura_stats.get("ultima_fecha")
                        else None
                    ),
                }

            if nivel_stats.get("count", 0) > 0:
                response["nivel_freatico"] = {
                    "total_registros": nivel_stats.get("count"),
                    "promedio": safe_round(nivel_stats.get("avg_val")),
                    "minimo": safe_round(nivel_stats.get("min_val")),
                    "maximo": safe_round(nivel_stats.get("max_val")),
                    "desviacion_estandar": safe_round(nivel_stats.get("std_val")),
                    "primera_fecha": (
                        str(nivel_stats.get("primera_fecha"))
                        if nivel_stats.get("primera_fecha")
                        else None
                    ),
                    "ultima_fecha": (
                        str(nivel_stats.get("ultima_fecha"))
                        if nivel_stats.get("ultima_fecha")
                        else None
                    ),
                }

            return [response]
        else:
            # Multiple locations analysis
            coords_conditions = " OR ".join(
                ["(UTM_Norte = ? AND UTM_Este = ?)" for _ in locations]
            )
            coords_params = []
            for loc in locations:
                coords_params.extend([loc.utm_norte, loc.utm_este])

            multi_stats_query = f"""
            SELECT
                COUNT(*) as count,
                AVG(CAST(CAUDAL AS FLOAT)) as avg_caudal,
                MIN(CAST(CAUDAL AS FLOAT)) as min_caudal,
                MAX(CAST(CAUDAL AS FLOAT)) as max_caudal,
                STDEV(CAST(CAUDAL AS FLOAT)) as std_caudal
            FROM dw.Series_Tiempo
            WHERE ({coords_conditions})
            AND CAUDAL IS NOT NULL
            """

            results = await execute_query(multi_stats_query, coords_params)
            result = results[0] if results else {}

            return [
                {
                    "puntos_consultados": len(locations),
                    "total_registros_con_caudal": result.get("count", 0),
                    "caudal_promedio": safe_round(result.get("avg_caudal")),
                    "caudal_minimo": safe_round(result.get("min_caudal")),
                    "caudal_maximo": safe_round(result.get("max_caudal")),
                    "desviacion_estandar_caudal": safe_round(result.get("std_caudal")),
                }
            ]

    except Exception as e:
        logging.error(f"Error in get_point_statistics: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})
