import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from core.database import execute_query
from core.cache_manager import CACHE_TTL_STATIC
from utils.helpers import safe_round

router = APIRouter()


@router.get(
    "/cuencas",
    tags=["Cuencas Hidrográficas"],
    summary="Listado de cuencas hidrográficas",
    description="Obtiene el listado completo de cuencas, subcuencas y subsubcuencas hidrográficas con sus códigos, nombres y región asociada.",
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

        results = await execute_query(cuencas_query, ttl=CACHE_TTL_STATIC)

        return {
            "cuencas": [
                {
                    "cod_cuenca": r.get("cod_cuenca"),
                    "nom_cuenca": r.get("nom_cuenca"),
                    "cod_region": r.get("cod_region"),
                    "cod_subcuenca": r.get("cod_subcuenca"),
                    "nom_subcuenca": r.get("nom_subcuenca"),
                    "cod_subsubcuenca": r.get("cod_subsubcuenca"),
                    "nom_subsubcuenca": r.get("nom_subsubcuenca"),
                }
                for r in results
            ]
        }
    except Exception as e:
        logging.error(f"Error in get_unique_cuencas: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get(
    "/shacs",
    tags=["Cuencas Hidrográficas"],
    summary="Listado de sectores SHAC",
    description="Obtiene el listado de Sectores Hidrogeológicos de Aprovechamiento Común (SHAC) con total de puntos.",
)
async def get_shacs(
    region: Optional[int] = Query(None, description="Código de región"),
    cod_cuenca: Optional[int] = Query(None, description="Código de cuenca"),
    cod_subcuenca: Optional[int] = Query(None, description="Código de subcuenca"),
):
    """Obtiene lista de SHACs disponibles con conteo de puntos, opcionalmente filtrados por región/cuenca/subcuenca"""
    filters = ["COD_SECTOR_SHA IS NOT NULL"]
    params = []
    if region is not None:
        filters.append("Region = ?")
        params.append(region)
    if cod_cuenca is not None:
        filters.append("Cod_Cuenca = ?")
        params.append(cod_cuenca)
    if cod_subcuenca is not None:
        filters.append("Cod_Subcuenca = ?")
        params.append(cod_subcuenca)

    where = " AND ".join(filters)
    query = f"""
    SELECT
        COD_SECTOR_SHA AS cod_sector_sha,
        SECTOR_SHA AS sector_sha,
        COUNT(*) AS total_puntos
    FROM dw.Puntos_Mapa
    WHERE {where}
    GROUP BY COD_SECTOR_SHA, SECTOR_SHA
    ORDER BY COD_SECTOR_SHA
    """
    try:
        results = await execute_query(query, params=params if params else None)
        return {
            "shacs": [
                {
                    "cod_sector_sha": r.get("cod_sector_sha"),
                    "sector_sha": r.get("sector_sha"),
                    "total_puntos": r.get("total_puntos", 0),
                }
                for r in results
            ]
        }
    except Exception as e:
        logging.error(f"Error in get_shacs: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get(
    "/juntas",
    tags=["Cuencas Hidrográficas"],
    summary="Listado de Juntas de Vigilancia",
    description="Obtiene el listado de Juntas de Vigilancia con total de puntos asociados.",
)
async def get_juntas(
    region: Optional[int] = Query(None, description="Código de región"),
    cod_cuenca: Optional[int] = Query(None, description="Código de cuenca"),
    cod_subcuenca: Optional[int] = Query(None, description="Código de subcuenca"),
):
    """Obtiene lista de Juntas disponibles con conteo de puntos, opcionalmente filtradas por región/cuenca/subcuenca"""
    filters = ["ID_JUNTA IS NOT NULL"]
    params = []
    if region is not None:
        filters.append("Region = ?")
        params.append(region)
    if cod_cuenca is not None:
        filters.append("Cod_Cuenca = ?")
        params.append(cod_cuenca)
    if cod_subcuenca is not None:
        filters.append("Cod_Subcuenca = ?")
        params.append(cod_subcuenca)

    where = " AND ".join(filters)
    query = f"""
    SELECT
        ID_JUNTA AS id_junta,
        COUNT(*) AS total_puntos
    FROM dw.Puntos_Mapa
    WHERE {where}
    GROUP BY ID_JUNTA
    ORDER BY ID_JUNTA
    """
    try:
        results = await execute_query(query, params=params if params else None)
        return {
            "juntas": [
                {
                    "id_junta": r.get("id_junta"),
                    "total_puntos": r.get("total_puntos", 0),
                }
                for r in results
            ]
        }
    except Exception as e:
        logging.error(f"Error in get_juntas: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get(
    "/filtrosreactivos",
    tags=["Cuencas Hidrográficas"],
    summary="Estadísticas de caudal para filtros reactivos",
    description="Obtiene estadísticas de caudal mínimo y máximo agregadas globalmente, por cuenca y por subcuenca. Usado para configurar filtros reactivos en el frontend.",
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
        results = await execute_query(stats_query)

        # Separate results by nivel
        global_stats = {}
        cuenca_stats = []
        subcuenca_stats = []

        for r in results:
            nivel = r.get("nivel")
            if nivel == "global":
                global_stats = {
                    "avgMin": safe_round(r.get("avgMin")),
                    "avgMax": safe_round(r.get("avgMax")),
                    "total_puntos_unicos": r.get("total_puntos", 0),
                }
            elif nivel == "cuenca":
                cuenca_stats.append(
                    {
                        "nom_cuenca": r.get("nom_cuenca"),
                        "avgMin": safe_round(r.get("avgMin")),
                        "avgMax": safe_round(r.get("avgMax")),
                        "total_puntos": r.get("total_puntos", 0),
                    }
                )
            elif nivel == "subcuenca":
                subcuenca_stats.append(
                    {
                        "nom_cuenca": r.get("nom_cuenca"),
                        "nom_subcuenca": r.get("nom_subcuenca"),
                        "avgMin": safe_round(r.get("avgMin")),
                        "avgMax": safe_round(r.get("avgMax")),
                        "total_puntos": r.get("total_puntos", 0),
                    }
                )

        return {
            "estadisticas": {
                "caudal_global": global_stats,
                "caudal_por_cuenca": cuenca_stats,
                "caudal_por_subcuenca": subcuenca_stats,
            }
        }
    except Exception as e:
        logging.error(f"Error in get_filtros_reactivos: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get(
    "/shacs/stats",
    tags=["Cuencas Hidrográficas"],
    summary="Estadísticas de caudal de un sector SHAC",
    description=(
        "Estadísticas agregadas de caudal para un Sector Hidrogeológico de "
        "Aprovechamiento Común. Equivale a /cuencas/stats pero para SHAC."
    ),
)
async def get_shac_stats(
    shac: int = Query(..., description="Código de sector SHAC (COD_SECTOR_SHA)"),
):
    """Estadísticas de un SHAC, calculadas desde dw.Puntos_Mapa.

    dw.Cuenca_Stats está agregada por la jerarquía de cuencas y no conoce el
    SHAC, así que no sirve de fuente. Puntos_Mapa sí trae COD_SECTOR_SHA.

    Igual que en /cuencas/stats_por_tipo, la subconsulta interna colapsa por
    coordenada antes de agregar: Puntos_Mapa tiene una fila por (punto, canal de
    transmisión), y sin ese paso el número de obras y el caudal total se inflan.

    Una diferencia con /cuencas/stats: la desviación estándar que devuelve es la
    dispersión del caudal promedio ENTRE obras del sector, no la dispersión de
    las mediciones individuales. Esa segunda no se puede reconstruir desde una
    tabla ya agregada; habría que ir a las mediciones crudas.
    """
    try:
        query = """
        SELECT
            COUNT(*) AS obras_con_datos,
            SUM(n_mediciones) AS total_mediciones,
            SUM(caudal_promedio) AS caudal_total,
            SUM(caudal_promedio * n_mediciones) / NULLIF(SUM(n_mediciones), 0)
                AS caudal_promedio,
            MIN(caudal_minimo) AS caudal_minimo,
            MAX(caudal_maximo) AS caudal_maximo,
            STDEV(caudal_promedio) AS caudal_desviacion_estandar
        FROM (
            SELECT
                UTM_Norte,
                UTM_Este,
                SUM(n_mediciones) AS n_mediciones,
                SUM(caudal_promedio * n_mediciones) / NULLIF(SUM(n_mediciones), 0)
                    AS caudal_promedio,
                MIN(caudal_minimo) AS caudal_minimo,
                MAX(caudal_maximo) AS caudal_maximo
            FROM dw.Puntos_Mapa
            WHERE COD_SECTOR_SHA = ?
              AND caudal_promedio IS NOT NULL
              AND n_mediciones > 0
            GROUP BY UTM_Norte, UTM_Este
        ) AS obras
        """
        results = await execute_query(query, [shac])
        fila = results[0] if results else {}

        return {
            "shac": shac,
            "estadisticas": {
                "obras_con_datos": fila.get("obras_con_datos", 0) or 0,
                "total_mediciones": fila.get("total_mediciones", 0) or 0,
                "caudal_total": safe_round(fila.get("caudal_total")),
                "caudal_promedio": safe_round(fila.get("caudal_promedio")),
                "caudal_minimo": safe_round(fila.get("caudal_minimo")),
                "caudal_maximo": safe_round(fila.get("caudal_maximo")),
                "caudal_desviacion_estandar": safe_round(
                    fila.get("caudal_desviacion_estandar")
                ),
            },
        }
    except Exception as e:
        logging.error(f"Error in get_shac_stats: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get(
    "/cuencas/stats",
    tags=["Cuencas Hidrográficas"],
    summary="Estadísticas de caudal por cuenca",
    description="Obtiene estadísticas de caudal agregadas por cuenca, subcuenca o subsubcuenca. Incluye caudal promedio, mínimo, máximo, total de puntos y mediciones. Opcionalmente incluye estadísticas globales del sistema.",
)
async def get_cuencas_stats(
    cod_cuenca: Optional[int] = Query(
        None, description="Código de cuenca", example=101
    ),
    cod_subcuenca: Optional[int] = Query(
        None, description="Código de subcuenca", example=10101
    ),
    cod_subsubcuenca: Optional[int] = Query(None, description="Código de subsubcuenca"),
    include_global: bool = Query(
        False, description="Incluir estadísticas globales del sistema completo"
    ),
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
            caudal_desviacion_estandar,
            total_puntos_unicos,
            total_mediciones
        FROM dw.Cuenca_Stats
        {where_clause}
        ORDER BY Cod_Cuenca, Cod_Subcuenca, Cod_Subsubcuenca
        """

        results = await execute_query(stats_query, params)

        if not results:
            return {"estadisticas": []}

        # Get global statistics only if requested
        global_stats = {}
        if include_global:
            global_stats_query = """
            SELECT
                AVG(CAST(caudal_promedio AS FLOAT)) as global_promedio,
                MIN(caudal_minimo) as global_minimo,
                MAX(caudal_maximo) as global_maximo
            FROM dw.Cuenca_Stats
            WHERE caudal_promedio IS NOT NULL
            """
            global_result = await execute_query(global_stats_query)
            global_stats = global_result[0] if global_result else {}

        # Build response
        estadisticas = []
        for r in results:
            stat = {
                "cod_cuenca": r.get("Cod_Cuenca"),
                "nom_cuenca": r.get("Nom_Cuenca"),
                "cod_region": r.get("Cod_Region"),
                "cod_subcuenca": r.get("Cod_Subcuenca"),
                "nom_subcuenca": r.get("Nom_Subcuenca"),
                "cod_subsubcuenca": r.get("Cod_Subsubcuenca"),
                "nom_subsubcuenca": r.get("Nom_Subsubcuenca"),
                "caudal_promedio": safe_round(r.get("caudal_promedio")),
                "caudal_minimo": safe_round(r.get("caudal_minimo")),
                "caudal_maximo": safe_round(r.get("caudal_maximo")),
                # La columna existe en dw.Cuenca_Stats pero no se estaba
                # seleccionando: el frontend recibía undefined y su `|| 0`
                # dejaba la desviación estándar en cero para toda cuenca.
                "caudal_desviacion_estandar": safe_round(
                    r.get("caudal_desviacion_estandar")
                ),
                "total_puntos_unicos": r.get("total_puntos_unicos", 0),
                "total_mediciones": r.get("total_mediciones", 0),
            }

            # Add global stats only if requested
            if include_global:
                stat["global_promedio"] = safe_round(
                    global_stats.get("global_promedio")
                )
                stat["global_minimo"] = safe_round(global_stats.get("global_minimo"))
                stat["global_maximo"] = safe_round(global_stats.get("global_maximo"))

            estadisticas.append(stat)

        return {"estadisticas": estadisticas}

    except Exception as e:
        logging.error(f"Error in get_cuencas_stats: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get(
    "/cuencas/stats_por_tipo",
    tags=["Cuencas Hidrográficas"],
    summary="Estadísticas de caudal separadas por tipo de extracción",
    description=(
        "Estadísticas de caudal de una cuenca, subcuenca o subsubcuenca, separadas "
        "en extracción superficial y subterránea. Incluye el caudal total (suma de "
        "los promedios de cada obra) y el número de obras con datos. La desviación "
        "estándar no se entrega acá: se obtiene de /cuencas/stats, para la cuenca "
        "completa."
    ),
)
async def get_cuencas_stats_por_tipo(
    cod_cuenca: Optional[int] = Query(
        None, description="Código de cuenca", example=101
    ),
    cod_subcuenca: Optional[int] = Query(
        None, description="Código de subcuenca", example=10101
    ),
    cod_subsubcuenca: Optional[int] = Query(None, description="Código de subsubcuenca"),
    shac: Optional[int] = Query(
        None, description="Código de sector SHAC (COD_SECTOR_SHA)"
    ),
):
    """Estadísticas por tipo de extracción, calculadas desde dw.Puntos_Mapa.

    dw.Cuenca_Stats no distingue superficial de subterránea, así que el desglose
    se arma desde dw.Puntos_Mapa, que sí trae es_pozo_subterraneo.

    Esa tabla está agregada por (punto, canal de transmisión), no por punto: en la
    cuenca 101 son 142 filas para 128 obras. Sin colapsar primero por coordenada,
    el número de obras se infla ~11% y el caudal total ~1%. La subconsulta interna
    hace ese colapso, promediando los canales de un mismo punto ponderado por su
    número de mediciones.
    """
    try:
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
        if shac is not None:
            filters.append("COD_SECTOR_SHA = ?")
            params.append(shac)

        if not filters:
            raise HTTPException(
                status_code=400,
                detail="Indique cod_cuenca, cod_subcuenca, cod_subsubcuenca o shac",
            )

        where_clause = "WHERE " + " AND ".join(filters)

        query = f"""
        SELECT
            es_pozo_subterraneo,
            COUNT(*) AS obras_con_datos,
            SUM(n_mediciones) AS total_mediciones,
            SUM(caudal_promedio) AS caudal_total,
            SUM(caudal_promedio * n_mediciones) / NULLIF(SUM(n_mediciones), 0)
                AS caudal_promedio,
            MIN(caudal_minimo) AS caudal_minimo,
            MAX(caudal_maximo) AS caudal_maximo
        FROM (
            SELECT
                UTM_Norte,
                UTM_Este,
                MAX(CAST(es_pozo_subterraneo AS INT)) AS es_pozo_subterraneo,
                SUM(n_mediciones) AS n_mediciones,
                SUM(caudal_promedio * n_mediciones) / NULLIF(SUM(n_mediciones), 0)
                    AS caudal_promedio,
                MIN(caudal_minimo) AS caudal_minimo,
                MAX(caudal_maximo) AS caudal_maximo
            FROM dw.Puntos_Mapa
            {where_clause}
              AND caudal_promedio IS NOT NULL
              AND n_mediciones > 0
            GROUP BY UTM_Norte, UTM_Este
        ) AS obras
        GROUP BY es_pozo_subterraneo
        """

        results = await execute_query(query, params)

        por_tipo = {}
        for r in results:
            clave = "subterranea" if r.get("es_pozo_subterraneo") else "superficial"
            por_tipo[clave] = {
                "obras_con_datos": r.get("obras_con_datos", 0),
                "total_mediciones": r.get("total_mediciones", 0),
                "caudal_total": safe_round(r.get("caudal_total")),
                "caudal_promedio": safe_round(r.get("caudal_promedio")),
                "caudal_minimo": safe_round(r.get("caudal_minimo")),
                "caudal_maximo": safe_round(r.get("caudal_maximo")),
            }

        # Un grupo sin obras no viene en el resultado; se devuelve explícito en cero
        # para que el frontend distinga "sin obras" de "no consultado".
        for clave in ("superficial", "subterranea"):
            por_tipo.setdefault(
                clave,
                {
                    "obras_con_datos": 0,
                    "total_mediciones": 0,
                    "caudal_total": None,
                    "caudal_promedio": None,
                    "caudal_minimo": None,
                    "caudal_maximo": None,
                },
            )

        return por_tipo

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_cuencas_stats_por_tipo: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})
