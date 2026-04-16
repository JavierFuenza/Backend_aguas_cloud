import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from core.database import execute_query
from core.cache_manager import CACHE_TTL_STATIC
from utils.helpers import safe_round, build_full_name
from models.schemas import CuencaData, CuencaStatsResponse

router = APIRouter()

@router.get(
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

        results = await execute_query(cuencas_query, ttl=CACHE_TTL_STATIC)

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


@router.get(
    "/shacs",
    tags=["Cuencas Hidrográficas"],
    summary="Listado de sectores SHAC",
    description="Obtiene el listado de Sectores Hidrogeológicos de Aprovechamiento Común (SHAC) con total de puntos."
)
async def get_shacs():
    """Obtiene lista de SHACs disponibles con conteo de puntos"""
    try:
        query = """
        SELECT
            COD_SECTOR_SHA AS cod_sector_sha,
            SECTOR_SHA AS sector_sha,
            COUNT(*) AS total_puntos
        FROM dw.Puntos_Mapa
        WHERE COD_SECTOR_SHA IS NOT NULL
        GROUP BY COD_SECTOR_SHA, SECTOR_SHA
        ORDER BY COD_SECTOR_SHA
        """
        results = await execute_query(query)
        return {
            "shacs": [
                {
                    "cod_sector_sha": r.get("cod_sector_sha"),
                    "sector_sha": r.get("sector_sha"),
                    "total_puntos": r.get("total_puntos", 0)
                } for r in results
            ]
        }
    except Exception as e:
        logging.error(f"Error in get_shacs: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get(
    "/juntas",
    tags=["Cuencas Hidrográficas"],
    summary="Listado de Juntas de Vigilancia",
    description="Obtiene el listado de Juntas de Vigilancia con total de puntos asociados."
)
async def get_juntas():
    """Obtiene lista de Juntas disponibles con conteo de puntos"""
    try:
        query = """
        SELECT
            ID_JUNTA AS id_junta,
            COUNT(*) AS total_puntos
        FROM dw.Puntos_Mapa
        WHERE ID_JUNTA IS NOT NULL
        GROUP BY ID_JUNTA
        ORDER BY ID_JUNTA
        """
        results = await execute_query(query)
        return {
            "juntas": [
                {
                    "id_junta": r.get("id_junta"),
                    "total_puntos": r.get("total_puntos", 0)
                } for r in results
            ]
        }
    except Exception as e:
        logging.error(f"Error in get_juntas: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get(
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
        results = await execute_query(stats_query)

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

@router.get(
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

        results = await execute_query(stats_query, params)

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
            global_result = await execute_query(global_stats_query)
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

