import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from core.database import execute_query
from core.cache_manager import CACHE_TTL_STATIC
from utils.helpers import safe_round, build_full_name

router = APIRouter()

@router.get(
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

        results = await execute_query(atlas_query, ttl=CACHE_TTL_STATIC)

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

