import logging
from fastapi import APIRouter, HTTPException
from core.database import execute_query, connection_pool
from core.cache_manager import memory_cache, clear_all_cache

router = APIRouter()

@router.get(
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

@router.post(
    "/cache/clear",
    tags=["Cache y Rendimiento"],
    summary="Limpiar caché del sistema",
    description="Elimina todos los datos almacenados en la caché del sistema. Las próximas consultas consultarán directamente la base de datos y el caché se reconstruirá automáticamente."
)
async def clear_cache():
    """Clear all cached data"""
    clear_all_cache()
    return {"message": "Cache cleared successfully"}

@router.get(
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
            ("SELECT SUM(row_count) as total FROM sys.dm_pdw_nodes_db_partition_stats WHERE object_id = OBJECT_ID('dw.Mediciones_full') AND index_id IN (0,1)", None),
            ("SELECT COUNT(*) as total_puntos_unicos FROM dw.Puntos_Mapa", None),
            ("SELECT DISTINCT Region FROM dw.Puntos_Mapa WHERE Region IS NOT NULL ORDER BY Region", None),
        ]

        warmed_queries = 0
        for query, params in queries:
            try:
                await execute_query(query, params, use_cache=True)
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
