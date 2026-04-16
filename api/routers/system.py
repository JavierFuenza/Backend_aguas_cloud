import logging
from fastapi import APIRouter, HTTPException
from core.database import execute_query
from models.schemas import HealthResponse

router = APIRouter()

@router.get(
    "/health",
    tags=["System"],
    response_model=HealthResponse,
    summary="Verificación de estado del servicio",
    description="Verifica el estado del servicio API y la conectividad con la base de datos Azure Synapse. Retorna el estado del servicio y de la conexión a base de datos."
)
async def health_check():
    """Health check endpoint with database connectivity test"""
    try:
        await execute_query("SELECT 1 as test", use_cache=False)
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

@router.get("/test-db", tags=["System"])
async def test_database_connection():
    """Test database connection with record count"""
    try:
        results = await execute_query(
            "SELECT SUM(row_count) as total FROM sys.dm_db_partition_stats "
            "WHERE object_id = OBJECT_ID('dw.Mediciones_full') AND index_id IN (0,1)"
        )
        total = results[0]['total'] if results and results[0]['total'] is not None else None
        if total is None:
            raise HTTPException(status_code=503, detail={
                "status": "error",
                "message": "Table dw.Mediciones_full not found or has no rows"
            })
        return {
            "status": "success",
            "message": "Database connection successful",
            "total_records": total
        }
    except Exception as e:
        logging.error(f"Database connection failed: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "status": "error",
            "message": f"Database connection failed: {str(e)}"
        })

@router.get("/count", tags=["System"])
async def get_obras_count():
    """Obtiene el número total de registros en la tabla de mediciones"""
    try:
        results = await execute_query(
            "SELECT SUM(row_count) as total FROM sys.dm_db_partition_stats "
            "WHERE object_id = OBJECT_ID('dw.Mediciones_full') AND index_id IN (0,1)"
        )
        total = results[0]['total'] if results and results[0]['total'] is not None else None
        if total is None:
            raise HTTPException(status_code=503, detail={"error": "Table dw.Mediciones_full not found or has no rows"})
        return {"total_records": total}
    except Exception as e:
        logging.error(f"Error in get_obras_count: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})
