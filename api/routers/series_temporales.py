import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from core.database import execute_query
from utils.helpers import safe_round, build_full_name
from models.schemas import TimeSeriesPoint, AlturaTimeSeriesPoint, NivelFreaticoTimeSeriesPoint

router = APIRouter()

@router.get("/cuencas/cuenca/series_de_tiempo/caudal", tags=["Series Temporales"], summary="Serie temporal de caudal por cuenca")
async def get_caudal_por_tiempo_por_cuenca(
    cuenca_identificador: str = Query(..., description="Código numérico o nombre de la cuenca", example="101"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio en formato YYYY-MM-DD", example="2023-01-01"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin en formato YYYY-MM-DD", example="2023-12-31"),
    pozo: Optional[bool] = Query(None, description="Filtrar por tipo de extracción: True (subterránea), False (superficial)")
):
    try:
        if cuenca_identificador.isdigit():
            subquery_filter = "s.COD_CUENCA = ?"
            params = [int(cuenca_identificador)]
        else:
            subquery_filter = "s.NOM_CUENCA = ?"
            params = [cuenca_identificador]

        join_puntos = "INNER JOIN dw.Puntos_Mapa p ON s.UTM_NORTE = p.UTM_Norte AND s.UTM_ESTE = p.UTM_Este" if pozo is not None else ""

        query = f"""
        SELECT 
            s.FECHA_MEDICION AS fecha_medicion, 
            AVG(CAST(s.CAUDAL AS FLOAT)) AS caudal_promedio,
            SUM(CAST(s.CAUDAL AS FLOAT)) AS caudal_sumado,
            SUM(CAST(s.TOTALIZADOR AS BIGINT)) AS totalizador_sumado,
            MAX(CAST(s.TOTALIZADOR AS BIGINT)) AS totalizador_max
        FROM dw.Series_tiempo s
        {join_puntos}
        WHERE {subquery_filter}
        AND s.CAUDAL IS NOT NULL
        """
        if fecha_inicio:
            query += " AND s.FECHA_MEDICION >= ?"
            params.append(fecha_inicio)
        if fecha_fin:
            query += " AND s.FECHA_MEDICION <= ?"
            params.append(fecha_fin)
        if pozo is not None:
            query += " AND p.es_pozo_subterraneo = ?"
            params.append(1 if pozo else 0)
        query += " GROUP BY s.FECHA_MEDICION ORDER BY s.FECHA_MEDICION DESC"

        results = await execute_query(query, params)

        caudal_por_tiempo = [{
            "fecha_medicion": str(r["fecha_medicion"]) if r.get("fecha_medicion") else None, 
            "caudal": r.get("caudal_promedio"),
            "caudal_promedio": r.get("caudal_promedio"),
            "caudal_sumado": r.get("caudal_sumado"),
            "totalizador_sumado": r.get("totalizador_sumado"),
            "totalizador_max": r.get("totalizador_max")
        } for r in results] if results else []
        return {"cuenca_identificador": cuenca_identificador, "total_registros": len(caudal_por_tiempo), "caudal_por_tiempo": caudal_por_tiempo}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail={"error": str(e)})

@router.get("/cuencas/cuenca/series_de_tiempo/altura_linimetrica", tags=["Series Temporales"], summary="Serie temporal de altura limnimétrica por cuenca")
async def get_altura_linimetrica_por_tiempo_por_cuenca(
    cuenca_identificador: str = Query(..., description="Código numérico o nombre de la cuenca", example="101"),
    fecha_inicio: Optional[str] = Query(None), fecha_fin: Optional[str] = Query(None),
    pozo: Optional[bool] = Query(None, description="Filtrar por tipo de extracción")
):
    try:
        if cuenca_identificador.isdigit():
            subquery_filter = "s.COD_CUENCA = ?"
            params = [int(cuenca_identificador)]
        else:
            subquery_filter = "s.NOM_CUENCA = ?"
            params = [cuenca_identificador]

        join_puntos = "INNER JOIN dw.Puntos_Mapa p ON s.UTM_NORTE = p.UTM_Norte AND s.UTM_ESTE = p.UTM_Este" if pozo is not None else ""

        query = f"""
        SELECT s.FECHA_MEDICION AS fecha_medicion, AVG(CAST(s.ALTURA_LIMNIMETRICA AS FLOAT)) AS altura_linimetrica
        FROM dw.Series_tiempo s
        {join_puntos}
        WHERE {subquery_filter}
        AND s.ALTURA_LIMNIMETRICA IS NOT NULL
        """
        if fecha_inicio:
            query += " AND s.FECHA_MEDICION >= ?"
            params.append(fecha_inicio)
        if fecha_fin:
            query += " AND s.FECHA_MEDICION <= ?"
            params.append(fecha_fin)
        if pozo is not None:
            query += " AND p.es_pozo_subterraneo = ?"
            params.append(1 if pozo else 0)
        query += " GROUP BY s.FECHA_MEDICION ORDER BY s.FECHA_MEDICION DESC"

        results = await execute_query(query, params)
        
        altura_por_tiempo = [{"fecha_medicion": str(r["fecha_medicion"]) if r.get("fecha_medicion") else None, "altura_linimetrica": r["altura_linimetrica"]} for r in results] if results else []
        return {"cuenca_identificador": cuenca_identificador, "total_registros": len(altura_por_tiempo), "altura_por_tiempo": altura_por_tiempo}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail={"error": str(e)})

@router.get("/cuencas/cuenca/series_de_tiempo/nivel_freatico", tags=["Series Temporales"], summary="Serie temporal de nivel freático por cuenca")
async def get_nivel_freatico_por_tiempo_por_cuenca(
    cuenca_identificador: str = Query(..., description="Código numérico o nombre de la cuenca", example="101"),
    fecha_inicio: Optional[str] = Query(None), fecha_fin: Optional[str] = Query(None),
    pozo: Optional[bool] = Query(None, description="Filtrar por tipo de extracción")
):
    try:
        if cuenca_identificador.isdigit():
            subquery_filter = "s.COD_CUENCA = ?"
            params = [int(cuenca_identificador)]
        else:
            subquery_filter = "s.NOM_CUENCA = ?"
            params = [cuenca_identificador]

        join_puntos = "INNER JOIN dw.Puntos_Mapa p ON s.UTM_NORTE = p.UTM_Norte AND s.UTM_ESTE = p.UTM_Este" if pozo is not None else ""

        query = f"""
        SELECT s.FECHA_MEDICION AS fecha_medicion, AVG(CAST(s.NIVEL_FREATICO AS FLOAT)) AS nivel_freatico
        FROM dw.Series_tiempo s
        {join_puntos}
        WHERE {subquery_filter}
        AND s.NIVEL_FREATICO IS NOT NULL
        """
        if fecha_inicio:
            query += " AND s.FECHA_MEDICION >= ?"
            params.append(fecha_inicio)
        if fecha_fin:
            query += " AND s.FECHA_MEDICION <= ?"
            params.append(fecha_fin)
        if pozo is not None:
            query += " AND p.es_pozo_subterraneo = ?"
            params.append(1 if pozo else 0)
        query += " GROUP BY s.FECHA_MEDICION ORDER BY s.FECHA_MEDICION DESC"

        results = await execute_query(query, params)
        
        nivel_por_tiempo = [{"fecha_medicion": str(r["fecha_medicion"]) if r.get("fecha_medicion") else None, "nivel_freatico": r["nivel_freatico"]} for r in results] if results else []
        return {"cuenca_identificador": cuenca_identificador, "total_registros": len(nivel_por_tiempo), "nivel_por_tiempo": nivel_por_tiempo}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail={"error": str(e)})

@router.get("/cuencas/subcuenca/series_de_tiempo/caudal", tags=["Series Temporales"], summary="Serie temporal de caudal por subcuenca")
async def get_caudal_por_tiempo_por_subcuenca(
    cuenca_identificador: str = Query(..., description="Código numérico o nombre de la subcuenca", example="101"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio en formato YYYY-MM-DD", example="2023-01-01"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin en formato YYYY-MM-DD", example="2023-12-31"),
    pozo: Optional[bool] = Query(None, description="Filtrar por tipo de extracción: True (subterránea), False (superficial)")
):
    try:
        if cuenca_identificador.isdigit():
            subquery_filter = "s.COD_SUBCUENCA = ?"
            params = [int(cuenca_identificador)]
        else:
            subquery_filter = "s.NOM_SUBCUENCA = ?"
            params = [cuenca_identificador]

        join_puntos = "INNER JOIN dw.Puntos_Mapa p ON s.UTM_NORTE = p.UTM_Norte AND s.UTM_ESTE = p.UTM_Este" if pozo is not None else ""

        query = f"""
        SELECT 
            s.FECHA_MEDICION AS fecha_medicion, 
            AVG(CAST(s.CAUDAL AS FLOAT)) AS caudal_promedio,
            SUM(CAST(s.CAUDAL AS FLOAT)) AS caudal_sumado,
            SUM(CAST(s.TOTALIZADOR AS BIGINT)) AS totalizador_sumado,
            MAX(CAST(s.TOTALIZADOR AS BIGINT)) AS totalizador_max
        FROM dw.Series_tiempo s
        {join_puntos}
        WHERE {subquery_filter}
        AND s.CAUDAL IS NOT NULL
        """
        if fecha_inicio:
            query += " AND s.FECHA_MEDICION >= ?"
            params.append(fecha_inicio)
        if fecha_fin:
            query += " AND s.FECHA_MEDICION <= ?"
            params.append(fecha_fin)
        if pozo is not None:
            query += " AND p.es_pozo_subterraneo = ?"
            params.append(1 if pozo else 0)
        query += " GROUP BY s.FECHA_MEDICION ORDER BY s.FECHA_MEDICION DESC"

        results = await execute_query(query, params)

        caudal_por_tiempo = [{
            "fecha_medicion": str(r["fecha_medicion"]) if r.get("fecha_medicion") else None, 
            "caudal": r.get("caudal_promedio"),
            "caudal_promedio": r.get("caudal_promedio"),
            "caudal_sumado": r.get("caudal_sumado"),
            "totalizador_sumado": r.get("totalizador_sumado"),
            "totalizador_max": r.get("totalizador_max")
        } for r in results] if results else []
        return {"subcuenca_identificador": cuenca_identificador, "total_registros": len(caudal_por_tiempo), "caudal_por_tiempo": caudal_por_tiempo}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail={"error": str(e)})

@router.get("/cuencas/subcuenca/series_de_tiempo/altura_linimetrica", tags=["Series Temporales"], summary="Serie temporal de altura limnimétrica por subcuenca")
async def get_altura_linimetrica_por_tiempo_por_subcuenca(
    cuenca_identificador: str = Query(..., description="Código numérico o nombre de la subcuenca", example="101"),
    fecha_inicio: Optional[str] = Query(None), fecha_fin: Optional[str] = Query(None),
    pozo: Optional[bool] = Query(None, description="Filtrar por tipo de extracción")
):
    try:
        if cuenca_identificador.isdigit():
            subquery_filter = "s.COD_SUBCUENCA = ?"
            params = [int(cuenca_identificador)]
        else:
            subquery_filter = "s.NOM_SUBCUENCA = ?"
            params = [cuenca_identificador]

        join_puntos = "INNER JOIN dw.Puntos_Mapa p ON s.UTM_NORTE = p.UTM_Norte AND s.UTM_ESTE = p.UTM_Este" if pozo is not None else ""

        query = f"""
        SELECT s.FECHA_MEDICION AS fecha_medicion, AVG(CAST(s.ALTURA_LIMNIMETRICA AS FLOAT)) AS altura_linimetrica
        FROM dw.Series_tiempo s
        {join_puntos}
        WHERE {subquery_filter}
        AND s.ALTURA_LIMNIMETRICA IS NOT NULL
        """
        if fecha_inicio:
            query += " AND s.FECHA_MEDICION >= ?"
            params.append(fecha_inicio)
        if fecha_fin:
            query += " AND s.FECHA_MEDICION <= ?"
            params.append(fecha_fin)
        if pozo is not None:
            query += " AND p.es_pozo_subterraneo = ?"
            params.append(1 if pozo else 0)
        query += " GROUP BY s.FECHA_MEDICION ORDER BY s.FECHA_MEDICION DESC"

        results = await execute_query(query, params)
        
        altura_por_tiempo = [{"fecha_medicion": str(r["fecha_medicion"]) if r.get("fecha_medicion") else None, "altura_linimetrica": r["altura_linimetrica"]} for r in results] if results else []
        return {"subcuenca_identificador": cuenca_identificador, "total_registros": len(altura_por_tiempo), "altura_por_tiempo": altura_por_tiempo}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail={"error": str(e)})

@router.get("/cuencas/subcuenca/series_de_tiempo/nivel_freatico", tags=["Series Temporales"], summary="Serie temporal de nivel freático por subcuenca")
async def get_nivel_freatico_por_tiempo_por_subcuenca(
    cuenca_identificador: str = Query(..., description="Código numérico o nombre de la subcuenca", example="101"),
    fecha_inicio: Optional[str] = Query(None), fecha_fin: Optional[str] = Query(None),
    pozo: Optional[bool] = Query(None, description="Filtrar por tipo de extracción")
):
    try:
        if cuenca_identificador.isdigit():
            subquery_filter = "s.COD_SUBCUENCA = ?"
            params = [int(cuenca_identificador)]
        else:
            subquery_filter = "s.NOM_SUBCUENCA = ?"
            params = [cuenca_identificador]

        join_puntos = "INNER JOIN dw.Puntos_Mapa p ON s.UTM_NORTE = p.UTM_Norte AND s.UTM_ESTE = p.UTM_Este" if pozo is not None else ""

        query = f"""
        SELECT s.FECHA_MEDICION AS fecha_medicion, AVG(CAST(s.NIVEL_FREATICO AS FLOAT)) AS nivel_freatico
        FROM dw.Series_tiempo s
        {join_puntos}
        WHERE {subquery_filter}
        AND s.NIVEL_FREATICO IS NOT NULL
        """
        if fecha_inicio:
            query += " AND s.FECHA_MEDICION >= ?"
            params.append(fecha_inicio)
        if fecha_fin:
            query += " AND s.FECHA_MEDICION <= ?"
            params.append(fecha_fin)
        if pozo is not None:
            query += " AND p.es_pozo_subterraneo = ?"
            params.append(1 if pozo else 0)
        query += " GROUP BY s.FECHA_MEDICION ORDER BY s.FECHA_MEDICION DESC"

        results = await execute_query(query, params)
        
        nivel_por_tiempo = [{"fecha_medicion": str(r["fecha_medicion"]) if r.get("fecha_medicion") else None, "nivel_freatico": r["nivel_freatico"]} for r in results] if results else []
        return {"subcuenca_identificador": cuenca_identificador, "total_registros": len(nivel_por_tiempo), "nivel_por_tiempo": nivel_por_tiempo}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail={"error": str(e)})

@router.get("/cuencas/subsubcuenca/series_de_tiempo/caudal", tags=["Series Temporales"], summary="Serie temporal de caudal por subsubcuenca")
async def get_caudal_por_tiempo_por_subsubcuenca(
    cuenca_identificador: str = Query(..., description="Código numérico o nombre de la subsubcuenca", example="101"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio en formato YYYY-MM-DD", example="2023-01-01"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin en formato YYYY-MM-DD", example="2023-12-31"),
    pozo: Optional[bool] = Query(None, description="Filtrar por tipo de extracción: True (subterránea), False (superficial)")
):
    try:
        if cuenca_identificador.isdigit():
            subquery_filter = "s.COD_SUBSUBCUENCA = ?"
            params = [int(cuenca_identificador)]
        else:
            subquery_filter = "s.NOM_SUBSUBCUENCA = ?"
            params = [cuenca_identificador]

        join_puntos = "INNER JOIN dw.Puntos_Mapa p ON s.UTM_NORTE = p.UTM_Norte AND s.UTM_ESTE = p.UTM_Este" if pozo is not None else ""

        query = f"""
        SELECT 
            s.FECHA_MEDICION AS fecha_medicion, 
            AVG(CAST(s.CAUDAL AS FLOAT)) AS caudal_promedio,
            SUM(CAST(s.CAUDAL AS FLOAT)) AS caudal_sumado,
            SUM(CAST(s.TOTALIZADOR AS BIGINT)) AS totalizador_sumado,
            MAX(CAST(s.TOTALIZADOR AS BIGINT)) AS totalizador_max
        FROM dw.Series_tiempo s
        {join_puntos}
        WHERE {subquery_filter}
        AND s.CAUDAL IS NOT NULL
        """
        if fecha_inicio:
            query += " AND s.FECHA_MEDICION >= ?"
            params.append(fecha_inicio)
        if fecha_fin:
            query += " AND s.FECHA_MEDICION <= ?"
            params.append(fecha_fin)
        if pozo is not None:
            query += " AND p.es_pozo_subterraneo = ?"
            params.append(1 if pozo else 0)
        query += " GROUP BY s.FECHA_MEDICION ORDER BY s.FECHA_MEDICION DESC"

        results = await execute_query(query, params)

        caudal_por_tiempo = [{
            "fecha_medicion": str(r["fecha_medicion"]) if r.get("fecha_medicion") else None, 
            "caudal": r.get("caudal_promedio"),
            "caudal_promedio": r.get("caudal_promedio"),
            "caudal_sumado": r.get("caudal_sumado"),
            "totalizador_sumado": r.get("totalizador_sumado"),
            "totalizador_max": r.get("totalizador_max")
        } for r in results] if results else []
        return {"subsubcuenca_identificador": cuenca_identificador, "total_registros": len(caudal_por_tiempo), "caudal_por_tiempo": caudal_por_tiempo}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail={"error": str(e)})

@router.get("/cuencas/subsubcuenca/series_de_tiempo/altura_linimetrica", tags=["Series Temporales"], summary="Serie temporal de altura limnimétrica por subsubcuenca")
async def get_altura_linimetrica_por_tiempo_por_subsubcuenca(
    cuenca_identificador: str = Query(..., description="Código numérico o nombre de la subsubcuenca", example="101"),
    fecha_inicio: Optional[str] = Query(None), fecha_fin: Optional[str] = Query(None),
    pozo: Optional[bool] = Query(None, description="Filtrar por tipo de extracción")
):
    try:
        if cuenca_identificador.isdigit():
            subquery_filter = "s.COD_SUBSUBCUENCA = ?"
            params = [int(cuenca_identificador)]
        else:
            subquery_filter = "s.NOM_SUBSUBCUENCA = ?"
            params = [cuenca_identificador]

        join_puntos = "INNER JOIN dw.Puntos_Mapa p ON s.UTM_NORTE = p.UTM_Norte AND s.UTM_ESTE = p.UTM_Este" if pozo is not None else ""

        query = f"""
        SELECT s.FECHA_MEDICION AS fecha_medicion, AVG(CAST(s.ALTURA_LIMNIMETRICA AS FLOAT)) AS altura_linimetrica
        FROM dw.Series_tiempo s
        {join_puntos}
        WHERE {subquery_filter}
        AND s.ALTURA_LIMNIMETRICA IS NOT NULL
        """
        if fecha_inicio:
            query += " AND s.FECHA_MEDICION >= ?"
            params.append(fecha_inicio)
        if fecha_fin:
            query += " AND s.FECHA_MEDICION <= ?"
            params.append(fecha_fin)
        if pozo is not None:
            query += " AND p.es_pozo_subterraneo = ?"
            params.append(1 if pozo else 0)
        query += " GROUP BY s.FECHA_MEDICION ORDER BY s.FECHA_MEDICION DESC"

        results = await execute_query(query, params)
        
        altura_por_tiempo = [{"fecha_medicion": str(r["fecha_medicion"]) if r.get("fecha_medicion") else None, "altura_linimetrica": r["altura_linimetrica"]} for r in results] if results else []
        return {"subsubcuenca_identificador": cuenca_identificador, "total_registros": len(altura_por_tiempo), "altura_por_tiempo": altura_por_tiempo}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail={"error": str(e)})

@router.get("/cuencas/subsubcuenca/series_de_tiempo/nivel_freatico", tags=["Series Temporales"], summary="Serie temporal de nivel freático por subsubcuenca")
async def get_nivel_freatico_por_tiempo_por_subsubcuenca(
    cuenca_identificador: str = Query(..., description="Código numérico o nombre de la subsubcuenca", example="101"),
    fecha_inicio: Optional[str] = Query(None), fecha_fin: Optional[str] = Query(None),
    pozo: Optional[bool] = Query(None, description="Filtrar por tipo de extracción")
):
    try:
        if cuenca_identificador.isdigit():
            subquery_filter = "s.COD_SUBSUBCUENCA = ?"
            params = [int(cuenca_identificador)]
        else:
            subquery_filter = "s.NOM_SUBSUBCUENCA = ?"
            params = [cuenca_identificador]

        join_puntos = "INNER JOIN dw.Puntos_Mapa p ON s.UTM_NORTE = p.UTM_Norte AND s.UTM_ESTE = p.UTM_Este" if pozo is not None else ""

        query = f"""
        SELECT s.FECHA_MEDICION AS fecha_medicion, AVG(CAST(s.NIVEL_FREATICO AS FLOAT)) AS nivel_freatico
        FROM dw.Series_tiempo s
        {join_puntos}
        WHERE {subquery_filter}
        AND s.NIVEL_FREATICO IS NOT NULL
        """
        if fecha_inicio:
            query += " AND s.FECHA_MEDICION >= ?"
            params.append(fecha_inicio)
        if fecha_fin:
            query += " AND s.FECHA_MEDICION <= ?"
            params.append(fecha_fin)
        if pozo is not None:
            query += " AND p.es_pozo_subterraneo = ?"
            params.append(1 if pozo else 0)
        query += " GROUP BY s.FECHA_MEDICION ORDER BY s.FECHA_MEDICION DESC"

        results = await execute_query(query, params)
        
        nivel_por_tiempo = [{"fecha_medicion": str(r["fecha_medicion"]) if r.get("fecha_medicion") else None, "nivel_freatico": r["nivel_freatico"]} for r in results] if results else []
        return {"subsubcuenca_identificador": cuenca_identificador, "total_registros": len(nivel_por_tiempo), "nivel_por_tiempo": nivel_por_tiempo}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail={"error": str(e)})

@router.get("/cuencas/shac/series_de_tiempo/caudal", 
    tags=["Series Temporales"], 
    summary="Serie temporal de caudal por SHAC",
    description="Obtiene la serie temporal de caudal usando dw.Series_tiempo con caudales sumados y totalizadores agrupados por Sector Hidrogeológico de Aprovechamiento Común (SHAC)."
)
async def get_caudal_por_tiempo_por_shac(
    shac_identificador: str = Query(..., description="Código o nombre del SHAC", example="121"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio en formato YYYY-MM-DD"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin en formato YYYY-MM-DD"),
    pozo: Optional[bool] = Query(None, description="Filtrar por tipo de extracción")
):
    try:
        if shac_identificador.isdigit():
            shac_filter = "p.COD_SECTOR_SHA = ?"
            params = [int(shac_identificador)]
        else:
            shac_filter = "p.SECTOR_SHA = ?"
            params = [shac_identificador]

        query = f"""
        SELECT 
            s.FECHA_MEDICION AS fecha_medicion, 
            AVG(CAST(s.CAUDAL AS FLOAT)) AS caudal_promedio,
            SUM(CAST(s.CAUDAL AS FLOAT)) AS caudal_sumado,
            SUM(CAST(s.TOTALIZADOR AS BIGINT)) AS totalizador_sumado,
            MAX(CAST(s.TOTALIZADOR AS BIGINT)) AS totalizador_max
        FROM dw.Series_tiempo s
        INNER JOIN dw.Puntos_Mapa p 
            ON s.UTM_NORTE = p.UTM_Norte 
            AND s.UTM_ESTE = p.UTM_Este
        WHERE {shac_filter}
          AND s.CAUDAL IS NOT NULL
        """

        if fecha_inicio:
            query += " AND s.FECHA_MEDICION >= ?"
            params.append(fecha_inicio)
            
        if fecha_fin:
            query += " AND s.FECHA_MEDICION <= ?"
            params.append(fecha_fin)
            
        if pozo is not None:
            query += " AND p.es_pozo_subterraneo = ?"
            params.append(1 if pozo else 0)
            
        query += " GROUP BY s.FECHA_MEDICION ORDER BY s.FECHA_MEDICION DESC"

        results = await execute_query(query, params)

        if not results:
            raise HTTPException(status_code=404, detail="No se encontraron datos de caudal para el período o SHAC especificado.")

        caudal_por_tiempo = [
            {
                "fecha_medicion": str(r["fecha_medicion"]) if r.get("fecha_medicion") else None,
                "caudal": r.get("caudal_promedio"),
                "caudal_promedio": r.get("caudal_promedio"),
                "caudal_sumado": r.get("caudal_sumado"),
                "totalizador_sumado": r.get("totalizador_sumado"),
                "totalizador_max": r.get("totalizador_max")
            }
            for r in results
        ]

        return {
            "shac_identificador": shac_identificador,
            "total_registros": len(caudal_por_tiempo),
            "caudal_por_tiempo": caudal_por_tiempo
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_caudal_por_tiempo_por_shac: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@router.get("/cuencas/shac/series_de_tiempo/altura_linimetrica",
    tags=["Series Temporales"],
    summary="Serie temporal de altura limnimétrica por SHAC"
)
async def get_altura_linimetrica_por_tiempo_por_shac(
    shac_identificador: str = Query(..., description="Código o nombre del SHAC", example="121"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio en formato YYYY-MM-DD"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin en formato YYYY-MM-DD"),
    pozo: Optional[bool] = Query(None, description="Filtrar por tipo de extracción")
):
    try:
        if shac_identificador.isdigit():
            shac_filter = "p.COD_SECTOR_SHA = ?"
            params = [int(shac_identificador)]
        else:
            shac_filter = "p.SECTOR_SHA = ?"
            params = [shac_identificador]

        query = f"""
        SELECT
            s.FECHA_MEDICION AS fecha_medicion,
            AVG(CAST(s.ALTURA_LIMNIMETRICA AS FLOAT)) AS altura_linimetrica
        FROM dw.Series_tiempo s
        INNER JOIN dw.Puntos_Mapa p
            ON s.UTM_NORTE = p.UTM_Norte
            AND s.UTM_ESTE = p.UTM_Este
        WHERE {shac_filter}
          AND s.ALTURA_LIMNIMETRICA IS NOT NULL
        """

        if fecha_inicio:
            query += " AND s.FECHA_MEDICION >= ?"
            params.append(fecha_inicio)
        if fecha_fin:
            query += " AND s.FECHA_MEDICION <= ?"
            params.append(fecha_fin)
        if pozo is not None:
            query += " AND p.es_pozo_subterraneo = ?"
            params.append(1 if pozo else 0)
        query += " GROUP BY s.FECHA_MEDICION ORDER BY s.FECHA_MEDICION DESC"

        results = await execute_query(query, params)

        if not results:
            raise HTTPException(status_code=404, detail="No se encontraron datos de altura limnimétrica para el período o SHAC especificado.")

        altura_por_tiempo = [
            {
                "fecha_medicion": str(r["fecha_medicion"]) if r.get("fecha_medicion") else None,
                "altura_linimetrica": r.get("altura_linimetrica")
            }
            for r in results
        ]

        return {
            "shac_identificador": shac_identificador,
            "total_registros": len(altura_por_tiempo),
            "altura_por_tiempo": altura_por_tiempo
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_altura_linimetrica_por_tiempo_por_shac: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get("/cuencas/shac/series_de_tiempo/nivel_freatico",
    tags=["Series Temporales"],
    summary="Serie temporal de nivel freático por SHAC"
)
async def get_nivel_freatico_por_tiempo_por_shac(
    shac_identificador: str = Query(..., description="Código o nombre del SHAC", example="121"),
    fecha_inicio: Optional[str] = Query(None, description="Fecha de inicio en formato YYYY-MM-DD"),
    fecha_fin: Optional[str] = Query(None, description="Fecha de fin en formato YYYY-MM-DD"),
    pozo: Optional[bool] = Query(None, description="Filtrar por tipo de extracción")
):
    try:
        if shac_identificador.isdigit():
            shac_filter = "p.COD_SECTOR_SHA = ?"
            params = [int(shac_identificador)]
        else:
            shac_filter = "p.SECTOR_SHA = ?"
            params = [shac_identificador]

        query = f"""
        SELECT
            s.FECHA_MEDICION AS fecha_medicion,
            AVG(CAST(s.NIVEL_FREATICO AS FLOAT)) AS nivel_freatico
        FROM dw.Series_tiempo s
        INNER JOIN dw.Puntos_Mapa p
            ON s.UTM_NORTE = p.UTM_Norte
            AND s.UTM_ESTE = p.UTM_Este
        WHERE {shac_filter}
          AND s.NIVEL_FREATICO IS NOT NULL
        """

        if fecha_inicio:
            query += " AND s.FECHA_MEDICION >= ?"
            params.append(fecha_inicio)
        if fecha_fin:
            query += " AND s.FECHA_MEDICION <= ?"
            params.append(fecha_fin)
        if pozo is not None:
            query += " AND p.es_pozo_subterraneo = ?"
            params.append(1 if pozo else 0)
        query += " GROUP BY s.FECHA_MEDICION ORDER BY s.FECHA_MEDICION DESC"

        results = await execute_query(query, params)

        if not results:
            raise HTTPException(status_code=404, detail="No se encontraron datos de nivel freático para el período o SHAC especificado.")

        nivel_por_tiempo = [
            {
                "fecha_medicion": str(r["fecha_medicion"]) if r.get("fecha_medicion") else None,
                "nivel_freatico": r.get("nivel_freatico")
            }
            for r in results
        ]

        return {
            "shac_identificador": shac_identificador,
            "total_registros": len(nivel_por_tiempo),
            "nivel_por_tiempo": nivel_por_tiempo
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_nivel_freatico_por_tiempo_por_shac: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get("/puntos/series_de_tiempo/caudal", tags=["Series Temporales"])
async def get_caudal_por_tiempo_por_punto(
    utm_norte: int = Query(...), utm_este: int = Query(...),
    fecha_inicio: Optional[str] = Query(None), fecha_fin: Optional[str] = Query(None)
):
    try:
        query = """
        SELECT 
            FECHA_MEDICION as fecha_medicion, 
            CAUDAL as caudal,
            CAUDAL as caudal_promedio,
            CAUDAL as caudal_sumado,
            TOTALIZADOR as totalizador_max,
            TOTALIZADOR as totalizador_sumado
        FROM dw.Series_tiempo
        WHERE UTM_NORTE = ? AND UTM_ESTE = ? AND CAUDAL IS NOT NULL
        """
        params = [utm_norte, utm_este]
        if fecha_inicio:
            query += " AND FECHA_MEDICION >= ?"
            params.append(fecha_inicio)
        if fecha_fin:
            query += " AND FECHA_MEDICION <= ?"
            params.append(fecha_fin)
        query += " ORDER BY FECHA_MEDICION DESC"

        results = await execute_query(query, params)
        
        caudal_por_tiempo = [{
            "fecha_medicion": str(r.get('fecha_medicion')) if r.get('fecha_medicion') else None, 
            "caudal": r.get('caudal'),
            "caudal_promedio": r.get('caudal_promedio'),
            "caudal_sumado": r.get('caudal_sumado'),
            "totalizador_sumado": r.get('totalizador_sumado'),
            "totalizador_max": r.get('totalizador_max')
        } for r in results] if results else []
        return {"utm_norte": utm_norte, "utm_este": utm_este, "total_registros": len(caudal_por_tiempo), "caudal_por_tiempo": caudal_por_tiempo}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail={"error": str(e)})

@router.get("/puntos/series_de_tiempo/altura_linimetrica", tags=["Series Temporales"])
async def get_altura_linimetrica_por_tiempo_por_punto(
    utm_norte: int = Query(...), utm_este: int = Query(...),
    fecha_inicio: Optional[str] = Query(None), fecha_fin: Optional[str] = Query(None)
):
    try:
        count_query = "SELECT COUNT(*) as total FROM dw.Series_tiempo WHERE UTM_NORTE = ? AND UTM_ESTE = ? AND ALTURA_LIMNIMETRICA IS NOT NULL"
        count_params = [utm_norte, utm_este]
        if fecha_inicio:
            count_query += " AND FECHA_MEDICION >= ?"
            count_params.append(fecha_inicio)
        if fecha_fin:
            count_query += " AND FECHA_MEDICION <= ?"
            count_params.append(fecha_fin)
            
        count_result = await execute_query(count_query, count_params)
        total_count = count_result[0]['total'] if count_result else 0

        query = "SELECT FECHA_MEDICION as fecha_medicion, ALTURA_LIMNIMETRICA as altura_linimetrica FROM dw.Series_tiempo WHERE UTM_NORTE = ? AND UTM_ESTE = ? AND ALTURA_LIMNIMETRICA IS NOT NULL"
        params = [utm_norte, utm_este]
        if fecha_inicio:
            query += " AND FECHA_MEDICION >= ?"
            params.append(fecha_inicio)
        if fecha_fin:
            query += " AND FECHA_MEDICION <= ?"
            params.append(fecha_fin)
        query += " ORDER BY FECHA_MEDICION DESC"

        results = await execute_query(query, params)
        
        altura_por_tiempo = [{"fecha_medicion": str(r.get('fecha_medicion')) if r.get('fecha_medicion') else None, "altura_linimetrica": r.get('altura_linimetrica')} for r in results] if results else []
        return {"utm_norte": utm_norte, "utm_este": utm_este, "total_registros": total_count, "registros_retornados": len(altura_por_tiempo), "altura_por_tiempo": altura_por_tiempo}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail={"error": str(e)})

@router.get("/puntos/series_de_tiempo/nivel_freatico", tags=["Series Temporales"])
async def get_nivel_freatico_por_tiempo_por_punto(
    utm_norte: int = Query(...), utm_este: int = Query(...),
    fecha_inicio: Optional[str] = Query(None), fecha_fin: Optional[str] = Query(None)
):
    try:
        count_query = "SELECT COUNT(*) as total FROM dw.Series_tiempo WHERE UTM_NORTE = ? AND UTM_ESTE = ? AND NIVEL_FREATICO IS NOT NULL"
        count_params = [utm_norte, utm_este]
        if fecha_inicio:
            count_query += " AND FECHA_MEDICION >= ?"
            count_params.append(fecha_inicio)
        if fecha_fin:
            count_query += " AND FECHA_MEDICION <= ?"
            count_params.append(fecha_fin)
            
        count_result = await execute_query(count_query, count_params)
        total_count = count_result[0]['total'] if count_result else 0

        query = "SELECT FECHA_MEDICION as fecha_medicion, NIVEL_FREATICO as nivel_freatico FROM dw.Series_tiempo WHERE UTM_NORTE = ? AND UTM_ESTE = ? AND NIVEL_FREATICO IS NOT NULL"
        params = [utm_norte, utm_este]
        if fecha_inicio:
            query += " AND FECHA_MEDICION >= ?"
            params.append(fecha_inicio)
        if fecha_fin:
            query += " AND FECHA_MEDICION <= ?"
            params.append(fecha_fin)
        query += " ORDER BY FECHA_MEDICION DESC"

        results = await execute_query(query, params)
        
        nivel_por_tiempo = [{"fecha_medicion": str(r.get('fecha_medicion')) if r.get('fecha_medicion') else None, "nivel_freatico": r.get('nivel_freatico')} for r in results] if results else []
        return {"utm_norte": utm_norte, "utm_este": utm_este, "total_registros": total_count, "registros_retornados": len(nivel_por_tiempo), "nivel_por_tiempo": nivel_por_tiempo}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail={"error": str(e)})
