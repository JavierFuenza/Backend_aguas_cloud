import logging
from fastapi import APIRouter, HTTPException, Query
from core.database import execute_query

router = APIRouter()

TIPO_DERECHO_LABELS = {
    1: "Consuntivo",
    2: "No Consuntivo",
}

MESES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
]

COLUMNAS_CAUDAL_PUNTOS = [f"CAUDAL_{m.upper()}" for m in MESES]
COLUMNAS_CAUDAL_CUENCAS = [f"caudal_{m}_sum" for m in MESES]


@router.get(
    "/puntos/derechos",
    tags=["Derechos de Agua"],
    summary="Derechos de agua de un punto",
    description="Devuelve tipo de derecho, volumen anual y caudal mensual autorizado para un punto (UTM)."
)
async def get_punto_derechos(
    utm_norte: int = Query(..., description="Coordenada UTM Norte"),
    utm_este: int = Query(..., description="Coordenada UTM Este"),
):
    cols = ", ".join(COLUMNAS_CAUDAL_PUNTOS)
    query = f"""
    SELECT TOP 1
        TIPO_DERECHO,
        VOLUMEN_ANUAL,
        {cols}
    FROM dw.Puntos_Mapa
    WHERE UTM_Norte = ? AND UTM_Este = ?
      AND TIPO_DERECHO IS NOT NULL
    """
    try:
        rows = await execute_query(query, params=[utm_norte, utm_este], use_cache=False)
    except Exception as e:
        logging.error(f"Error get_punto_derechos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})

    if not rows:
        raise HTTPException(status_code=404, detail="No se encontraron derechos para este punto")

    row = rows[0]
    tipo = row.get("TIPO_DERECHO")
    return {
        "tipo_derecho": tipo,
        "tipo_derecho_label": TIPO_DERECHO_LABELS.get(tipo, "Desconocido"),
        "volumen_anual": row.get("VOLUMEN_ANUAL"),
        "caudal_mensual": {
            mes: row.get(col)
            for mes, col in zip(MESES, COLUMNAS_CAUDAL_PUNTOS)
        }
    }


def _build_cuenca_stats_query(where_clause: str) -> str:
    sums = ",\n    ".join(
        f"SUM(ISNULL({col}, 0)) AS {col}" for col in COLUMNAS_CAUDAL_CUENCAS
    )
    return f"""
    SELECT
        SUM(ISNULL(puntos_con_derechos, 0)) AS puntos_con_derechos,
        SUM(ISNULL(volumen_anual_total, 0)) AS volumen_anual_total,
        {sums}
    FROM dw.Cuenca_Stats
    WHERE {where_clause}
    """


@router.get(
    "/cuencas/derechos",
    tags=["Derechos de Agua"],
    summary="Derechos agregados de una cuenca",
)
async def get_cuenca_derechos(
    cod_cuenca: int = Query(..., description="Código de cuenca"),
):
    query = _build_cuenca_stats_query("Cod_Cuenca = ?")
    try:
        rows = await execute_query(query, params=[cod_cuenca], use_cache=False)
    except Exception as e:
        logging.error(f"Error get_cuenca_derechos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})

    if not rows or rows[0].get("puntos_con_derechos") == 0:
        return {"puntos_con_derechos": 0, "volumen_anual_total": 0, "caudal_mensual_suma": {m: 0 for m in MESES}}

    row = rows[0]
    return {
        "puntos_con_derechos": row.get("puntos_con_derechos", 0),
        "volumen_anual_total": row.get("volumen_anual_total", 0),
        "caudal_mensual_suma": {
            mes: row.get(col, 0)
            for mes, col in zip(MESES, COLUMNAS_CAUDAL_CUENCAS)
        }
    }


@router.get(
    "/subcuencas/derechos",
    tags=["Derechos de Agua"],
    summary="Derechos agregados de una subcuenca",
)
async def get_subcuenca_derechos(
    cod_cuenca: int = Query(..., description="Código de cuenca"),
    cod_subcuenca: int = Query(..., description="Código de subcuenca"),
):
    query = _build_cuenca_stats_query("Cod_Cuenca = ? AND Cod_Subcuenca = ?")
    try:
        rows = await execute_query(query, params=[cod_cuenca, cod_subcuenca], use_cache=False)
    except Exception as e:
        logging.error(f"Error get_subcuenca_derechos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})

    if not rows or rows[0].get("puntos_con_derechos") == 0:
        return {"puntos_con_derechos": 0, "volumen_anual_total": 0, "caudal_mensual_suma": {m: 0 for m in MESES}}

    row = rows[0]
    return {
        "puntos_con_derechos": row.get("puntos_con_derechos", 0),
        "volumen_anual_total": row.get("volumen_anual_total", 0),
        "caudal_mensual_suma": {
            mes: row.get(col, 0)
            for mes, col in zip(MESES, COLUMNAS_CAUDAL_CUENCAS)
        }
    }
