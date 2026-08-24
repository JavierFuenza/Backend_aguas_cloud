import logging
from fastapi import APIRouter, HTTPException, Query
from core.database import execute_query

router = APIRouter()

TIPO_DERECHO_LABELS = {
    1: "Consuntivo",
    2: "No Consuntivo",
}

MESES = [
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
]

COLUMNAS_CAUDAL_PUNTOS = [f"CAUDAL_{m.upper()}" for m in MESES]
COLUMNAS_CAUDAL_CUENCAS = [f"caudal_{m}_sum" for m in MESES]


@router.get(
    "/puntos/derechos",
    tags=["Derechos de Agua"],
    summary="Derechos de agua de un punto",
    description="Devuelve tipo de derecho, volumen anual y caudal mensual autorizado para un punto (UTM).",
)
async def get_punto_derechos(
    utm_norte: int = Query(..., description="Coordenada UTM Norte"),
    utm_este: int = Query(..., description="Coordenada UTM Este"),
):
    cols = ", ".join(COLUMNAS_CAUDAL_PUNTOS)
    algun_caudal = " OR ".join(f"{col} IS NOT NULL" for col in COLUMNAS_CAUDAL_PUNTOS)
    # Antes se exigía TIPO_DERECHO IS NOT NULL: una obra con volumen anual o
    # caudal mensual registrado pero sin tipo quedaba como "sin derechos".
    # Ahora basta con que exista cualquier dato de derecho.
    # ORDER BY: Puntos_Mapa tiene una fila por canal de transmisión; se prioriza
    # la que trae información de derecho para no caer en una fila vacía.
    query = f"""
    SELECT TOP 1
        TIPO_DERECHO,
        VOLUMEN_ANUAL,
        {cols}
    FROM dw.Puntos_Mapa
    WHERE UTM_Norte = ? AND UTM_Este = ?
      AND (TIPO_DERECHO IS NOT NULL OR VOLUMEN_ANUAL IS NOT NULL OR {algun_caudal})
    ORDER BY
        CASE WHEN TIPO_DERECHO IS NOT NULL THEN 0 ELSE 1 END,
        CASE WHEN VOLUMEN_ANUAL IS NOT NULL THEN 0 ELSE 1 END
    """
    # Respaldo: el proceso que genera Puntos_Mapa pierde el derecho de 1.081 de
    # las 5.571 obras que sí lo traen en Mediciones_full (ej. OB-0202-1, con
    # volumen anual 78 y caudal mensual 2,5, que el visualizador informaba como
    # "sin derechos"). Mientras eso no se corrija en el pipeline, se consulta la
    # tabla de origen sólo cuando la pre-agregada no trae nada. Cuesta ~2-5 s,
    # así que se cachea y nunca se ejecuta si Puntos_Mapa ya respondió.
    query_respaldo = f"""
    SELECT TOP 1
        TIPO_DERECHO,
        VOLUMEN_ANUAL,
        {cols}
    FROM dw.Mediciones_full
    WHERE UTM_NORTE = ? AND UTM_ESTE = ?
      AND (TIPO_DERECHO IS NOT NULL OR VOLUMEN_ANUAL IS NOT NULL OR {algun_caudal})
    """

    try:
        rows = await execute_query(query, params=[utm_norte, utm_este], use_cache=False)
        origen = "puntos_mapa"

        if not rows:
            rows = await execute_query(query_respaldo, params=[utm_norte, utm_este])
            origen = "mediciones_full"
    except Exception as e:
        logging.error(f"Error get_punto_derechos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})

    if not rows:
        raise HTTPException(
            status_code=404, detail="No se encontraron derechos para este punto"
        )

    if origen == "mediciones_full":
        logging.warning(
            f"Derechos de {utm_norte}/{utm_este} recuperados desde Mediciones_full: "
            f"faltan en Puntos_Mapa"
        )

    row = rows[0]
    tipo = row.get("TIPO_DERECHO")
    return {
        "tipo_derecho": tipo,
        # Sin dato ≠ código no reconocido: una obra puede tener volumen y caudal
        # registrados con TIPO_DERECHO nulo (ver respaldo de abajo).
        "tipo_derecho_label": (
            "No informado"
            if tipo is None
            else TIPO_DERECHO_LABELS.get(tipo, "Desconocido")
        ),
        "volumen_anual": row.get("VOLUMEN_ANUAL"),
        "caudal_mensual": {
            mes: row.get(col) for mes, col in zip(MESES, COLUMNAS_CAUDAL_PUNTOS)
        },
        "origen_datos": origen,
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
        return {
            "puntos_con_derechos": 0,
            "volumen_anual_total": 0,
            "caudal_mensual_suma": {m: 0 for m in MESES},
        }

    row = rows[0]
    return {
        "puntos_con_derechos": row.get("puntos_con_derechos", 0),
        "volumen_anual_total": row.get("volumen_anual_total", 0),
        "caudal_mensual_suma": {
            mes: row.get(col, 0) for mes, col in zip(MESES, COLUMNAS_CAUDAL_CUENCAS)
        },
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
        rows = await execute_query(
            query, params=[cod_cuenca, cod_subcuenca], use_cache=False
        )
    except Exception as e:
        logging.error(f"Error get_subcuenca_derechos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})

    if not rows or rows[0].get("puntos_con_derechos") == 0:
        return {
            "puntos_con_derechos": 0,
            "volumen_anual_total": 0,
            "caudal_mensual_suma": {m: 0 for m in MESES},
        }

    row = rows[0]
    return {
        "puntos_con_derechos": row.get("puntos_con_derechos", 0),
        "volumen_anual_total": row.get("volumen_anual_total", 0),
        "caudal_mensual_suma": {
            mes: row.get(col, 0) for mes, col in zip(MESES, COLUMNAS_CAUDAL_CUENCAS)
        },
    }
