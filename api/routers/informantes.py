import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from core.database import execute_query
from utils.helpers import build_full_name
from models.schemas import InformanteResponse

router = APIRouter()

MAX_LIMIT = 10000


@router.get(
    "/informantes",
    tags=["Informantes"],
    response_model=List[InformanteResponse],
    summary="Obtener informantes",
    description="Obtiene la lista de informantes basándose en filtros como coordenadas UTM, cuenca, subcuenca o subsubcuenca.",
)
async def get_informantes(
    utm_norte: Optional[int] = Query(
        None, description="Coordenada UTM Norte en metros"
    ),
    utm_este: Optional[int] = Query(None, description="Coordenada UTM Este en metros"),
    cod_cuenca: Optional[int] = Query(None, description="Código de cuenca"),
    cod_subcuenca: Optional[int] = Query(None, description="Código de subcuenca"),
    cod_subsubcuenca: Optional[int] = Query(None, description="Código de subsubcuenca"),
    limit: Optional[int] = Query(
        100, description="Número máximo de informantes a retornar"
    ),
):
    try:
        logging.info("Consultando informantes con filtros...")

        # dw.Informante trae una fila por (persona, obra): 6.135 filas para 1.409
        # personas. Sin agregar, un informante con 13 obras ocupaba 13 lugares del
        # "Top 10". Se agrupa por persona y se cuentan sus obras distintas.
        query = """
        SELECT
            NOMB_INF,
            A_PAT_INF,
            A_MAT_INF,
            COUNT(DISTINCT CONCAT(UTM_NORTE, '|', UTM_ESTE)) AS CANTIDAD_OBRAS,
            SUM(CANTIDAD_REPORTES) AS CANTIDAD_REPORTES,
            MAX(ULTIMA_FECHA_MEDICION) AS ULTIMA_FECHA_MEDICION
        FROM dw.Informante
        WHERE 1=1
        """
        params = []

        if utm_norte is not None:
            query += " AND UTM_NORTE = ?"
            params.append(utm_norte)
        if utm_este is not None:
            query += " AND UTM_ESTE = ?"
            params.append(utm_este)
        if cod_cuenca is not None:
            query += " AND COD_CUENCA = ?"
            params.append(cod_cuenca)
        if cod_subcuenca is not None:
            query += " AND COD_SUBCUENCA = ?"
            params.append(cod_subcuenca)
        if cod_subsubcuenca is not None:
            query += " AND COD_SUBSUBCUENCA = ?"
            params.append(cod_subsubcuenca)

        # El GROUP BY y el orden van después de los filtros. Antes no había ORDER BY
        # alguno, así que "TOP 10" devolvía diez filas cualesquiera y el ranking no
        # rankeaba nada. La DGA lo pidió por mayor número de obras (cat. 3.8);
        # los reportes desempatan.
        query += """
        GROUP BY NOMB_INF, A_PAT_INF, A_MAT_INF
        ORDER BY CANTIDAD_OBRAS DESC, CANTIDAD_REPORTES DESC
        """

        # Apply limit if necessary (clamped to safe range)
        if limit is not None:
            safe_limit = max(1, min(int(limit), MAX_LIMIT))
            query = query.replace("SELECT\n", f"SELECT TOP {safe_limit}\n", 1)

        results = await execute_query(query, params)

        informantes_out = []
        for r in results:
            nombre = build_full_name(
                r.get("NOMB_INF"), r.get("A_PAT_INF"), r.get("A_MAT_INF")
            )

            # format date if it exists
            ultima_fecha = r.get("ULTIMA_FECHA_MEDICION")
            if ultima_fecha:
                ultima_fecha = str(ultima_fecha)

            informantes_out.append(
                {
                    "nombre_completo": nombre,
                    "cantidad_obras": r.get("CANTIDAD_OBRAS") or 0,
                    "cantidad_reportes": r.get("CANTIDAD_REPORTES") or 0,
                    "ultima_fecha_medicion": ultima_fecha,
                }
            )

        return informantes_out

    except Exception as e:
        logging.error(f"Error in get_informantes: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})
