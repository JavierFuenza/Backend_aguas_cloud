#!/usr/bin/env python3
"""
Genera frontend-aguas/public/datos/top_usuarios.json.

Por qué existe
--------------
El gráfico rotulado "Top 10 Usuarios" mostraba informantes. No son lo mismo: el
informante carga la medición, el usuario es el titular del derecho. En la cuenca
101 el primer informante es una persona con 282.900 reportes, que son exactamente
las mediciones de Celulosa Arauco y Constitución S.A. El visualizador nombraba a
quien aprieta el botón, no a quien tiene el agua.

El dato de usuario existe en `dw.Mediciones_full` (`NOMBRE_COMPLETO_USUARIO`),
pero no hay dónde consultarlo rápido: `dw.Puntos_Mapa` no trae ninguna columna de
usuario, `dw.Informante` tampoco, y no hay índice por cuenca. Agregarlo para una
sola cuenca contra la tabla de 71,8 millones de filas tarda unos cuatro minutos.

Por eso se resuelve con una sola pasada, acá, fuera de línea, y al navegador le
llega un archivo estático que se descarga solo cuando el panel pide los gráficos.

Esto es un parche consciente. El arreglo de fondo es que el pipeline publique una
`dw.Usuario` o agregue las columnas a `dw.Puntos_Mapa`; mientras eso no exista,
este script mantiene el gráfico diciendo la verdad.

Limitación conocida
-------------------
`dw.Mediciones_full` no trae RUT de usuario, solo el nombre. Dos titulares
homónimos se fusionan en una sola fila. Es otra razón para pedir el modelo de
usuario al pipeline.

Cuándo re-correrlo
------------------
Con cada carga del DW.

Uso
---
    cd Backend_aguas_cloud
    uv run python scripts/generar_top_usuarios.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import setup_config  # noqa: E402

setup_config()

from core.database import _execute_query_sync as execute_query  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ.parent / "frontend-aguas" / "public" / "datos" / "top_usuarios.json"

TOPE = 10

# GROUPING SETS resuelve los dos niveles en una sola pasada. Con dos consultas
# separadas serían dos escaneos completos de la tabla, o sea media hora.
#
# GROUPING(COD_SUBCUENCA) no es opcional: la fila de total de cuenca trae la
# subcuenca en nulo, y las obras que no tienen subcuenca asignada también. Por
# valor son idénticas. Sin este flag, "Celulosa Arauco" aparecía dos veces en el
# Top 10 de la cuenca 101 — una como total y otra como su tramo sin subcuenca.
CONSULTA = """
SELECT
    COD_CUENCA,
    COD_SUBCUENCA,
    COD_SECTOR_SHA,
    GROUPING(COD_SUBCUENCA) AS es_total_cuenca,
    -- Distingue las filas del grouping set de SHAC: ahí COD_CUENCA viene nulo y
    -- es_total_cuenca vale 1, así que sin este flag se confundirían con el total
    -- de una cuenca sin código.
    GROUPING(COD_SECTOR_SHA) AS sin_shac,
    LTRIM(RTRIM(NOMBRE_COMPLETO_USUARIO)) AS usuario,
    COUNT(*) AS reportes,
    COUNT(DISTINCT CONCAT(UTM_NORTE, '|', UTM_ESTE)) AS obras
FROM dw.Mediciones_full
WHERE NOMBRE_COMPLETO_USUARIO IS NOT NULL
  AND LTRIM(RTRIM(NOMBRE_COMPLETO_USUARIO)) <> ''
  AND COD_CUENCA IS NOT NULL
GROUP BY GROUPING SETS (
    (COD_CUENCA, LTRIM(RTRIM(NOMBRE_COMPLETO_USUARIO))),
    (COD_CUENCA, COD_SUBCUENCA, LTRIM(RTRIM(NOMBRE_COMPLETO_USUARIO))),
    (COD_SECTOR_SHA, LTRIM(RTRIM(NOMBRE_COMPLETO_USUARIO)))
)
"""


def main() -> None:
    print("Escaneando dw.Mediciones_full (una pasada, varios minutos)...")
    t = time.time()
    filas = execute_query(CONSULTA, use_cache=False)
    print(f"  {len(filas)} combinaciones en {time.time() - t:.0f}s")

    por_cuenca: dict[str, list] = {}
    por_subcuenca: dict[str, list] = {}
    por_shac: dict[str, list] = {}

    for f in filas:
        entrada = {
            "nombre": f["usuario"],
            "obras": f["obras"] or 0,
            "reportes": f["reportes"] or 0,
        }
        # es_total_cuenca = 1 sólo en las filas que GROUPING SETS generó
        # agregando por encima de la subcuenca. Un nulo en COD_SUBCUENCA con el
        # flag en 0 es una obra sin subcuenca asignada, que va al otro cesto.
        # El grouping set de SHAC va primero: en esas filas COD_CUENCA es nulo y
        # es_total_cuenca vale 1, así que caerían en el cesto equivocado.
        if f["sin_shac"] == 0:
            if f["COD_SECTOR_SHA"] is not None:
                por_shac.setdefault(str(f["COD_SECTOR_SHA"]), []).append(entrada)
        elif f["es_total_cuenca"] == 1:
            por_cuenca.setdefault(str(f["COD_CUENCA"]), []).append(entrada)
        elif f["COD_SUBCUENCA"] is not None:
            por_subcuenca.setdefault(str(f["COD_SUBCUENCA"]), []).append(entrada)

    def recortar(d: dict) -> dict:
        # Mismo criterio que la API de informantes: manda el número de obras y
        # los reportes desempatan (cat. 3.8).
        return {
            k: sorted(v, key=lambda x: (-x["obras"], -x["reportes"]))[:TOPE]
            for k, v in d.items()
        }

    salida = {
        "generado": time.strftime("%Y-%m-%d"),
        "cuenca": recortar(por_cuenca),
        "subcuenca": recortar(por_subcuenca),
        "shac": recortar(por_shac),
    }

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(
        json.dumps(salida, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    kb = SALIDA.stat().st_size / 1024
    print(
        f"  cuencas: {len(salida['cuenca'])}, "
        f"subcuencas: {len(salida['subcuenca'])}, "
        f"shacs: {len(salida['shac'])}"
    )
    print(f"  escrito {SALIDA} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
