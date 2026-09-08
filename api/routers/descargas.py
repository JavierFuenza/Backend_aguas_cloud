"""
Descarga de mediciones crudas de una obra, en CSV, Excel o Parquet.

Existe para cubrir la observación 7.1 del seguimiento con la DGA
(`observaciones_visualizador_dga.xlsx`): el Convenio exige que toda la data
obtenida pueda ser descargada por el usuario en formato Excel o texto plano.

Dos decisiones que conviene entender antes de tocar este módulo:

1. **Ninguna consulta lleva `ORDER BY`.** `dw.Mediciones_full` no está ordenada
   por fecha y ordenar 71,8 M de filas en el servidor es prohibitivo. Las filas
   salen en el orden físico de la tabla. Cuando la tabla se reordene por
   `FECHA_MEDICION` aguas arriba, la descarga queda ordenada sola y este módulo
   no necesita cambios.

2. **Ningún índice de la tabla sirve para filtrar por `CODIGO`.** La tabla es un
   heap y tiene dos índices creados fuera del repo —`IX_temp_export` sobre
   (REGION, FECHA_MEDICION) e `IX_Mediciones_full_Punto_Fecha` sobre
   (UTM_NORTE, UTM_ESTE, FECHA_MEDICION)— pero en ninguno `CODIGO` es la
   columna principal, así que cada descarga es un scan completo.
   `sql/indicesv3.sql` agrega `IX_Mediciones_full_Codigo`, que lo vuelve un
   seek; mientras no se ejecute, esperar decenas de segundos por descarga.

El catálogo `COLUMNAS` es la única fuente de verdad de qué se puede descargar.
La UI nunca ve los nombres reales del DW: manda `clave`, y acá se traduce a la
expresión SQL. Eso también es lo que impide inyección en la lista de columnas,
porque nada de lo que llega del cliente entra concatenado en el SQL.
"""

import csv
import io
import logging
import os
import tempfile
from datetime import date, datetime
from decimal import Decimal
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from core.database import execute_query, get_db_connection, return_db_connection

router = APIRouter()

TABLA = "dw.Mediciones_full"
FILAS_PREVIEW = 10
TAMANO_LOTE = 5000

# Excel es el único formato con tope. El límite del formato .xlsx es 1.048.576
# filas, pero mucho antes de eso el archivo se vuelve inmanejable y la Function
# corre riesgo de quedarse sin memoria (la instancia tiene 2048 MB). 100.000
# filas son del orden de 5-10 MB, y cubren de sobra la obra promedio: 71,8 M de
# mediciones repartidas en 6.098 obras dan ~11.800 filas por obra.
LIMITE_FILAS_EXCEL = 100_000

# CSV y Parquet se generan sin tope: el Convenio habla de "toda la data
# obtenida" y ambos formatos aguantan el volumen sin problema.
FORMATOS = {
    "csv": {
        "extension": "csv",
        "media_type": "text/csv; charset=utf-8",
        "limite_filas": None,
        "etiqueta": "CSV (texto plano)",
    },
    "excel": {
        "extension": "xlsx",
        "media_type": (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        "limite_filas": LIMITE_FILAS_EXCEL,
        "etiqueta": "Excel (.xlsx)",
    },
    "parquet": {
        "extension": "parquet",
        "media_type": "application/vnd.apache.parquet",
        "limite_filas": None,
        "etiqueta": "Parquet",
    },
}

# Las columnas del DW están correctamente tipadas (verificado contra la base el
# 2026-09-07): CAUDAL y las demás medidas son decimal, FECHA_MEDICION es
# datetime2, UTM_* y TOTALIZADOR son bigint. Ojo que docs/ARCHITECTURE.md afirma
# lo contrario ("todas las columnas entran como string"); está desactualizado.
#
# Los TRY_CAST que quedan no son defensivos contra strings sucios sino
# conversiones deliberadas: decimal -> float para que pyodbc devuelva float
# nativo y no Decimal, que ni Excel ni Parquet escriben como número.
_SQL_NATURALEZA = (
    "CASE WHEN TRY_CAST(NATURALEZA AS int) = 1 THEN 'Subterránea' "
    "ELSE 'Superficial' END"
)
_SQL_TRANSMISION = (
    "CASE TRY_CAST(CANAL_TRANSMISION AS int) "
    "WHEN 0 THEN 'Online' "
    "WHEN 1 THEN 'Por archivo' "
    "WHEN 2 THEN 'Por formulario digital' ELSE NULL END"
)
_SQL_TIPO_DERECHO = (
    "CASE TRY_CAST(TIPO_DERECHO AS int) "
    "WHEN 1 THEN 'Consuntivo' WHEN 2 THEN 'No Consuntivo' ELSE NULL END"
)
_SQL_APR = "CASE WHEN TRY_CAST(APR AS int) = 1 THEN 'Sí' ELSE 'No' END"

# Las etiquetas siguen la nomenclatura que la DGA pidió en el seguimiento de
# observaciones: "Naturaleza de la obra" (1.3), "Tipo de transmisión" con sus
# tres valores (2.1 y 5.4), "SHAC" en vez de "Sector SHAC" (5.2) y "Usuario de
# la obra" (6.1). No se expone el informante: la observación 5.6 pidió sacarlo
# porque identifica a quien reporta, no al titular de la obra.
COLUMNAS: List[Dict] = [
    {
        "clave": "codigo_obra",
        "etiqueta": "Código de obra",
        "sql": "CODIGO",
        "grupo": "Identificación",
        "tipo": "texto",
        "por_defecto": True,
        "descripcion": (
            "Código con que la DGA identifica la obra de captación, en formato "
            "OB-XXXX-N. Es el mismo que aparece en el buscador del visualizador."
        ),
    },
    {
        "clave": "fecha_medicion",
        "etiqueta": "Fecha de medición",
        "sql": "FECHA_MEDICION",
        "grupo": "Identificación",
        "tipo": "fecha",
        "por_defecto": True,
        "descripcion": (
            "Fecha a la que corresponde la medición reportada por el titular "
            "del derecho."
        ),
    },
    {
        "clave": "usuario_obra",
        "etiqueta": "Usuario de la obra",
        "sql": "NOMBRE_COMPLETO_USUARIO",
        "grupo": "Identificación",
        "tipo": "texto",
        "por_defecto": True,
        "descripcion": (
            "Titular del derecho de aprovechamiento asociado a la obra. No "
            "confundir con el informante, que es quien carga el dato."
        ),
    },
    {
        "clave": "caudal",
        "etiqueta": "Caudal (L/s)",
        "sql": "TRY_CAST(CAUDAL AS float)",
        "grupo": "Mediciones",
        "tipo": "numero",
        "por_defecto": True,
        "descripcion": (
            "Caudal instantáneo extraído en la fecha de la medición, en litros "
            "por segundo. Un cero es un dato válido: significa que la obra no "
            "extrajo agua."
        ),
    },
    {
        "clave": "totalizador",
        "etiqueta": "Totalizador (m³)",
        "sql": "TRY_CAST(TOTALIZADOR AS float)",
        "grupo": "Mediciones",
        "tipo": "numero",
        "por_defecto": True,
        "descripcion": (
            "Lectura acumulada del flujómetro. Puede volver a cero si el equipo "
            "se reemplaza o se reinicia, así que la diferencia entre dos "
            "lecturas sólo es válida si la posterior es mayor que la anterior."
        ),
    },
    {
        "clave": "altura_limnimetrica",
        "etiqueta": "Altura limnimétrica (m)",
        "sql": "TRY_CAST(ALTURA_LIMNIMETRICA AS float)",
        "grupo": "Mediciones",
        "tipo": "numero",
        "por_defecto": True,
        "descripcion": (
            "Altura de la lámina de agua medida en la obra, en metros. Aplica a "
            "captaciones superficiales."
        ),
    },
    {
        "clave": "nivel_freatico",
        "etiqueta": "Nivel freático (m)",
        "sql": "TRY_CAST(NIVEL_FREATICO AS float)",
        "grupo": "Mediciones",
        "tipo": "numero",
        "por_defecto": True,
        "descripcion": (
            "Profundidad a la que se encuentra el agua en el pozo, en metros. "
            "Sólo existe para captaciones subterráneas."
        ),
    },
    {
        "clave": "naturaleza",
        "etiqueta": "Naturaleza de la obra",
        "sql": _SQL_NATURALEZA,
        "grupo": "Características",
        "tipo": "texto",
        "por_defecto": True,
        "descripcion": "Si la captación es Superficial o Subterránea.",
    },
    {
        "clave": "tipo_transmision",
        "etiqueta": "Tipo de transmisión",
        "sql": _SQL_TRANSMISION,
        "grupo": "Características",
        "tipo": "texto",
        "por_defecto": False,
        "descripcion": (
            "Vía por la que el titular envió el dato: Online, Por archivo o Por "
            "formulario digital."
        ),
    },
    {
        "clave": "apr",
        "etiqueta": "APR",
        "sql": _SQL_APR,
        "grupo": "Características",
        "tipo": "texto",
        "por_defecto": False,
        "descripcion": (
            "Indica si la obra pertenece a un sistema de Agua Potable Rural."
        ),
    },
    {
        "clave": "utm_norte",
        "etiqueta": "UTM Norte",
        "sql": "UTM_NORTE",
        "grupo": "Ubicación",
        "tipo": "entero",
        "por_defecto": False,
        "descripcion": "Coordenada UTM Norte, Datum WGS84, Huso 19 Sur.",
    },
    {
        "clave": "utm_este",
        "etiqueta": "UTM Este",
        "sql": "UTM_ESTE",
        "grupo": "Ubicación",
        "tipo": "entero",
        "por_defecto": False,
        "descripcion": "Coordenada UTM Este, Datum WGS84, Huso 19 Sur.",
    },
    {
        "clave": "huso",
        "etiqueta": "Huso UTM",
        "sql": "HUSO",
        "grupo": "Ubicación",
        "tipo": "entero",
        "por_defecto": False,
        "descripcion": "Huso del sistema de coordenadas UTM de la obra.",
    },
    {
        "clave": "region",
        "etiqueta": "Región",
        "sql": "REGION",
        "grupo": "Ubicación",
        "tipo": "entero",
        "por_defecto": False,
        "descripcion": "Código de la región administrativa donde está la obra.",
    },
    {
        "clave": "provincia",
        "etiqueta": "Provincia",
        "sql": "PROVINCIA",
        "grupo": "Ubicación",
        "tipo": "entero",
        "por_defecto": False,
        "descripcion": "Código de la provincia donde está la obra.",
    },
    {
        "clave": "comuna",
        "etiqueta": "Comuna",
        "sql": "COMUNA",
        "grupo": "Ubicación",
        "tipo": "entero",
        "por_defecto": False,
        "descripcion": "Código de la comuna donde está la obra.",
    },
    {
        "clave": "cuenca",
        "etiqueta": "Cuenca",
        "sql": "NOM_CUENCA",
        "grupo": "Ubicación",
        "tipo": "texto",
        "por_defecto": False,
        "descripcion": "Nombre de la cuenca hidrográfica que contiene la obra.",
    },
    {
        "clave": "subcuenca",
        "etiqueta": "Subcuenca",
        "sql": "NOM_SUBCUENCA",
        "grupo": "Ubicación",
        "tipo": "texto",
        "por_defecto": False,
        "descripcion": "Nombre de la subcuenca que contiene la obra.",
    },
    {
        "clave": "subsubcuenca",
        "etiqueta": "Subsubcuenca",
        "sql": "NOM_SUBSUBCUENCA",
        "grupo": "Ubicación",
        "tipo": "texto",
        "por_defecto": False,
        "descripcion": "Nombre de la subsubcuenca que contiene la obra.",
    },
    {
        "clave": "shac",
        "etiqueta": "SHAC",
        "sql": "SECTOR_SHA",
        "grupo": "Ubicación",
        "tipo": "texto",
        "por_defecto": False,
        "descripcion": (
            "Nombre del Sector Hidrogeológico de Aprovechamiento Común. Sólo "
            "tiene sentido para captaciones subterráneas."
        ),
    },
    {
        "clave": "cod_shac",
        "etiqueta": "Código SHAC",
        "sql": "COD_SECTOR_SHA",
        "grupo": "Ubicación",
        "tipo": "entero",
        "por_defecto": False,
        "descripcion": "Código del Sector Hidrogeológico de Aprovechamiento Común.",
    },
    {
        "clave": "junta_vigilancia",
        "etiqueta": "Junta de vigilancia (código)",
        "sql": "TRY_CAST(ID_JUNTA AS bigint)",
        "grupo": "Administración",
        "tipo": "entero",
        "por_defecto": False,
        "descripcion": (
            "Identificador de la junta de vigilancia que administra el cauce. "
            "La base entregada por la DGA trae el código y no el nombre."
        ),
    },
    {
        "clave": "tipo_derecho",
        "etiqueta": "Tipo de derecho",
        "sql": _SQL_TIPO_DERECHO,
        "grupo": "Derechos",
        "tipo": "texto",
        "por_defecto": False,
        "descripcion": (
            "Consuntivo si el agua se consume sin devolverla al cauce, No "
            "Consuntivo si se restituye."
        ),
    },
    {
        "clave": "volumen_anual",
        "etiqueta": "Volumen anual autorizado (m³/año)",
        "sql": "VOLUMEN_ANUAL",
        "grupo": "Derechos",
        "tipo": "entero",
        "por_defecto": False,
        "descripcion": (
            "Volumen máximo que el derecho de aprovechamiento autoriza extraer "
            "en un año."
        ),
    },
]

COLUMNAS_POR_CLAVE = {c["clave"]: c for c in COLUMNAS}
CLAVES_POR_DEFECTO = [c["clave"] for c in COLUMNAS if c["por_defecto"]]


def _catalogo_publico() -> List[Dict]:
    """El catálogo tal como lo ve la UI: sin la expresión SQL."""
    return [{k: v for k, v in c.items() if k != "sql"} for c in COLUMNAS]


def _resolver_columnas(columnas: Optional[str]) -> List[Dict]:
    if not columnas:
        return [COLUMNAS_POR_CLAVE[c] for c in CLAVES_POR_DEFECTO]

    claves = [c.strip() for c in columnas.split(",") if c.strip()]
    if not claves:
        raise HTTPException(
            status_code=400, detail="Debe seleccionar al menos una columna."
        )

    desconocidas = [c for c in claves if c not in COLUMNAS_POR_CLAVE]
    if desconocidas:
        raise HTTPException(
            status_code=400,
            detail=f"Columnas no reconocidas: {', '.join(desconocidas)}",
        )

    # dict.fromkeys respeta el orden y elimina repetidas
    return [COLUMNAS_POR_CLAVE[c] for c in dict.fromkeys(claves)]


def _construir_where(
    codigo_obra: str, fecha_inicio: Optional[str], fecha_fin: Optional[str]
):
    # Igualdad exacta y no LIKE: la observación 1.7 del seguimiento reportó que
    # buscar "OB-0202-1" traía también OB-0202-1x. Mismo criterio que /puntos.
    where = "WHERE CODIGO = ?"
    params: List = [codigo_obra.strip()]

    if fecha_inicio:
        where += " AND FECHA_MEDICION >= ?"
        params.append(fecha_inicio)
    if fecha_fin:
        where += " AND FECHA_MEDICION <= ?"
        params.append(fecha_fin)

    return where, params


def _iterar_filas(sql: str, params: List, limite: Optional[int] = None):
    """
    Recorre el resultado por lotes en vez de traerlo entero a memoria.

    No usa `execute_query` a propósito: esa función hace `fetchall()` y cachea el
    resultado, que es justo lo que no queremos para una descarga que puede tener
    cientos de miles de filas.
    """
    conn = get_db_connection()
    cursor = None
    emitidas = 0
    try:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        while True:
            lote = cursor.fetchmany(TAMANO_LOTE)
            if not lote:
                break
            for fila in lote:
                if limite is not None and emitidas >= limite:
                    return
                yield fila
                emitidas += 1
    finally:
        if cursor is not None:
            try:
                # Cerrar el cursor descarta el resto del result set y deja la
                # conexión reutilizable aunque hayamos cortado por el límite.
                cursor.close()
            except Exception:
                logging.warning("No se pudo cerrar el cursor de descarga")
        return_db_connection(conn)


def _valor_csv(valor):
    """Normaliza para CSV, que es texto y necesita una fecha legible."""
    if valor is None:
        return ""
    if isinstance(valor, datetime):
        # ISO con espacio: Excel lo reconoce como fecha al abrir el CSV.
        return valor.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(valor, date):
        return valor.strftime("%Y-%m-%d")
    if isinstance(valor, Decimal):
        return float(valor)
    if isinstance(valor, (int, float, str)):
        return valor
    return str(valor)


def _valor_excel(valor):
    """
    Normaliza para xlsxwriter.

    Las fechas se pasan como objeto, no como texto: write() las despacha a
    write_datetime y quedan como fecha real en la planilla, ordenable y
    filtrable. Convertirlas a str acá las dejaría como texto.
    """
    if valor is None:
        return ""
    if isinstance(valor, (datetime, date)):
        return valor
    if isinstance(valor, Decimal):
        return float(valor)
    if isinstance(valor, (int, float, str)):
        return valor
    return str(valor)


def _generar_csv(sql: str, params: List, seleccion: List[Dict]):
    buffer = io.StringIO()
    escritor = csv.writer(buffer, lineterminator="\n")

    # BOM para que Excel abra el CSV con los acentos correctos al hacer
    # doble click, que es como lo va a abrir la mayoría.
    yield "﻿".encode("utf-8")

    escritor.writerow([c["etiqueta"] for c in seleccion])
    yield buffer.getvalue().encode("utf-8")
    buffer.seek(0)
    buffer.truncate(0)

    for fila in _iterar_filas(sql, params):
        escritor.writerow([_valor_csv(v) for v in fila])
        if buffer.tell() > 64 * 1024:
            yield buffer.getvalue().encode("utf-8")
            buffer.seek(0)
            buffer.truncate(0)

    if buffer.tell():
        yield buffer.getvalue().encode("utf-8")


def _generar_excel(sql: str, params: List, seleccion: List[Dict]) -> str:
    import xlsxwriter

    fd, ruta = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)

    # constant_memory mantiene en RAM sólo la fila en curso; sin esto un archivo
    # de 100.000 filas se construye entero en memoria.
    libro = xlsxwriter.Workbook(
        ruta, {"constant_memory": True, "default_date_format": "yyyy-mm-dd"}
    )
    hoja = libro.add_worksheet("Mediciones")
    negrita = libro.add_format({"bold": True})

    for col, columna in enumerate(seleccion):
        hoja.write(0, col, columna["etiqueta"], negrita)

    fila_actual = 0
    for fila_actual, fila in enumerate(
        _iterar_filas(sql, params, limite=LIMITE_FILAS_EXCEL), start=1
    ):
        for col, valor in enumerate(fila):
            hoja.write(fila_actual, col, _valor_excel(valor))

    libro.close()
    return ruta


def _generar_parquet(sql: str, params: List, seleccion: List[Dict]) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    tipos = {
        "numero": pa.float64(),
        "entero": pa.int64(),
        "fecha": pa.timestamp("us"),
        "texto": pa.string(),
    }
    esquema = pa.schema(
        [pa.field(c["etiqueta"], tipos.get(c["tipo"], pa.string())) for c in seleccion]
    )

    fd, ruta = tempfile.mkstemp(suffix=".parquet")
    os.close(fd)

    escritor = pq.ParquetWriter(ruta, esquema, compression="snappy")
    try:
        lote: List[List] = []
        for fila in _iterar_filas(sql, params):
            lote.append(list(fila))
            if len(lote) >= TAMANO_LOTE:
                _escribir_lote_parquet(escritor, esquema, seleccion, lote)
                lote = []
        if lote:
            _escribir_lote_parquet(escritor, esquema, seleccion, lote)
    finally:
        escritor.close()

    return ruta


def _escribir_lote_parquet(escritor, esquema, seleccion, lote):
    import pyarrow as pa

    columnas = []
    for indice, columna in enumerate(seleccion):
        valores = [fila[indice] for fila in lote]
        tipo = columna["tipo"]
        if tipo == "numero":
            valores = [
                float(v) if isinstance(v, (int, float, Decimal)) else None
                for v in valores
            ]
        elif tipo == "entero":
            valores = [
                int(v) if isinstance(v, (int, Decimal)) else None for v in valores
            ]
        elif tipo == "fecha":
            # pa.timestamp acepta datetime; date hay que promoverlo.
            valores = [
                (
                    v
                    if isinstance(v, datetime)
                    else (
                        datetime(v.year, v.month, v.day)
                        if isinstance(v, date)
                        else None
                    )
                )
                for v in valores
            ]
        else:
            valores = [None if v is None else str(v) for v in valores]
        columnas.append(valores)

    escritor.write_table(
        pa.Table.from_arrays(
            [pa.array(c, type=f.type) for c, f in zip(columnas, esquema)],
            schema=esquema,
        )
    )


def _stream_archivo(ruta: str):
    try:
        with open(ruta, "rb") as archivo:
            while True:
                trozo = archivo.read(64 * 1024)
                if not trozo:
                    break
                yield trozo
    finally:
        try:
            os.unlink(ruta)
        except OSError:
            logging.warning(f"No se pudo borrar el temporal {ruta}")


@router.get("/mediciones/columnas", tags=["Descarga de Datos"])
async def get_columnas_descargables():
    """
    Catálogo de columnas que se pueden descargar, con la descripción que la UI
    muestra en el botón de información.

    Es la única fuente de verdad de los alias: el frontend no conoce los nombres
    reales de las columnas del data warehouse.
    """
    return {
        "columnas": _catalogo_publico(),
        "por_defecto": CLAVES_POR_DEFECTO,
        "formatos": [
            {
                "clave": clave,
                "etiqueta": cfg["etiqueta"],
                "extension": cfg["extension"],
                "limite_filas": cfg["limite_filas"],
            }
            for clave, cfg in FORMATOS.items()
        ],
    }


@router.get("/mediciones/preview", tags=["Descarga de Datos"])
async def get_preview_mediciones(
    codigo_obra: str = Query(..., description="Código de obra, coincidencia exacta"),
    fecha_inicio: Optional[str] = Query(
        None, description="Fecha inicial del rango (YYYY-MM-DD)"
    ),
    fecha_fin: Optional[str] = Query(
        None, description="Fecha final del rango (YYYY-MM-DD)"
    ),
    columnas: Optional[str] = Query(
        None, description="Claves de columna separadas por coma"
    ),
):
    """
    Primeras filas de lo que se va a descargar, más el total y el rango de
    fechas disponible para la obra.

    El total es lo que permite avisar al usuario si Excel va a truncar el
    archivo antes de que se ponga a esperar la descarga.
    """
    seleccion = _resolver_columnas(columnas)
    where, params = _construir_where(codigo_obra, fecha_inicio, fecha_fin)

    proyeccion = ", ".join(f"{c['sql']} AS [{c['etiqueta']}]" for c in seleccion)

    try:
        # Sin ORDER BY: ver el docstring del módulo.
        filas = await execute_query(
            f"SELECT TOP {FILAS_PREVIEW} {proyeccion} FROM {TABLA} {where}",
            params,
        )

        resumen = await execute_query(
            f"""
            SELECT
                COUNT(*) AS total_filas,
                MIN(FECHA_MEDICION) AS primera_fecha,
                MAX(FECHA_MEDICION) AS ultima_fecha
            FROM {TABLA} {where}
            """,
            params,
        )
    except Exception as e:
        logging.error(f"Error al generar el preview de {codigo_obra}: {e}")
        raise HTTPException(
            status_code=500, detail="No se pudo consultar las mediciones de la obra."
        )

    total = resumen[0]["total_filas"] if resumen else 0
    primera = resumen[0]["primera_fecha"] if resumen else None
    ultima = resumen[0]["ultima_fecha"] if resumen else None

    return {
        "codigo_obra": codigo_obra,
        "total_filas": total,
        "rango_fechas": {
            "primera": str(primera) if primera else None,
            "ultima": str(ultima) if ultima else None,
        },
        "filtro_aplicado": {"fecha_inicio": fecha_inicio, "fecha_fin": fecha_fin},
        "columnas": [
            {"clave": c["clave"], "etiqueta": c["etiqueta"], "tipo": c["tipo"]}
            for c in seleccion
        ],
        "filas": [
            {k: (str(v) if v is not None else None) for k, v in fila.items()}
            for fila in filas
        ],
        "limite_excel": LIMITE_FILAS_EXCEL,
        "excede_limite_excel": total > LIMITE_FILAS_EXCEL,
    }


@router.get("/mediciones/descarga", tags=["Descarga de Datos"])
async def descargar_mediciones(
    codigo_obra: str = Query(..., description="Código de obra, coincidencia exacta"),
    formato: str = Query("csv", description="csv, excel o parquet"),
    fecha_inicio: Optional[str] = Query(
        None, description="Fecha inicial del rango (YYYY-MM-DD)"
    ),
    fecha_fin: Optional[str] = Query(
        None, description="Fecha final del rango (YYYY-MM-DD)"
    ),
    columnas: Optional[str] = Query(
        None, description="Claves de columna separadas por coma"
    ),
):
    """
    Descarga las mediciones de una obra en CSV, Excel o Parquet.

    CSV y Parquet no tienen tope. Excel se corta en `LIMITE_FILAS_EXCEL`; el
    preview avisa antes si la obra excede ese número.
    """
    formato = formato.lower().strip()
    if formato not in FORMATOS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado. Use uno de: {', '.join(FORMATOS)}.",
        )

    seleccion = _resolver_columnas(columnas)
    where, params = _construir_where(codigo_obra, fecha_inicio, fecha_fin)
    proyeccion = ", ".join(c["sql"] for c in seleccion)
    sql = f"SELECT {proyeccion} FROM {TABLA} {where}"

    cfg = FORMATOS[formato]
    seguro = codigo_obra.strip().replace("/", "-").replace("\\", "-")
    nombre = f"mediciones_{seguro}.{cfg['extension']}"
    cabeceras = {"Content-Disposition": f'attachment; filename="{nombre}"'}

    logging.info(f"Descarga {formato} de {codigo_obra} ({len(seleccion)} columnas)")

    try:
        if formato == "csv":
            return StreamingResponse(
                _generar_csv(sql, params, seleccion),
                media_type=cfg["media_type"],
                headers=cabeceras,
            )

        constructor = _generar_excel if formato == "excel" else _generar_parquet
        ruta = constructor(sql, params, seleccion)
        return StreamingResponse(
            _stream_archivo(ruta),
            media_type=cfg["media_type"],
            headers=cabeceras,
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error al generar la descarga {formato} de {codigo_obra}: {e}")
        raise HTTPException(
            status_code=500, detail="No se pudo generar el archivo de descarga."
        )
