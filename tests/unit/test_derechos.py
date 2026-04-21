# Backend_aguas_cloud/tests/unit/test_derechos.py
import pytest
from unittest.mock import patch


@pytest.mark.asyncio
async def test_get_punto_derechos_returns_data():
    mock_row = {
        "TIPO_DERECHO": 1,
        "CAUDAL_ENERO": 19.0, "CAUDAL_FEBRERO": 19.0, "CAUDAL_MARZO": 19.0,
        "CAUDAL_ABRIL": 15.5, "CAUDAL_MAYO": 12.0, "CAUDAL_JUNIO": 9.5,
        "CAUDAL_JULIO": 9.5, "CAUDAL_AGOSTO": 11.0, "CAUDAL_SEPTIEMBRE": 13.5,
        "CAUDAL_OCTUBRE": 16.0, "CAUDAL_NOVIEMBRE": 18.0, "CAUDAL_DICIEMBRE": 19.0,
        "VOLUMEN_ANUAL": 599184,
    }

    with patch("api.routers.derechos.execute_query", return_value=[mock_row]):
        from api.routers.derechos import get_punto_derechos
        result = await get_punto_derechos(utm_norte=6300000, utm_este=350000)

    assert result["tipo_derecho"] == 1
    assert result["tipo_derecho_label"] == "Consuntivo"
    assert result["volumen_anual"] == 599184
    assert result["caudal_mensual"]["enero"] == 19.0
    assert result["caudal_mensual"]["junio"] == 9.5


@pytest.mark.asyncio
async def test_get_punto_derechos_404_when_no_data():
    from fastapi import HTTPException

    with patch("api.routers.derechos.execute_query", return_value=[]):
        from api.routers.derechos import get_punto_derechos
        with pytest.raises(HTTPException) as exc_info:
            await get_punto_derechos(utm_norte=0, utm_este=0)

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_cuenca_derechos_aggregates():
    mock_row = {
        "puntos_con_derechos": 5,
        "volumen_anual_total": 1000000,
        "CAUDAL_ENERO": 50.0, "CAUDAL_FEBRERO": 50.0, "CAUDAL_MARZO": 50.0,
        "CAUDAL_ABRIL": 40.0, "CAUDAL_MAYO": 30.0, "CAUDAL_JUNIO": 20.0,
        "CAUDAL_JULIO": 20.0, "CAUDAL_AGOSTO": 25.0, "CAUDAL_SEPTIEMBRE": 35.0,
        "CAUDAL_OCTUBRE": 40.0, "CAUDAL_NOVIEMBRE": 45.0, "CAUDAL_DICIEMBRE": 50.0,
    }

    with patch("api.routers.derechos.execute_query", return_value=[mock_row]):
        from api.routers.derechos import get_cuenca_derechos
        result = await get_cuenca_derechos(cod_cuenca=401)

    assert result["puntos_con_derechos"] == 5
    assert result["volumen_anual_total"] == 1000000
    assert result["caudal_mensual_suma"]["enero"] == 50.0


@pytest.mark.asyncio
async def test_get_cuenca_derechos_empty_returns_zeros():
    with patch("api.routers.derechos.execute_query", return_value=[{"puntos_con_derechos": 0}]):
        from api.routers.derechos import get_cuenca_derechos
        result = await get_cuenca_derechos(cod_cuenca=999)

    assert result["puntos_con_derechos"] == 0
    assert result["caudal_mensual_suma"]["enero"] == 0


@pytest.mark.asyncio
async def test_tipo_derecho_unknown_label():
    mock_row = {
        "TIPO_DERECHO": 99,
        "CAUDAL_ENERO": 0.0, "CAUDAL_FEBRERO": 0.0, "CAUDAL_MARZO": 0.0,
        "CAUDAL_ABRIL": 0.0, "CAUDAL_MAYO": 0.0, "CAUDAL_JUNIO": 0.0,
        "CAUDAL_JULIO": 0.0, "CAUDAL_AGOSTO": 0.0, "CAUDAL_SEPTIEMBRE": 0.0,
        "CAUDAL_OCTUBRE": 0.0, "CAUDAL_NOVIEMBRE": 0.0, "CAUDAL_DICIEMBRE": 0.0,
        "VOLUMEN_ANUAL": 0,
    }
    with patch("api.routers.derechos.execute_query", return_value=[mock_row]):
        from api.routers.derechos import get_punto_derechos
        result = await get_punto_derechos(utm_norte=1, utm_este=1)

    assert result["tipo_derecho_label"] == "Desconocido"
