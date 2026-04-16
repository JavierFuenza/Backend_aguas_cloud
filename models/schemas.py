from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Pydantic Models for Request/Response validation
class UTMLocation(BaseModel):
    utm_norte: int = Field(..., ge=0, le=10000000, description="Coordenada UTM Norte (metros)")
    utm_este: int = Field(..., ge=0, le=1000000, description="Coordenada UTM Este (metros)")

    class Config:
        json_schema_extra = {
            "example": {
                "utm_norte": 6300000,
                "utm_este": 350000
            }
        }

class PuntoResponse(BaseModel):
    utm_norte: int = Field(..., description="Coordenada UTM Norte")
    utm_este: int = Field(..., description="Coordenada UTM Este")
    huso: int = Field(..., description="Huso UTM (zona)")
    es_pozo_subterraneo: bool = Field(..., description="Indica si es un pozo subterráneo")
    cod_subsubcuenca: Optional[int] = Field(None, description="Código de la subsubcuenca")
    sector_sha: Optional[str] = Field(None, description="Sector Hidráulico de Aprovechamiento Común")
    apr: Optional[bool] = Field(None, description="Agua Potable Rural")
    id_junta: Optional[float] = Field(None, description="ID Junta de Vigilancia")

    class Config:
        json_schema_extra = {
            "example": {
                "utm_norte": 6300000,
                "utm_este": 350000,
                "huso": 19,
                "es_pozo_subterraneo": False,
                "sector_sha": "Lluta",
                "apr": False,
                "id_junta": 1.0
            }
        }

class PuntoInfoResponse(BaseModel):
    utm_norte: int
    utm_este: int
    huso: int
    es_pozo_subterraneo: bool
    codigo: Optional[str] = Field(None, description="Código de obra del punto")
    cod_cuenca: Optional[int] = None
    cod_subcuenca: Optional[int] = None
    cod_subsubcuenca: Optional[int] = None
    nombre_cuenca: Optional[str] = None
    nombre_subcuenca: Optional[str] = None
    nombre_subsubcuenca: Optional[str] = None
    caudal_promedio: Optional[float] = Field(None, description="Caudal promedio en l/s")
    n_mediciones: int = Field(..., description="Número de mediciones registradas")
    sector_sha: Optional[str] = Field(None, description="Sector Hidráulico de Aprovechamiento Común")
    apr: Optional[bool] = Field(None, description="Agua Potable Rural")
    id_junta: Optional[float] = Field(None, description="ID Junta de Vigilancia")
    parte_junta: Optional[bool] = None
    representa_junta: Optional[bool] = None
    canal_transmision: Optional[int] = Field(None, description="Último canal de transmisión reportado")

    class Config:
        json_schema_extra = {
            "example": {
                "utm_norte": 6300000,
                "utm_este": 350000,
                "huso": 19,
                "es_pozo_subterraneo": False,
                "codigo": "PB-1234",
                "cod_cuenca": 101,
                "cod_subcuenca": 10101,
                "nombre_cuenca": "Río Lluta",
                "nombre_subcuenca": "Río Lluta Alto",
                "caudal_promedio": 25.5,
                "n_mediciones": 120,
                "sector_sha": "Lluta",
                "apr": False,
                "id_junta": 1.0,
                "parte_junta": True,
                "representa_junta": False,
                "canal_transmision": 10
            }
        }

class CuencaData(BaseModel):
    cod_cuenca: Optional[int] = Field(None, description="Código de cuenca")
    nom_cuenca: Optional[str] = Field(None, description="Nombre de cuenca")
    cod_subcuenca: Optional[int] = Field(None, description="Código de subcuenca")
    nom_subcuenca: Optional[str] = Field(None, description="Nombre de subcuenca")
    cod_subsubcuenca: Optional[int] = Field(None, description="Código de subsubcuenca")
    nom_subsubcuenca: Optional[str] = Field(None, description="Nombre de subsubcuenca")
    cod_region: Optional[int] = Field(None, description="Código de región")

    class Config:
        json_schema_extra = {
            "example": {
                "cod_cuenca": 101,
                "nom_cuenca": "Río Lluta",
                "cod_subcuenca": 10101,
                "nom_subcuenca": "Río Lluta Alto",
                "cod_subsubcuenca": None,
                "nom_subsubcuenca": None,
                "cod_region": 15
            }
        }

class CuencaStatsResponse(BaseModel):
    cod_cuenca: Optional[int]
    nom_cuenca: Optional[str]
    cod_region: Optional[int]
    cod_subcuenca: Optional[int] = None
    nom_subcuenca: Optional[str] = None
    cod_subsubcuenca: Optional[int] = None
    nom_subsubcuenca: Optional[str] = None
    caudal_promedio: Optional[float] = Field(None, description="Caudal promedio en l/s")
    caudal_minimo: Optional[float] = Field(None, description="Caudal mínimo registrado en l/s")
    caudal_maximo: Optional[float] = Field(None, description="Caudal máximo registrado en l/s")
    total_puntos_unicos: int = Field(..., description="Número de puntos únicos de medición")
    total_mediciones: int = Field(..., description="Número total de mediciones")

    class Config:
        json_schema_extra = {
            "example": {
                "cod_cuenca": 101,
                "nom_cuenca": "Río Lluta",
                "cod_region": 15,
                "cod_subcuenca": None,
                "nom_subcuenca": None,
                "cod_subsubcuenca": None,
                "nom_subsubcuenca": None,
                "caudal_promedio": 45.3,
                "caudal_minimo": 5.2,
                "caudal_maximo": 120.5,
                "total_puntos_unicos": 15,
                "total_mediciones": 1850
            }
        }

class TimeSeriesPoint(BaseModel):
    fecha_medicion: str = Field(..., description="Fecha de medición (ISO format)")
    caudal: Optional[float] = Field(None, description="Caudal promedio medido en l/s (compatibilidad original)")
    caudal_promedio: Optional[float] = Field(None, description="Caudal promedio medido en l/s")
    caudal_sumado: Optional[float] = Field(None, description="Suma de caudales en l/s")
    totalizador_sumado: Optional[int] = Field(None, description="Suma de totalizadores reportados")
    totalizador_max: Optional[int] = Field(None, description="Valor máximo del totalizador reportado")

    class Config:
        json_schema_extra = {
            "example": {
                "fecha_medicion": "2023-06-15",
                "caudal": 35.2,
                "caudal_promedio": 35.2,
                "caudal_sumado": 105.6,
                "totalizador_sumado": 1500000,
                "totalizador_max": 500000
            }
        }

class AlturaTimeSeriesPoint(BaseModel):
    fecha_medicion: str = Field(..., description="Fecha de medición (ISO format)")
    altura_linimetrica: Optional[float] = Field(None, description="Altura limnimétrica en metros")

    class Config:
        json_schema_extra = {
            "example": {
                "fecha_medicion": "2023-06-15",
                "altura_linimetrica": 2.5
            }
        }

class NivelFreaticoTimeSeriesPoint(BaseModel):
    fecha_medicion: str = Field(..., description="Fecha de medición (ISO format)")
    nivel_freatico: Optional[float] = Field(None, description="Nivel freático en metros")

    class Config:
        json_schema_extra = {
            "example": {
                "fecha_medicion": "2023-06-15",
                "nivel_freatico": 15.3
            }
        }

class InformanteResponse(BaseModel):
    nombre_completo: str = Field(..., description="Nombre completo del informante")
    cantidad_reportes: int = Field(..., description="Cantidad de reportes emitidos por el informante")
    ultima_fecha_medicion: Optional[str] = Field(None, description="Última fecha de medición del informante")

    class Config:
        json_schema_extra = {
            "example": {
                "nombre_completo": "Juan Perez",
                "cantidad_reportes": 5,
                "ultima_fecha_medicion": "2023-10-25 14:30:00"
            }
        }

class HealthResponse(BaseModel):
    status: str = Field(..., description="Estado del servicio")
    message: str = Field(..., description="Mensaje descriptivo")
    database: str = Field(..., description="Estado de la conexión a base de datos")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "message": "Water Data API is running",
                "database": "connected"
            }
        }
