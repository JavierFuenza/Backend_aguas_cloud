import math
from typing import Optional

def build_full_name(nomb_inf: Optional[str], a_pat_inf: Optional[str], a_mat_inf: Optional[str]) -> str:
    """Build full name efficiently"""
    parts = [part.strip() for part in [nomb_inf, a_pat_inf, a_mat_inf] if part and part.strip()]
    return " ".join(parts) if parts else "Desconocido"

def safe_round(value: Optional[float], decimals: int = 2) -> Optional[float]:
    """Safely round a value, handling None"""
    return round(value, decimals) if value is not None else None

def utm_to_latlon(utm_este: float, utm_norte: float, huso: int = 19) -> tuple:
    """Convertir coordenadas UTM a lat/lon"""
    try:
        central_meridian = -69.0  # Para zona 19

        lat = (utm_norte - 10000000) / 111320
        lon = central_meridian + (utm_este - 500000) / (111320 * math.cos(math.radians(lat)))

        return lon, lat
    except:
        return None, None
