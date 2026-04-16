import sys
from core.config import setup_config
setup_config()
from core.database import _execute_query_sync as execute_query

print("--- SHACS CON DATOS ---")
try:
    res = execute_query("""
    SELECT TOP 1 p.COD_SECTOR_SHA, p.SECTOR_SHA
    FROM dw.Series_tiempo s
    JOIN dw.Puntos_Mapa p ON s.UTM_Norte = p.UTM_Norte AND s.UTM_Este = p.UTM_Este
    WHERE p.COD_SECTOR_SHA IS NOT NULL
    """)
    print(res)
except Exception as e:
    print("Error:", e)

