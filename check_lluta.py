import sys
from core.config import setup_config
setup_config()
from core.database import _execute_query_sync as execute_query

print("--- Testing Lluta ---")
try:
    res = execute_query("SELECT TOP 5 UTM_Norte, UTM_Este, SECTOR_SHA FROM dw.Mediciones_full WHERE SECTOR_SHA LIKE '%Lluta%'")
    print("Found in Mediciones_full:", res)
    
    if res:
        norte, este = res[0]['UTM_Norte'], res[0]['UTM_Este']
        res2 = execute_query("SELECT COUNT(*) FROM dw.Datos WHERE UTM_Norte=? AND UTM_Este=?", [norte, este])
        print("Count in dw.Datos:", res2)
        
        res3 = execute_query("SELECT COUNT(*) FROM dw.Series_tiempo WHERE UTM_NORTE=? AND UTM_ESTE=?", [norte, este])
        print("Count in dw.Series_tiempo:", res3)
        
except Exception as e:
    print("Error:", e)
