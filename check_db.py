from core.database import execute_query

try:
    print("Testing dw.Datos columns...")
    res = execute_query("SELECT TOP 1 * FROM dw.Datos")
    if res: print("dw.Datos:", list(res[0].keys()))
except Exception as e: print("Error dw.Datos:", e)

try:
    print("\nTesting dw.Mediciones_full columns...")
    res = execute_query("SELECT TOP 1 * FROM dw.Mediciones_full")
    if res: print("dw.Mediciones_full:", list(res[0].keys()))
except Exception as e: print("Error dw.Mediciones_full:", e)

