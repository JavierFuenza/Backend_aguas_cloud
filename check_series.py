import sys
from core.config import setup_config
setup_config()
from core.database import _execute_query_sync as execute_query

print("--- Testing Series_tiempo ---")
try:
    res = execute_query("SELECT TOP 2 * FROM dw.Series_tiempo")
    print(res)
except Exception as e:
    print("Error:", e)
