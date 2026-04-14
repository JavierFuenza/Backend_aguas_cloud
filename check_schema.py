import sys
from core.config import setup_config
setup_config()
from core.database import execute_query

print("--- TABLES ---")
try:
    res = execute_query("SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'")
    for r in res:
        if r['TABLE_SCHEMA'] == 'dw': print(r['TABLE_NAME'])
except Exception as e: print("Error:", e)

print("--- COLUMNS IN dw.Series_tiempo ---")
try:
    res = execute_query("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='Series_tiempo'")
    print([r['COLUMN_NAME'] for r in res])
except Exception as e: print("Error:", e)

