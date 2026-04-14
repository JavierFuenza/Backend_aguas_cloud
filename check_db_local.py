import sys
sys.path.append(".")
from core.database import execute_query

print("Tables in schema:")
res = execute_query("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'dw'")
print([r['TABLE_NAME'] for r in res] if res else "No tables found")
