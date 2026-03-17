from main import execute_query
import json
res = execute_query("SELECT TOP 1 * FROM dw.Mediciones_full")
print(json.dumps([{k: str(type(v)) for k, v in r.items()} for r in res], indent=2))
