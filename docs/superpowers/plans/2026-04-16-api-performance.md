# API Performance Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve API throughput and response latency through three independent changes: async DB execution, cache/compression tuning, and fixing broken Azure SQL queries.

**Architecture:** (A) Wrap synchronous pyodbc calls in `asyncio.run_in_executor` so FastAPI's event loop is never blocked. (B) Add per-query TTL to cache + GZip middleware. (C) Replace Synapse-specific DMV queries with Azure SQL equivalents in `system.py`.

**Tech Stack:** FastAPI, pyodbc, asyncio, Starlette GZipMiddleware (bundled with fastapi)

---

## File Map

| File | Change |
|---|---|
| `core/database.py` | Rename sync fn to `_execute_query_sync`, add async `execute_query`, remove `SELECT 1` health check |
| `core/cache_manager.py` | Add `ttl` param to `is_cache_valid` and `cache_timestamps` stores per-key TTL |
| `main.py` | Add `GZipMiddleware` |
| `api/routers/system.py` | Replace `sys.dm_pdw_nodes_db_partition_stats` with Azure SQL queries |
| `api/routers/atlas.py` | `execute_query(...)` → `await execute_query(...)` |
| `api/routers/informantes.py` | `execute_query(...)` → `await execute_query(...)` |
| `api/routers/cuencas_hidrograficas.py` | `execute_query(...)` → `await execute_query(...)` |
| `api/routers/series_temporales.py` | `execute_query(...)` → `await execute_query(...)` |
| `api/routers/cache_y_rendimiento.py` | `execute_query(...)` → `await execute_query(...)` |
| `api/routers/puntos_de_medicion.py` | `execute_query(...)` → `await execute_query(...)` |
| `tests/unit/test_database.py` | New: unit tests for async execute_query |
| `tests/unit/test_cache_manager.py` | New: unit tests for per-key TTL |

---

## Task 1: Fix event-loop blocking — async execute_query

**Files:**
- Modify: `core/database.py`
- Create: `tests/unit/test_database.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_database.py`:

```python
import asyncio
import pytest
from unittest.mock import patch, MagicMock


def test_execute_query_is_coroutine():
    from core.database import execute_query
    import inspect
    assert inspect.iscoroutinefunction(execute_query)


@pytest.mark.asyncio
async def test_execute_query_returns_list_of_dicts():
    mock_row = (42,)
    mock_cursor = MagicMock()
    mock_cursor.description = [("count",)]
    mock_cursor.fetchall.return_value = [mock_row]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch("core.database.get_db_connection", return_value=mock_conn), \
         patch("core.database.return_db_connection"):
        from core.database import execute_query
        result = await execute_query("SELECT 1", use_cache=False)

    assert result == [{"count": 42}]


@pytest.mark.asyncio
async def test_execute_query_retries_on_dead_connection():
    dead_conn = MagicMock()
    dead_conn.cursor.side_effect = Exception("Connection closed")
    fresh_cursor = MagicMock()
    fresh_cursor.description = [("val",)]
    fresh_cursor.fetchall.return_value = [(1,)]
    fresh_conn = MagicMock()
    fresh_conn.cursor.return_value = fresh_cursor

    call_count = 0
    def get_conn():
        nonlocal call_count
        call_count += 1
        return dead_conn if call_count == 1 else fresh_conn

    with patch("core.database.get_db_connection", side_effect=get_conn), \
         patch("core.database.create_db_connection", return_value=fresh_conn), \
         patch("core.database.return_db_connection"):
        from core.database import execute_query
        result = await execute_query("SELECT 1", use_cache=False)

    assert result == [{"val": 1}]
```

- [ ] **Step 2: Install pytest-asyncio, run tests to confirm they fail**

```bash
uv add --dev pytest-asyncio
uv run pytest tests/unit/test_database.py -v
```

Expected: `ImportError` or `AttributeError` — `execute_query` is not yet a coroutine.

- [ ] **Step 3: Rewrite `core/database.py`**

Replace the file content with:

```python
import asyncio
import os
import time
import logging
import pyodbc
from queue import Queue, Empty
from typing import List, Dict, Optional
import threading

from core.cache_manager import memory_cache, cache_timestamps, get_cache_key, is_cache_valid

connection_pool: Optional[Queue] = None
POOL_SIZE = 10
pool_lock = threading.Lock()


def create_db_connection():
    server = os.getenv('SYNAPSE_SERVER')
    database = os.getenv('SYNAPSE_DATABASE')
    username = os.getenv('SYNAPSE_USERNAME')
    password = os.getenv('SYNAPSE_PASSWORD')
    connection_string = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={server};DATABASE={database};UID={username};PWD={password};"
        "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
    )
    return pyodbc.connect(connection_string)


def get_db_connection():
    global connection_pool
    if connection_pool is None:
        return create_db_connection()
    try:
        return connection_pool.get(timeout=5.0)
    except Empty:
        return create_db_connection()


def return_db_connection(conn):
    global connection_pool
    if connection_pool is None:
        conn.close()
        return
    try:
        if not connection_pool.full():
            connection_pool.put_nowait(conn)
        else:
            conn.close()
    except Exception:
        conn.close()


def _execute_query_sync(query: str, params: List = None, use_cache: bool = True) -> List[Dict]:
    if use_cache:
        cache_key = get_cache_key(query, params)
        if cache_key in memory_cache and is_cache_valid(cache_key):
            logging.info(f"Cache hit for query: {query[:50]}...")
            return memory_cache[cache_key]

    conn = get_db_connection()
    try:
        start_time = time.time()
        try:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
        except Exception:
            # Dead connection — get a fresh one and retry once
            conn.close()
            conn = create_db_connection()
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

        columns = [col[0] for col in cursor.description]
        results = cursor.fetchall()
        result_list = [dict(zip(columns, row)) for row in results]

        execution_time = time.time() - start_time
        logging.info(f"Query executed in {execution_time:.3f}s, returned {len(result_list)} rows")

        if use_cache and result_list:
            cache_key = get_cache_key(query, params)
            memory_cache[cache_key] = result_list
            cache_timestamps[cache_key] = time.time()

        return result_list
    finally:
        return_db_connection(conn)


async def execute_query(query: str, params: List = None, use_cache: bool = True) -> List[Dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _execute_query_sync, query, params, use_cache
    )
```

- [ ] **Step 4: Run tests — confirm they pass**

```bash
uv run pytest tests/unit/test_database.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Update all routers to `await execute_query(...)`**

All 7 router files import and call `execute_query(...)` without `await`. Because `execute_query` is now async, every call site must add `await`.

Run this sed command to bulk-update all routers:

```bash
sed -i 's/= execute_query(/= await execute_query(/g' api/routers/*.py
```

Then verify no bare `execute_query(` calls remain (excluding imports):

```bash
grep -n "execute_query(" api/routers/*.py | grep -v "await execute_query" | grep -v "from core"
```

Expected: no output (all calls are now `await`-ed).

- [ ] **Step 6: Commit**

```bash
git add core/database.py api/routers/*.py tests/unit/test_database.py
git commit -m "perf: make execute_query async to unblock FastAPI event loop"
```

---

## Task 2: Per-query cache TTL + GZip middleware

**Files:**
- Modify: `core/cache_manager.py`
- Modify: `main.py`
- Create: `tests/unit/test_cache_manager.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_cache_manager.py`:

```python
import time
import pytest
from unittest.mock import patch


def test_is_cache_valid_uses_custom_ttl():
    from core.cache_manager import cache_timestamps, is_cache_valid
    cache_timestamps["test_key"] = time.time() - 10  # 10s ago
    assert is_cache_valid("test_key", ttl=300) is True
    assert is_cache_valid("test_key", ttl=5) is False


def test_is_cache_valid_missing_key():
    from core.cache_manager import is_cache_valid
    assert is_cache_valid("nonexistent_key") is False


def test_default_ttl_is_300():
    from core.cache_manager import cache_timestamps, is_cache_valid
    cache_timestamps["key2"] = time.time() - 290
    assert is_cache_valid("key2") is True
    cache_timestamps["key2"] = time.time() - 310
    assert is_cache_valid("key2") is False
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/unit/test_cache_manager.py -v
```

Expected: `TypeError` — `is_cache_valid` doesn't accept `ttl` param yet.

- [ ] **Step 3: Update `core/cache_manager.py`**

Replace file content with:

```python
import time
from typing import Dict, List, Optional

memory_cache: Dict[str, Dict] = {}
cache_timestamps: Dict[str, float] = {}

CACHE_TTL_DEFAULT = 300       # 5 min — dynamic data (puntos with filters, series)
CACHE_TTL_STATIC = 3600       # 1 hour — atlas, cuencas catalog


def get_cache_key(query: str, params: List = None) -> str:
    key = query
    if params:
        key += str(params)
    return str(hash(key))


def is_cache_valid(cache_key: str, ttl: int = CACHE_TTL_DEFAULT) -> bool:
    if cache_key not in cache_timestamps:
        return False
    return time.time() - cache_timestamps[cache_key] < ttl


def clear_all_cache():
    memory_cache.clear()
    cache_timestamps.clear()
```

- [ ] **Step 4: Run tests — confirm they pass**

```bash
uv run pytest tests/unit/test_cache_manager.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Add GZipMiddleware to `main.py`**

In `main.py`, add the import after the existing imports:

```python
from starlette.middleware.gzip import GZipMiddleware
```

Then add the middleware registration immediately after the `CORSMiddleware` block (after line 93):

```python
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

- [ ] **Step 6: Apply static TTL in atlas and cuencas routers**

In `api/routers/atlas.py`, update the `execute_query` call to pass `ttl` via a wrapper. Since `_execute_query_sync` doesn't expose TTL directly, pass it as a kwarg to the sync layer.

First update `_execute_query_sync` in `core/database.py` to accept `ttl`:

```python
def _execute_query_sync(query: str, params: List = None, use_cache: bool = True, ttl: int = None) -> List[Dict]:
    from core.cache_manager import CACHE_TTL_DEFAULT
    effective_ttl = ttl if ttl is not None else CACHE_TTL_DEFAULT

    if use_cache:
        cache_key = get_cache_key(query, params)
        if cache_key in memory_cache and is_cache_valid(cache_key, ttl=effective_ttl):
            logging.info(f"Cache hit for query: {query[:50]}...")
            return memory_cache[cache_key]
    # ... rest unchanged
```

Update `execute_query` async wrapper to forward `ttl`:

```python
async def execute_query(query: str, params: List = None, use_cache: bool = True, ttl: int = None) -> List[Dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _execute_query_sync, query, params, use_cache, ttl
    )
```

In `api/routers/atlas.py`, pass `ttl`:

```python
from core.cache_manager import CACHE_TTL_STATIC
# ...
results = await execute_query(atlas_query, ttl=CACHE_TTL_STATIC)
```

In `api/routers/cuencas_hidrograficas.py`, find the catalog endpoint (the one that queries `dw.Cuencas_Regiones`) and pass `ttl=CACHE_TTL_STATIC` similarly.

- [ ] **Step 7: Commit**

```bash
git add core/cache_manager.py core/database.py main.py api/routers/atlas.py api/routers/cuencas_hidrograficas.py tests/unit/test_cache_manager.py
git commit -m "perf: per-query cache TTL and GZip response compression"
```

---

## Task 3: Fix broken Azure SQL queries in system.py

**Files:**
- Modify: `api/routers/system.py`

- [ ] **Step 1: Replace Synapse DMV queries**

`sys.dm_pdw_nodes_db_partition_stats` is Synapse-only. Azure SQL uses `sys.dm_db_partition_stats`.

Open `api/routers/system.py`. The query appears in two endpoints (`/test-db` and `/count`). Replace both occurrences:

Old:
```python
"SELECT SUM(row_count) as total FROM sys.dm_pdw_nodes_db_partition_stats "
"WHERE object_id = OBJECT_ID('dw.Mediciones_full') AND index_id IN (0,1)"
```

New:
```python
"SELECT SUM(row_count) as total FROM sys.dm_db_partition_stats "
"WHERE object_id = OBJECT_ID('dw.Mediciones_full') AND index_id IN (0,1)"
```

Also update all calls to `await execute_query(...)` if not already done by Task 1's sed command (verify with `grep execute_query api/routers/system.py`).

- [ ] **Step 2: Also fix the warm-up query in `cache_y_rendimiento.py`**

`api/routers/cache_y_rendimiento.py` line 47 has the same Synapse DMV query. Replace it:

Old:
```python
("SELECT SUM(row_count) as total FROM sys.dm_pdw_nodes_db_partition_stats WHERE object_id = OBJECT_ID('dw.Mediciones_full') AND index_id IN (0,1)", None),
```

New:
```python
("SELECT SUM(row_count) as total FROM sys.dm_db_partition_stats WHERE object_id = OBJECT_ID('dw.Mediciones_full') AND index_id IN (0,1)", None),
```

- [ ] **Step 3: Verify endpoints return 200**

With server running (`uv run uvicorn main:app --reload`):

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
curl -s http://localhost:8000/count | python3 -m json.tool
curl -s http://localhost:8000/test-db | python3 -m json.tool
```

Expected: all return `200` with valid JSON. `/count` returns `{"total_records": <number>}`.

- [ ] **Step 4: Commit**

```bash
git add api/routers/system.py api/routers/cache_y_rendimiento.py
git commit -m "fix: replace Synapse DMV with Azure SQL equivalent in system endpoints"
```

---

## Self-Review

- **Spec A covered:** async execute_query ✓, pool health check removal ✓, dead-conn retry ✓
- **Spec B covered:** per-query TTL ✓, GZip middleware ✓, static TTL for atlas/cuencas ✓
- **Spec C covered:** DMV fix in system.py ✓, warm-up query in cache_y_rendimiento.py ✓
- **No placeholders:** all steps have exact code and commands ✓
- **Type consistency:** `execute_query` signature consistent across tasks ✓ — Task 2 Step 6 extends the signature defined in Task 1 Step 3
