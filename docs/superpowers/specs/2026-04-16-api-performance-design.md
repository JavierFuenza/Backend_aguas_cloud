# API Performance Improvements Design

**Date:** 2026-04-16  
**Status:** Approved

## Overview

Three independent performance improvements to the Aguas Transparentes FastAPI backend. No schema changes, no breaking API changes. All improvements are additive or in-place fixes.

## A — Fix Event Loop Blocking (pyodbc async)

**Problem:** `pyodbc` is synchronous. Every call to `execute_query()` in `core/database.py` blocks FastAPI's async event loop. Under concurrent requests, all requests queue behind each other.

**Fix:**
- Add `async def execute_query_async()` wrapper that calls `execute_query()` via `asyncio.get_event_loop().run_in_executor(None, ...)`.
- Update all router endpoint functions to `await execute_query_async(...)` instead of calling `execute_query()` directly.
- Remove the `SELECT 1` health check that runs on every `get_db_connection()` pool retrieval. Replace with try/except on the real query — if connection is dead, catch and retry with a fresh connection.

**Files affected:** `core/database.py`, all routers in `api/routers/`.

**Trade-off:** `run_in_executor` uses a thread pool (default: min(32, cpu_count+4) threads). Under very high concurrency this creates many threads, but for this workload it's far better than blocking the event loop.

## B — Cache TTL + GZip Compression

**Problem 1:** Cache TTL is 5 minutes for all queries including static data (regions, cuencas, atlas). Re-queries Synapse more than needed.

**Fix 1:** Introduce per-query TTL in `cache_manager.py`. Static endpoints (atlas, cuencas catalog) use 3600s (1 hour). Dynamic endpoints (puntos with filters, series) keep 300s (5 min).

**Problem 2:** Large JSON payloads (puntos list, time series) sent uncompressed.

**Fix 2:** Add `GZipMiddleware` to FastAPI in `main.py` with `minimum_size=1000`. Compresses responses ≥1KB automatically. Expected 60-80% size reduction on list endpoints.

**Files affected:** `core/cache_manager.py`, `main.py`.

## C — Fix Broken Synapse DMV Queries

**Problem:** `system.py` uses `sys.dm_pdw_nodes_db_partition_stats` — a Synapse-specific DMV. The API now uses Azure SQL. These queries fail silently or return errors.

**Affected endpoints:** `GET /count`, `GET /test-db`

**Fix:** Replace with standard Azure SQL approach:
- Use `SELECT COUNT(*) FROM dw.Mediciones_full` for exact count (or `sys.dm_db_partition_stats` for fast approximate).
- `/test-db` just needs to confirm connectivity — `SELECT 1` is sufficient, add a record count as bonus.

**Files affected:** `api/routers/system.py`.

## Architecture

No new dependencies required for A and C. B requires `fastapi` GZipMiddleware (already bundled with Starlette — no new package).

## Testing

- `/health` and `/test-db` must return 200 after C.
- Cache stats endpoint `/cache/stats` verifies cache is populated after warm-up.
- Manual concurrent requests (e.g., `ab -n 100 -c 10`) should show improved throughput after A.
