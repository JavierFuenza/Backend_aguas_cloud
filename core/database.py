import asyncio
import os
import time
import logging
import pyodbc
from queue import Queue, Empty
from typing import List, Dict, Optional
from core.cache_manager import memory_cache, cache_timestamps, get_cache_key, is_cache_valid

connection_pool: Optional[Queue] = None
POOL_SIZE = 10


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


def _execute_query_sync(query: str, params: List = None, use_cache: bool = True, ttl: int = None) -> List[Dict]:
    from core.cache_manager import CACHE_TTL_DEFAULT
    effective_ttl = ttl if ttl is not None else CACHE_TTL_DEFAULT
    cache_key = get_cache_key(query, params)

    if use_cache:
        if cache_key in memory_cache and is_cache_valid(cache_key, ttl=effective_ttl):
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
            # Dead connection — retry once with fresh connection
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
        cursor.close()

        execution_time = time.time() - start_time
        logging.info(f"Query executed in {execution_time:.3f}s, returned {len(result_list)} rows")

        if use_cache and result_list:
            memory_cache[cache_key] = result_list
            cache_timestamps[cache_key] = time.time()

        return result_list
    finally:
        return_db_connection(conn)


async def execute_query(query: str, params: List = None, use_cache: bool = True, ttl: int = None) -> List[Dict]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _execute_query_sync, query, params, use_cache, ttl
    )
