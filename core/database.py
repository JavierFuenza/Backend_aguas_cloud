import os
import time
import logging
import pyodbc
from queue import Queue, Empty
from typing import List, Dict, Optional
import threading

from core.cache_manager import memory_cache, cache_timestamps, get_cache_key, is_cache_valid

# Global connection pool
connection_pool: Optional[Queue] = None
POOL_SIZE = 10
pool_lock = threading.Lock()

def create_db_connection():
    """Create a new database connection"""
    server = os.getenv('SYNAPSE_SERVER')
    database = os.getenv('SYNAPSE_DATABASE')
    username = os.getenv('SYNAPSE_USERNAME')
    password = os.getenv('SYNAPSE_PASSWORD')

    connection_string = f"""
    DRIVER={{ODBC Driver 18 for SQL Server}};
    SERVER={server};
    DATABASE={database};
    UID={username};
    PWD={password};
    Encrypt=yes;
    TrustServerCertificate=no;
    Connection Timeout=30;
    """

    return pyodbc.connect(connection_string)

def get_db_connection():
    """Get connection from pool"""
    global connection_pool
    if connection_pool is None:
        return create_db_connection()

    try:
        # Try to get from pool with timeout
        conn = connection_pool.get(timeout=5.0)

        # Test connection is still alive
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return conn
        except:
            # Connection is dead, create new one
            conn.close()
            return create_db_connection()

    except Empty:
        # Pool is empty, create new connection
        return create_db_connection()

def return_db_connection(conn):
    """Return connection to pool"""
    global connection_pool
    if connection_pool is None:
        conn.close()
        return

    try:
        if not connection_pool.full():
            connection_pool.put_nowait(conn)
        else:
            conn.close()
    except:
        conn.close()

def execute_query(query: str, params: List = None, use_cache: bool = True) -> List[Dict]:
    """Execute query with connection pooling and caching"""
    # Check cache first
    if use_cache:
        cache_key = get_cache_key(query, params)
        if cache_key in memory_cache and is_cache_valid(cache_key):
            logging.info(f"Cache hit for query: {query[:50]}...")
            return memory_cache[cache_key]

    # Execute query
    conn = get_db_connection()
    try:
        start_time = time.time()
        cursor = conn.cursor()

        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        columns = [column[0] for column in cursor.description]
        results = cursor.fetchall()
        result_list = [dict(zip(columns, row)) for row in results]

        execution_time = time.time() - start_time
        logging.info(f"Query executed in {execution_time:.3f}s, returned {len(result_list)} rows")

        # Cache the result
        if use_cache and len(result_list) > 0:
            memory_cache[cache_key] = result_list
            cache_timestamps[cache_key] = time.time()
            logging.info(f"Cached {len(result_list)} rows for query")

        return result_list

    finally:
        return_db_connection(conn)
