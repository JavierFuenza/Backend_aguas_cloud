import time
from typing import Dict, List

memory_cache: Dict[str, Dict] = {}
cache_timestamps: Dict[str, float] = {}
CACHE_TTL = 300  # 5 minutes

def get_cache_key(query: str, params: List = None) -> str:
    """Generate cache key from query and parameters"""
    key = query
    if params:
        key += str(params)
    return str(hash(key))

def is_cache_valid(cache_key: str) -> bool:
    """Check if cache entry is still valid"""
    if cache_key not in cache_timestamps:
        return False
    return time.time() - cache_timestamps[cache_key] < CACHE_TTL

def clear_all_cache():
    """Helper to clear all cached items"""
    memory_cache.clear()
    cache_timestamps.clear()
