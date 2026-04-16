import time
from typing import Dict, List

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
