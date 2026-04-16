import time
import pytest
from core.cache_manager import clear_all_cache


@pytest.fixture(autouse=True)
def reset_cache():
    yield
    clear_all_cache()


def test_is_cache_valid_uses_custom_ttl():
    from core.cache_manager import cache_timestamps, is_cache_valid
    cache_timestamps["test_key"] = time.time() - 10  # 10s ago
    assert is_cache_valid("test_key", ttl=300) is True
    assert is_cache_valid("test_key", ttl=5) is False


def test_is_cache_valid_missing_key():
    from core.cache_manager import is_cache_valid
    assert is_cache_valid("nonexistent_key_xyz") is False


def test_default_ttl_is_300():
    from core.cache_manager import cache_timestamps, is_cache_valid, CACHE_TTL_DEFAULT
    cache_timestamps["key2"] = time.time() - (CACHE_TTL_DEFAULT - 10)
    assert is_cache_valid("key2") is True
    cache_timestamps["key2"] = time.time() - (CACHE_TTL_DEFAULT + 10)
    assert is_cache_valid("key2") is False
