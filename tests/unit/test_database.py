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
