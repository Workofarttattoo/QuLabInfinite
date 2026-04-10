import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from materials_api import app

client = TestClient(app)

@pytest.fixture
def mock_db():
    with patch("materials_api.get_db") as mock_get_db:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Setup mock return data
        def dict_factory(*args, **kwargs):
            return {
                'material_id': 'test_id',
                'formula': 'H2O',
                'sources': '[]',
                'element_composition': '{}',
                'data_sources': '{}'
            }

        mock_cursor.fetchall.return_value = [dict_factory()]
        yield mock_cursor

def test_search_endpoint_limit_parameterization(mock_db):
    """
    Test that the limit parameter is passed via query params to sqlite3 cursor.execute
    and not string interpolated, protecting against SQL injection.
    """
    limit_val = 50
    response = client.get(f"/search?limit={limit_val}")

    assert response.status_code == 200
    assert mock_db.execute.called

    call_args = mock_db.execute.call_args[0]
    query = call_args[0]
    params = call_args[1]

    assert query.endswith(" LIMIT ?")
    assert limit_val in params
    assert params[-1] == limit_val
