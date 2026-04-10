import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Set up mock DB path before importing the app
import materials_api

client = TestClient(materials_api.app)

@pytest.fixture
def mock_db():
    with patch("materials_api.get_db") as mock_get_db:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Setup mock fetchall for search
        mock_row = {
            'material_id': 'mat-1',
            'formula': 'H2O',
            'category': 'liquid',
            'density': 1.0,
            'band_gap': 0,
            'cost_per_kg': 0.1,
            'melting_point': 273,
            'sources': '[]',
            'element_composition': '{}',
            'data_sources': '{}'
        }
        mock_cursor.fetchall.return_value = [mock_row]

        yield mock_cursor

def test_search_endpoint_limits(mock_db):
    response = client.get("/search?limit=5")
    assert response.status_code == 200

    # Verify the execute call uses the parameterized query
    call_args = mock_db.execute.call_args[0]
    query = call_args[0]
    params = call_args[1]

    assert "LIMIT ?" in query
    assert params == [5]

def test_recommend_endpoint_limits(mock_db):
    response = client.get("/recommend?use_case=structural&limit=3")
    assert response.status_code == 200

    # Verify the execute call uses the parameterized query
    call_args = mock_db.execute.call_args[0]
    query = call_args[0]
    params = call_args[1]

    assert "LIMIT ?" in query
    assert params == ['metal', 3]
