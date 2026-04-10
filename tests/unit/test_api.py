"""
Tests for QuLabInfinite API endpoints.
"""

import pytest

try:
    from fastapi.testclient import TestClient
    from qulab.api.main import app

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
class TestAPI:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "QuLabInfinite"

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "total_labs" in data

    def test_list_labs(self, client):
        response = client.get("/labs")
        assert response.status_code == 200

    def test_list_categories(self, client):
        response = client.get("/labs/categories")
        assert response.status_code == 200

    def test_list_medical(self, client):
        response = client.get("/labs/medical")
        assert response.status_code == 200

    def test_summary(self, client):
        response = client.get("/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_labs" in data

    def test_simulate_unknown_lab(self, client):
        response = client.post(
            "/simulate",
            json={
                "lab_name": "nonexistent_lab",
                "experiment_spec": {},
            },
        )
        assert response.status_code == 404
