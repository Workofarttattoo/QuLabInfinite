import pytest
from fastapi.testclient import TestClient
from materials_api import app

client = TestClient(app)

def test_recommend_endpoint():
    response = client.get("/recommend?use_case=structural&limit=10")
    # Even if DB is missing, we check it doesn't crash from the async/def change
    assert response.status_code in (200, 503)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
