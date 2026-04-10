"""Tests for the FastAPI endpoints."""

import os
import pytest

os.environ["QULAB_AUTH_ENABLED"] = "false"

from fastapi.testclient import TestClient
from qulab.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "labs_loaded" in data

    def test_root_returns_info(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "QuLabInfinite API"


class TestLabEndpoints:
    def test_list_labs(self):
        response = client.get("/labs")
        assert response.status_code == 200

    def test_list_categories(self):
        response = client.get("/labs/categories")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_unknown_lab_returns_404(self):
        response = client.post("/simulate", json={
            "lab_name": "nonexistent",
            "experiment_spec": {},
        })
        assert response.status_code == 404
