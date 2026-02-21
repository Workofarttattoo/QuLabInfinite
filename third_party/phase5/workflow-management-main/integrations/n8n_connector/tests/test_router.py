"""
Tests for N8N Router

Test the inbound webhook handling functionality from n8n to Canvas.
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from unittest.mock import AsyncMock, patch, MagicMock
import json
import hmac
import hashlib

from ..router import N8NRouter, create_n8n_router
from ..config import config
from ..utils import ExperimentInfo, ExperimentStatus


@pytest.fixture
def mock_experiment_service():
    """Mock experiment service"""
    service = AsyncMock()
    service.template_exists.return_value = True
    service.start_experiment.return_value = "exp_test_123"
    service.pause_experiment.return_value = True
    service.resume_experiment.return_value = True
    service.inject_user_input.return_value = True
    return service


@pytest.fixture
def mock_status_service():
    """Mock status service"""
    service = AsyncMock()
    service.get_experiment_status.return_value = ExperimentInfo(
        experiment_id="exp_test_123",
        status=ExperimentStatus.RUNNING,
        progress=50.0,
        current_step="step_3",
        total_steps=10
    )
    return service


@pytest.fixture
def test_app(mock_experiment_service, mock_status_service):
    """Create test FastAPI app with N8N router"""
    app = FastAPI()
    router = create_n8n_router(mock_experiment_service, mock_status_service)
    app.include_router(router)
    return app


@pytest.fixture
def client(test_app):
    """Test client"""
    return TestClient(test_app)


def create_signature(payload: dict, secret: str) -> str:
    """Create test signature for webhook verification"""
    payload_bytes = json.dumps(payload).encode('utf-8')
    signature = hmac.new(secret.encode('utf-8'), payload_bytes, hashlib.sha256).hexdigest()
    return f"sha256={signature}"


class TestN8NRouter:
    """Test N8N Router functionality"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/api/v1/n8n/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "config_valid" in data
    
    @patch.object(config, 'debug_mode', True)
    def test_trigger_experiment_success(self, client, mock_experiment_service):
        """Test successful experiment trigger"""
        payload = {
            "template_id": "catalyst_test_basic",
            "parameters": {
                "temperature": 80,
                "duration": 300
            },
            "triggered_by": "slack_user:@alice"
        }
        
        response = client.post("/api/v1/n8n/trigger-experiment", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["experiment_id"] == "exp_test_123"
        assert data["status"] == "started"
    
    @patch.object(config, 'debug_mode', True)
    def test_trigger_experiment_missing_template(self, client):
        """Test trigger experiment with missing template_id"""
        payload = {
            "parameters": {"temperature": 80},
            "triggered_by": "test_user"
        }
        
        response = client.post("/api/v1/n8n/trigger-experiment", json=payload)
        assert response.status_code == 422  # Validation error
    
    @patch.object(config, 'debug_mode', True)
    def test_pause_experiment_success(self, client, mock_experiment_service):
        """Test successful experiment pause"""
        payload = {
            "experiment_id": "exp_test_123",
            "reason": "User requested pause"
        }
        
        response = client.post("/api/v1/n8n/pause-experiment", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["experiment_id"] == "exp_test_123"
        assert data["status"] == "paused"
    
    @patch.object(config, 'debug_mode', True)
    def test_resume_experiment_success(self, client, mock_experiment_service):
        """Test successful experiment resume"""
        payload = {
            "experiment_id": "exp_test_123",
            "parameters": {"adjusted_temperature": 85}
        }
        
        response = client.post("/api/v1/n8n/resume-experiment", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["experiment_id"] == "exp_test_123"
        assert data["status"] == "resumed"
    
    def test_get_experiment_status(self, client, mock_status_service):
        """Test get experiment status"""
        response = client.get("/api/v1/n8n/get-status?experiment_id=exp_test_123")
        assert response.status_code == 200
        
        data = response.json()
        assert data["experiment_id"] == "exp_test_123"
        assert data["status"] == "running"
        assert data["progress"] == 50.0
        assert data["current_step"] == "step_3"
        assert data["total_steps"] == 10
    
    def test_get_experiment_status_with_details(self, client, mock_status_service):
        """Test get experiment status with details"""
        response = client.get("/api/v1/n8n/get-status?experiment_id=exp_test_123&include_details=true")
        assert response.status_code == 200
        
        data = response.json()
        assert data["experiment_id"] == "exp_test_123"
        assert "duration" in data
        assert "started_at" in data or data["started_at"] is None
    
    @patch.object(config, 'debug_mode', True)
    def test_inject_user_input_success(self, client, mock_experiment_service):
        """Test successful user input injection"""
        payload = {
            "experiment_id": "exp_test_123",
            "user_input": {
                "action": "continue",
                "selected_option": "A"
            }
        }
        
        response = client.post("/api/v1/n8n/inject-user-input", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["experiment_id"] == "exp_test_123"
    
    @patch.object(config, 'webhook_secret', 'test_secret')
    @patch.object(config, 'debug_mode', False)
    def test_webhook_signature_verification_success(self, client, mock_experiment_service):
        """Test successful webhook signature verification"""
        payload = {
            "template_id": "test_template",
            "parameters": {},
            "triggered_by": "test"
        }
        
        signature = create_signature(payload, 'test_secret')
        headers = {"x-n8n-signature": signature}
        
        response = client.post("/api/v1/n8n/trigger-experiment", json=payload, headers=headers)
        assert response.status_code == 200
    
    @patch.object(config, 'webhook_secret', 'test_secret')
    @patch.object(config, 'debug_mode', False)
    def test_webhook_signature_verification_failure(self, client):
        """Test failed webhook signature verification"""
        payload = {
            "template_id": "test_template",
            "parameters": {},
            "triggered_by": "test"
        }
        
        headers = {"x-n8n-signature": "invalid_signature"}
        
        response = client.post("/api/v1/n8n/trigger-experiment", json=payload, headers=headers)
        assert response.status_code == 401
    
    @patch.object(config, 'webhook_secret', 'test_secret')
    @patch.object(config, 'debug_mode', False)
    def test_webhook_missing_signature(self, client):
        """Test missing webhook signature"""
        payload = {
            "template_id": "test_template",
            "parameters": {},
            "triggered_by": "test"
        }
        
        response = client.post("/api/v1/n8n/trigger-experiment", json=payload)
        assert response.status_code == 401


class TestN8NRouterIntegration:
    """Integration tests for N8N Router"""
    
    @patch.object(config, 'debug_mode', True)
    def test_full_experiment_lifecycle(self, client, mock_experiment_service, mock_status_service):
        """Test full experiment lifecycle through n8n"""
        
        # 1. Trigger experiment
        trigger_payload = {
            "template_id": "test_template",
            "parameters": {"temperature": 80},
            "triggered_by": "integration_test"
        }
        
        response = client.post("/api/v1/n8n/trigger-experiment", json=trigger_payload)
        assert response.status_code == 200
        experiment_id = response.json()["experiment_id"]
        
        # 2. Check status
        response = client.get(f"/api/v1/n8n/get-status?experiment_id={experiment_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "running"
        
        # 3. Pause experiment
        pause_payload = {"experiment_id": experiment_id}
        response = client.post("/api/v1/n8n/pause-experiment", json=pause_payload)
        assert response.status_code == 200
        
        # 4. Resume experiment
        resume_payload = {"experiment_id": experiment_id}
        response = client.post("/api/v1/n8n/resume-experiment", json=resume_payload)
        assert response.status_code == 200
        
        # 5. Inject user input
        input_payload = {
            "experiment_id": experiment_id,
            "user_input": {"action": "continue"}
        }
        response = client.post("/api/v1/n8n/inject-user-input", json=input_payload)
        assert response.status_code == 200