"""
Tests for N8N Client

Test the outbound communication functionality from Canvas to n8n.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from ..client import N8NClient, N8NNotificationBuilder
from ..utils import ExperimentInfo, ExperimentStatus, NotificationLevel
from ..config import N8NConfig


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    config = MagicMock(spec=N8NConfig)
    config.timeout = 30
    config.retry_attempts = 3
    config.retry_delay = 1
    config.endpoints.notify_status = "test-notify"
    config.endpoints.alert_intervention = "test-alert"
    config.endpoints.report_ready = "test-report"
    config.endpoints.error_notification = "test-error"
    config.get_webhook_url.return_value = "https://test.n8n.io/webhook/test-endpoint"
    config.get_headers.return_value = {"Content-Type": "application/json"}
    return config


@pytest.fixture
def sample_experiment():
    """Sample experiment for testing"""
    return ExperimentInfo(
        experiment_id="exp_123",
        status=ExperimentStatus.COMPLETED,
        template_id="test_template",
        started_by="test_user",
        started_at=datetime(2024, 1, 1, 12, 0, 0),
        completed_at=datetime(2024, 1, 1, 12, 30, 0),
        progress=100.0,
        current_step="finished",
        total_steps=5,
        parameters={"temperature": 80},
        results={"peak_current": 1.2}
    )


@pytest.mark.asyncio
class TestN8NClient:
    """Test N8N Client functionality"""
    
    async def test_notify_experiment_status_success(self, mock_config, sample_experiment):
        """Test successful experiment status notification"""
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            
            mock_session = AsyncMock()
            mock_session.post.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            # Test the client
            client = N8NClient(custom_config=mock_config.__dict__)
            client.config = mock_config
            
            result = await client.notify_experiment_status(sample_experiment)
            
            assert result is True
            mock_session.post.assert_called_once()
    
    async def test_notify_experiment_status_retry_on_failure(self, mock_config, sample_experiment):
        """Test retry logic on failed notification"""
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            # Mock failed response
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text.return_value = "Internal Server Error"
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            
            mock_session = AsyncMock()
            mock_session.post.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            client = N8NClient(custom_config=mock_config.__dict__)
            client.config = mock_config
            
            result = await client.notify_experiment_status(sample_experiment)
            
            assert result is False
            assert mock_session.post.call_count == mock_config.retry_attempts
    
    async def test_send_alert(self, mock_config):
        """Test sending alert notification"""
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            
            mock_session = AsyncMock()
            mock_session.post.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            client = N8NClient(custom_config=mock_config.__dict__)
            client.config = mock_config
            
            result = await client.send_alert(
                experiment_id="exp_123",
                alert_type="temperature_warning",
                message="Temperature exceeded safe limits",
                level=NotificationLevel.WARNING,
                suggested_actions=["Lower temperature", "Pause experiment"]
            )
            
            assert result is True
            mock_session.post.assert_called_once()
    
    async def test_notify_report_ready(self, mock_config):
        """Test report ready notification"""
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            
            mock_session = AsyncMock()
            mock_session.post.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            client = N8NClient(custom_config=mock_config.__dict__)
            client.config = mock_config
            
            result = await client.notify_report_ready(
                experiment_id="exp_123",
                report_url="https://canvas.io/reports/exp_123.pdf",
                report_type="pdf"
            )
            
            assert result is True
            mock_session.post.assert_called_once()


class TestN8NNotificationBuilder:
    """Test N8N Notification Builder"""
    
    def test_builder_pattern(self):
        """Test builder pattern functionality"""
        
        builder = N8NNotificationBuilder()
        payload = (builder
                  .experiment("exp_123")
                  .status(ExperimentStatus.COMPLETED)
                  .summary("Test completed successfully")
                  .level(NotificationLevel.INFO)
                  .add_metadata("duration", "30m")
                  .build())
        
        assert payload["experiment_id"] == "exp_123"
        assert payload["status"] == "completed"
        assert payload["summary"] == "Test completed successfully"
        assert payload["level"] == "info"
        assert payload["metadata"]["duration"] == "30m"
        assert "timestamp" in payload
    
    @pytest.mark.asyncio
    async def test_builder_send_to(self, mock_config):
        """Test builder send_to functionality"""
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            
            mock_session = AsyncMock()
            mock_session.post.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            client = N8NClient(custom_config=mock_config.__dict__)
            client.config = mock_config
            
            builder = N8NNotificationBuilder()
            result = await (builder
                           .experiment("exp_123")
                           .summary("Test notification")
                           .send_to(client, "test-endpoint", "test_event"))
            
            assert result is True