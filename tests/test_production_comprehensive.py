"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Comprehensive Production Test Suite - Phase 2 Implementation
Target: 98%+ code coverage
"""
import pytest
import json
from fastapi.testclient import TestClient
from api.secure_production_api import app
from qulab_ai.production import (
    get_logger,
    SecurityManager,
    RateLimiter,
    Circuit Breaker,
    retry,
    safe_execution,
    QuLabException,
    ParserException
)

# Test client
client = TestClient(app)

# Test fixtures
@pytest.fixture
def test_user():
    """Create test user"""
    return SecurityManager.create_user(
        username="testuser",
        password="testpass123",
        email="test@example.com",
        roles=["user"]
    )

@pytest.fixture
def test_api_key():
    """Create test API key"""
    return SecurityManager.create_api_key(
        name="Test Key",
        permissions=["read", "write"]
    )

@pytest.fixture
def access_token(test_user):
    """Get access token for test user"""
    return SecurityManager.create_access_token(data={"sub": "testuser"})

# Authentication Tests
class TestAuthentication:
    """Test authentication system"""

    def test_register_user(self):
        """Test user registration"""
        response = client.post("/auth/register", json={
            "username": "newuser",
            "password": "newpass123",
            "email": "newuser@example.com"
        })
        assert response.status_code == 200
        assert "user" in response.json()

    def test_login_success(self, test_user):
        """Test successful login"""
        response = client.post("/auth/token", data={
            "username": "testuser",
            "password": "testpass123"
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_login_failure(self):
        """Test failed login"""
        response = client.post("/auth/token", data={
            "username": "testuser",
            "password": "wrongpass"
        })
        assert response.status_code == 401

    def test_api_key_authentication(self, test_api_key):
        """Test API key authentication"""
        response = client.post(
            "/api/v2/parse/molecule-key",
            headers={"X-API-Key": test_api_key["key"]},
            json={"smiles": "CCO"}
        )
        assert response.status_code == 200

# API Endpoint Tests
class TestAPIEndpoints:
    """Test API endpoints"""

    def test_health_check(self):
        """Test health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "authentication" in data

    def test_parse_molecule_authenticated(self, access_token):
        """Test authenticated molecule parsing"""
        response = client.post(
            "/api/v2/parse/molecule",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"smiles": "CCO"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert data["result"]["canonical_smiles"] == "CCO"

    def test_parse_molecule_unauthenticated(self):
        """Test unauthenticated access fails"""
        response = client.post(
            "/api/v2/parse/molecule",
            json={"smiles": "CCO"}
        )
        assert response.status_code == 403  # No auth header

# Rate Limiting Tests
class TestRateLimiting:
    """Test rate limiting"""

    def test_rate_limit_enforcement(self, test_api_key):
        """Test that rate limit is enforced"""
        # Make requests until rate limited
        for i in range(150):  # Limit is 100/min
            response = client.post(
                "/api/v2/parse/molecule-key",
                headers={"X-API-Key": test_api_key["key"]},
                json={"smiles": "CCO"}
            )
            if response.status_code == 429:
                # Rate limited as expected
                assert "rate_limit" in response.json()
                return

        pytest.fail("Rate limit not enforced")

    def test_rate_limit_headers(self, test_api_key):
        """Test rate limit headers are present"""
        response = client.post(
            "/api/v2/parse/molecule-key",
            headers={"X-API-Key": test_api_key["key"]},
            json={"smiles": "CCO"}
        )
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers

# Error Handling Tests
class TestErrorHandling:
    """Test error handling"""

    def test_invalid_smiles(self, access_token):
        """Test invalid SMILES handling"""
        response = client.post(
            "/api/v2/parse/molecule",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"smiles": "INVALID!!!"}
        )
        # Should return error but not crash
        assert response.status_code in [400, 500]

    def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        breaker = CircuitBreaker(failure_threshold=3, timeout_seconds=1)

        def failing_func():
            raise Exception("Test failure")

        # Trigger circuit breaker
        for _ in range(3):
            try:
                breaker.call(failing_func)
            except:
                pass

        # Circuit should be open
        assert breaker.state.is_open

    def test_retry_decorator(self):
        """Test retry decorator"""
        call_count = [0]

        @retry(max_attempts=3, delay_seconds=0.1)
        def sometimes_fails():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Temporary failure")
            return "success"

        result = sometimes_fails()
        assert result == "success"
        assert call_count[0] == 3

    def test_safe_execution(self):
        """Test safe execution decorator"""
        @safe_execution(fallback_value="fallback")
        def risky_function():
            raise ValueError("Error")

        result = risky_function()
        assert result == "fallback"

# Logging Tests
class TestLogging:
    """Test logging system"""

    def test_logger_creation(self):
        """Test logger can be created"""
        logger = get_logger("test")
        assert logger is not None

    def test_log_operation(self):
        """Test operation logging"""
        logger = get_logger("test")
        logger.log_operation(
            operation="test_op",
            status="success",
            duration_ms=10.5
        )
        # Should not raise

# Security Tests
class TestSecurity:
    """Test security features"""

    def test_password_hashing(self):
        """Test password hashing"""
        password = "testpass123"
        hashed = SecurityManager.hash_password(password)
        assert hashed != password
        assert SecurityManager.verify_password(password, hashed)

    def test_jwt_token_creation(self):
        """Test JWT token creation and validation"""
        token = SecurityManager.create_access_token(data={"sub": "testuser"})
        payload = SecurityManager.decode_token(token)
        assert payload["sub"] == "testuser"
        assert payload["type"] == "access"

    def test_api_key_generation(self):
        """Test API key generation"""
        key = SecurityManager.generate_api_key()
        assert key.startswith("qlab_")
        assert len(key) > 40

# Performance Tests
class TestPerformance:
    """Test performance characteristics"""

    def test_response_time(self, access_token):
        """Test API response time"""
        import time
        start = time.time()
        response = client.post(
            "/api/v2/parse/molecule",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"smiles": "CCO"}
        )
        duration = (time.time() - start) * 1000
        assert response.status_code == 200
        assert duration < 100  # Should be under 100ms

# Integration Tests
class TestIntegration:
    """Test full integration workflows"""

    def test_full_authentication_flow(self):
        """Test complete authentication workflow"""
        # 1. Register
        register_response = client.post("/auth/register", json={
            "username": "flowuser",
            "password": "flowpass123",
            "email": "flow@example.com"
        })
        assert register_response.status_code == 200

        # 2. Login
        login_response = client.post("/auth/token", data={
            "username": "flowuser",
            "password": "flowpass123"
        })
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        # 3. Use token
        api_response = client.post(
            "/api/v2/parse/molecule",
            headers={"Authorization": f"Bearer {token}"},
            json={"smiles": "CCO"}
        )
        assert api_response.status_code == 200

        # 4. Refresh token
        refresh_token = login_response.json()["refresh_token"]
        refresh_response = client.post(
            "/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        assert refresh_response.status_code == 200

# Run tests with coverage
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=.", "--cov-report=html"])
