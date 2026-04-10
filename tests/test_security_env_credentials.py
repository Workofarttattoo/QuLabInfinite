import os
import pytest
from unittest.mock import patch
from api.secure_production_api import startup_event, SecurityManager
from qulab_ai.production.security import USERS


@pytest.fixture(autouse=True)
def clear_users():
    """Clear USERS dict before and after each test."""
    USERS.clear()
    yield
    USERS.clear()


@pytest.mark.asyncio
async def test_startup_event_with_env_vars():
    """Test that admin user is created when environment variables are set."""
    env_vars = {
        "QU_LAB_ADMIN_USERNAME": "testadmin",
        "QU_LAB_ADMIN_PASSWORD": "securepassword123",
        "QU_LAB_ADMIN_EMAIL": "admin@example.com",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        await startup_event()

    assert "testadmin" in USERS
    assert USERS["testadmin"]["email"] == "admin@example.com"
    assert SecurityManager.verify_password(
        "securepassword123", USERS["testadmin"]["hashed_password"]
    )


@pytest.mark.asyncio
async def test_startup_event_without_env_vars():
    """Test that admin user is NOT created when environment variables are missing."""
    with patch.dict(os.environ, {}, clear=True):
        await startup_event()

    assert len(USERS) == 0


@pytest.mark.asyncio
async def test_startup_event_partial_env_vars():
    """Test that admin user is NOT created when only some environment variables are set."""
    env_vars = {
        "QU_LAB_ADMIN_USERNAME": "testadmin",
        "QU_LAB_ADMIN_PASSWORD": "securepassword123",
        # missing email
    }

    with patch.dict(os.environ, env_vars, clear=True):
        await startup_event()

    assert len(USERS) == 0
