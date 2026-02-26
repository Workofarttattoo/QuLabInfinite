
import asyncio
import sys
from unittest.mock import MagicMock

# Create a more realistic mock for pydantic and fastapi
class MockBaseModel:
    pass

pydantic_mock = MagicMock()
pydantic_mock.BaseModel = MockBaseModel
sys.modules['pydantic'] = pydantic_mock

fastapi_mock = MagicMock()
sys.modules['fastapi'] = fastapi_mock
sys.modules['fastapi.security'] = MagicMock()
sys.modules['fastapi.responses'] = MagicMock()
sys.modules['uvicorn'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.stats'] = MagicMock()

import genetic_variant_analyzer_api

async def test_verify_api_key_logic():
    print("Testing verify_api_key logic...")

    # Mock credentials
    mock_credentials = MagicMock()

    # Test with valid key
    valid_key = "valid_test_key"
    genetic_variant_analyzer_api.VALID_API_KEYS = {valid_key: {"tier": "standard"}}
    mock_credentials.credentials = valid_key

    user_info = await genetic_variant_analyzer_api.verify_api_key(mock_credentials)
    assert user_info == {"tier": "standard"}
    print("✓ Valid key accepted")

    # Test with invalid key
    mock_credentials.credentials = "invalid_key"
    try:
        await genetic_variant_analyzer_api.verify_api_key(mock_credentials)
        print("✗ Invalid key was accepted (FAILED)")
        sys.exit(1)
    except Exception as e:
        print(f"✓ Invalid key rejected as expected")

if __name__ == "__main__":
    asyncio.run(test_verify_api_key_logic())
