import sys
from unittest.mock import MagicMock

# Mock dependencies
sys.modules["fastapi"] = MagicMock()
sys.modules["fastapi.middleware.cors"] = MagicMock()
sys.modules["fastapi.responses"] = MagicMock()
sys.modules["core.security"] = MagicMock()
sys.modules["materials_lab.materials_lab"] = MagicMock()
sys.modules["quantum_lab.quantum_simulator"] = MagicMock()
sys.modules["chemistry_lab.synthesis_optimizer"] = MagicMock()
sys.modules["oncology_lab.cancer_simulator"] = MagicMock()
sys.modules["numpy"] = MagicMock()

# Mock HTTPException
class MockHTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail

sys.modules["fastapi"].HTTPException = MockHTTPException

import api.unified_api as unified_api

# Test verify_api_key logic
unified_api.API_KEYS = {"test_key": {"tier": "free", "rate_limit": 100}}

def test_verify_api_key():
    # Case 1: Header key
    result = unified_api.verify_api_key(x_api_key="test_key")
    assert result["tier"] == "free"
    print("Case 1 passed")

    # Case 2: Query key
    result = unified_api.verify_api_key(api_key="test_key")
    assert result["tier"] == "free"
    print("Case 2 passed")

    # Case 3: Missing key
    try:
        unified_api.verify_api_key()
        assert False, "Should have raised HTTPException"
    except MockHTTPException as e:
        assert e.status_code == 401
        assert e.detail == "Missing API key"
        print("Case 3 passed")

    # Case 4: Invalid key
    try:
        unified_api.verify_api_key(x_api_key="invalid")
        assert False, "Should have raised HTTPException"
    except MockHTTPException as e:
        assert e.status_code == 401
        assert e.detail == "Invalid API key"
        print("Case 4 passed")

if __name__ == "__main__":
    try:
        test_verify_api_key()
        print("All mock tests passed!")
    except Exception as e:
        print(f"Mock tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
