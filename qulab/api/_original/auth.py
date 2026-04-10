from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import json
import os

API_KEY_HEADER = APIKeyHeader(name="X-API-KEY")
api_keys_path = os.path.join(os.path.dirname(__file__), "..", "api_keys.json")

def load_api_keys() -> list[str]:
    """Load valid API keys from a JSON file."""
    try:
        with open(api_keys_path, "r") as f:
            data = json.load(f)
            return data.get("valid_keys", [])
    except FileNotFoundError:
        return []

VALID_API_KEYS = load_api_keys()

async def get_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Dependency to verify the API key."""
    if api_key in VALID_API_KEYS:
        return api_key
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
