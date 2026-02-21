from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from core.security import load_api_keys_from_env

API_KEY_HEADER = APIKeyHeader(name="X-API-KEY")

VALID_API_KEYS = load_api_keys_from_env()

async def get_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Dependency to verify the API key."""
    if api_key in VALID_API_KEYS:
        return api_key
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
