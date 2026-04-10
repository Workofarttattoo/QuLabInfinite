"""
API Authentication for QuLabInfinite.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> Optional[str]:
    """
    Validate API key from header.

    If QULAB_API_KEY env var is set, requires matching key.
    If not set, allows unauthenticated access (development mode).
    """
    expected = os.getenv("QULAB_API_KEY")

    if expected is None:
        # Development mode — no key required
        return None

    if api_key != expected:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key. Set X-API-Key header.",
        )

    return api_key
