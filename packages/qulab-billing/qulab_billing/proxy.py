"""
MCP HTTP proxy with credit deduction for QuLab Billing.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime

import httpx
from sqlalchemy.orm import Session

from qulab_billing.db import Token, UsageLog, User
from qulab_billing.pricing import FREE_METHODS, get_tool_cost

# ---------------------------------------------------------------------------
# Backend URL
# ---------------------------------------------------------------------------

BACKEND_URL: str = os.environ.get("QULAB_BACKEND_URL", "http://127.0.0.1:8000")

# ---------------------------------------------------------------------------
# Singleton httpx client
# ---------------------------------------------------------------------------

_httpx_client: httpx.AsyncClient | None = None


def get_httpx_client() -> httpx.AsyncClient:
    """Return the module-level singleton AsyncClient, creating it on first call."""
    global _httpx_client
    if _httpx_client is None:
        _httpx_client = httpx.AsyncClient(timeout=120.0)
    return _httpx_client


# ---------------------------------------------------------------------------
# Proxy logic
# ---------------------------------------------------------------------------

# Headers to forward to the backend (case-insensitive matching done below)
_FORWARD_HEADERS = {"content-type", "accept", "mcp-protocol-version"}


def _build_backend_headers(request_headers: dict) -> dict:
    """
    Build the header dict to forward to the backend.
    Strips Authorization; forwards Content-Type, Accept, MCP-Protocol-Version.
    """
    forwarded: dict[str, str] = {}
    for k, v in request_headers.items():
        if k.lower() in _FORWARD_HEADERS:
            forwarded[k] = v
    return forwarded


def _parse_jsonrpc(body: bytes) -> dict | None:
    """Try to parse body as JSON-RPC. Returns dict or None on failure."""
    try:
        return json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None


async def proxy_mcp(
    request_body: bytes,
    request_headers: dict,
    token: Token,
    user: User,
    db: Session,
) -> tuple[bytes, int, dict]:
    """
    Proxy a JSON-RPC MCP request to the backend with pre-debit credit logic.

    Steps:
    1. Parse JSON-RPC body — if method == 'tools/call', get tool name and credits.
    2. If credits needed: check balance, deduct, return 402 if insufficient.
    3. Forward POST to BACKEND_URL/mcp with original body + headers (no Authorization).
    4. Collect full response (buffered).
    5. Log UsageLog record.
    6. Return (response_body, status_code, response_headers).

    Returns a 402 tuple directly if the user cannot afford the call.
    """
    # --- Step 1: parse body ---
    rpc = _parse_jsonrpc(request_body)
    method: str = ""
    tool_name: str = ""
    credits_required: int = 0

    if rpc is not None:
        method = rpc.get("method", "")
        if method == "tools/call":
            params = rpc.get("params", {})
            tool_name = params.get("name", "unknown")
            credits_required = get_tool_cost(tool_name)
        elif method in FREE_METHODS:
            credits_required = 0
        else:
            # Other methods (e.g. prompts/get, resources/read) — treat as free
            credits_required = 0

    # --- Step 2: pre-debit ---
    if credits_required > 0:
        # Re-query user inside this transaction to get a fresh balance
        from qulab_billing.db import User as UserModel
        fresh_user = db.query(UserModel).filter(UserModel.id == user.id).with_for_update().first()
        if fresh_user is None or fresh_user.credit_balance < credits_required:
            current_balance = fresh_user.credit_balance if fresh_user else 0
            error_body = json.dumps(
                {
                    "error": "Insufficient credits",
                    "balance": current_balance,
                    "required": credits_required,
                }
            ).encode()
            return error_body, 402, {"content-type": "application/json"}

        fresh_user.credit_balance -= credits_required
        db.commit()
        # Refresh the in-memory user object
        user.credit_balance = fresh_user.credit_balance

    # --- Step 3: forward to backend ---
    backend_headers = _build_backend_headers(request_headers)
    backend_url = f"{os.environ.get('QULAB_BACKEND_URL', BACKEND_URL)}/mcp"

    client = get_httpx_client()
    start_time = time.monotonic()
    success = True
    response_body = b""
    status_code = 500
    response_headers: dict = {}

    try:
        response = await client.post(
            backend_url,
            content=request_body,
            headers=backend_headers,
        )
        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        response_body = response.content
        status_code = response.status_code
        # Only pass through safe headers
        response_headers = {
            k: v
            for k, v in response.headers.items()
            if k.lower() in {"content-type", "content-length", "mcp-protocol-version"}
        }
        success = response.status_code < 400
    except httpx.RequestError as exc:
        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        success = False
        error_body = json.dumps({"error": f"Backend unreachable: {exc}"}).encode()
        return error_body, 502, {"content-type": "application/json"}

    # --- Step 5: log usage ---
    if method or tool_name:
        log_tool = tool_name if tool_name else method if method else "unknown"
        log = UsageLog(
            token_id=token.id,
            user_id=user.id,
            tool_name=log_tool,
            credits_charged=credits_required,
            success=success,
            latency_ms=elapsed_ms,
            timestamp=datetime.utcnow(),
        )
        db.add(log)
        db.commit()

    return response_body, status_code, response_headers
