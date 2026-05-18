"""
FastAPI application for the QuLab Billing Proxy.

Architecture:
    Client ──Bearer token──► qulab-billing :8080
                                ├── POST /mcp  (proxy to qulab-mcp :8000)
                                ├── REST API for users/tokens/billing
                                └── POST /billing/webhook (Stripe)
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from qulab_billing.auth import (
    create_token,
    create_user,
    get_user_by_email,
    get_user_for_token,
    validate_token,
)
from qulab_billing.db import (
    CreditPurchase,
    Token,
    UsageLog,
    User,
    create_tables,
    get_db,
)
from qulab_billing.pricing import CREDIT_BUNDLES
from qulab_billing.proxy import BACKEND_URL, proxy_mcp

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="QuLab Billing Proxy",
    description="Pay-per-use billing proxy for the QuLab MCP server.",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup() -> None:
    """Create DB tables on startup."""
    create_tables()


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


def get_token_from_header(
    authorization: str = Header(..., description="Bearer <token>"),
    db: Session = Depends(get_db),
) -> Token:
    """
    Parse 'Authorization: Bearer <key>' header and validate the token.
    Raises HTTP 401 if invalid or missing.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header. Use 'Bearer <token>'.")
    raw_key = authorization[len("Bearer "):]
    token = validate_token(db, raw_key)
    if token is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return token


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: str


class CreateTokenRequest(BaseModel):
    name: str = "default"
    expires_days: Optional[int] = None


class CheckoutRequest(BaseModel):
    bundle_index: int
    success_url: str
    cancel_url: str


# ---------------------------------------------------------------------------
# User routes
# ---------------------------------------------------------------------------


@app.post("/users/register", status_code=201)
def register_user(body: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user. Optionally creates a Stripe customer if configured."""
    existing = get_user_by_email(db, body.email)
    if existing is not None:
        raise HTTPException(status_code=409, detail="Email already registered.")

    user = create_user(db, body.email)

    # Optionally create a Stripe customer
    stripe_key = os.environ.get("STRIPE_SECRET_KEY", "")
    if stripe_key:
        try:
            from qulab_billing.stripe_client import create_stripe_customer
            stripe_customer_id = create_stripe_customer(body.email)
            user.stripe_customer_id = stripe_customer_id
            db.commit()
            db.refresh(user)
        except Exception:
            # Non-fatal: Stripe customer creation failure doesn't block registration
            pass

    return {
        "user_id": user.id,
        "email": user.email,
        "credit_balance": user.credit_balance,
    }


@app.get("/users/me")
def get_current_user(
    token: Token = Depends(get_token_from_header),
    db: Session = Depends(get_db),
):
    """Return info about the authenticated user."""
    user = get_user_for_token(db, token)
    tokens = (
        db.query(Token)
        .filter(Token.user_id == user.id)
        .order_by(Token.created_at.desc())
        .all()
    )
    return {
        "user_id": user.id,
        "email": user.email,
        "credit_balance": user.credit_balance,
        "tokens": [
            {
                "token_id": t.id,
                "prefix": t.key_prefix,
                "name": t.name,
                "is_active": t.is_active,
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "expires_at": t.expires_at.isoformat() if t.expires_at else None,
            }
            for t in tokens
        ],
    }


# ---------------------------------------------------------------------------
# Token routes
# ---------------------------------------------------------------------------


@app.post("/tokens", status_code=201)
def create_new_token(
    body: CreateTokenRequest,
    token: Token = Depends(get_token_from_header),
    db: Session = Depends(get_db),
):
    """
    Create a new MCP token for the authenticated user.
    The raw key is returned ONCE and is never stored — only the hash is kept.
    """
    user = get_user_for_token(db, token)
    new_token, raw_key = create_token(
        db,
        user_id=user.id,
        name=body.name,
        expires_days=body.expires_days,
    )
    return {
        "token_id": new_token.id,
        "key": raw_key,
        "prefix": new_token.key_prefix,
        "name": new_token.name,
    }


@app.get("/tokens")
def list_tokens(
    token: Token = Depends(get_token_from_header),
    db: Session = Depends(get_db),
):
    """Return all tokens belonging to the authenticated user (no raw keys — prefix only)."""
    user = get_user_for_token(db, token)
    tokens = (
        db.query(Token)
        .filter(Token.user_id == user.id)
        .order_by(Token.created_at.desc())
        .all()
    )
    return [
        {
            "token_id": t.id,
            "prefix": t.key_prefix,
            "name": t.name,
            "is_active": t.is_active,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "expires_at": t.expires_at.isoformat() if t.expires_at else None,
        }
        for t in tokens
    ]


@app.delete("/tokens/{token_id}", status_code=200)
def deactivate_token(
    token_id: int,
    token: Token = Depends(get_token_from_header),
    db: Session = Depends(get_db),
):
    """Deactivate a token. Returns 404 if it doesn't belong to the authenticated user."""
    user = get_user_for_token(db, token)
    target = db.query(Token).filter(Token.id == token_id, Token.user_id == user.id).first()
    if target is None:
        raise HTTPException(status_code=404, detail="Token not found.")
    target.is_active = False
    db.commit()
    return {"ok": True, "token_id": token_id, "deactivated": True}


# ---------------------------------------------------------------------------
# Billing routes
# ---------------------------------------------------------------------------


@app.get("/billing/bundles")
def list_bundles():
    """Return available credit bundles (no auth required)."""
    return CREDIT_BUNDLES


@app.post("/billing/checkout")
def create_checkout(
    body: CheckoutRequest,
    token: Token = Depends(get_token_from_header),
    db: Session = Depends(get_db),
):
    """Create a Stripe Checkout Session for purchasing credits."""
    stripe_key = os.environ.get("STRIPE_SECRET_KEY", "")
    if not stripe_key:
        raise HTTPException(status_code=503, detail="Stripe billing is not configured.")

    user = get_user_for_token(db, token)

    # Ensure user has a Stripe customer ID
    if not user.stripe_customer_id:
        try:
            from qulab_billing.stripe_client import create_stripe_customer
            stripe_customer_id = create_stripe_customer(user.email)
            user.stripe_customer_id = stripe_customer_id
            db.commit()
            db.refresh(user)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Failed to create Stripe customer: {exc}")

    try:
        from qulab_billing.stripe_client import create_checkout_session
        checkout_url = create_checkout_session(
            stripe_customer_id=user.stripe_customer_id,
            bundle_index=body.bundle_index,
            success_url=body.success_url,
            cancel_url=body.cancel_url,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Stripe error: {exc}")

    return {"checkout_url": checkout_url}


@app.post("/billing/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """
    Handle Stripe webhook events.
    Verified by Stripe signature — no Bearer token required.
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        from qulab_billing.stripe_client import construct_webhook_event
        event = construct_webhook_event(payload, sig_header)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Webhook signature verification failed: {exc}")

    if event["type"] == "checkout.session.completed":
        session_obj = event["data"]["object"]
        metadata = session_obj.get("metadata", {})

        # Only handle QuLab credit purchases
        if metadata.get("purchase_type") != "qulab_credits":
            return {"ok": True}

        bundle_index_str = metadata.get("user_credit_bundle", "")
        try:
            bundle_index = int(bundle_index_str)
        except (ValueError, TypeError):
            return {"ok": True}

        # Resolve user by stripe_customer_id from the session
        stripe_customer_id = session_obj.get("customer")
        user: Optional[User] = None
        if stripe_customer_id:
            user = db.query(User).filter(User.stripe_customer_id == stripe_customer_id).first()

        if user is None:
            # Fallback: try customer_email if present
            customer_email = session_obj.get("customer_details", {}).get("email")
            if customer_email:
                user = get_user_by_email(db, customer_email)

        if user is None:
            # Cannot find user — log and skip
            return {"ok": True}

        # Get bundle info
        try:
            from qulab_billing.stripe_client import get_bundle
            bundle = get_bundle(bundle_index)
        except ValueError:
            return {"ok": True}

        credits_to_add = bundle["credits"]
        amount_cents = bundle["price_cents"]

        # Idempotency: use payment_intent as unique key
        payment_intent_id = session_obj.get("payment_intent", f"cs_{session_obj.get('id', 'unknown')}")

        existing_purchase = (
            db.query(CreditPurchase)
            .filter(CreditPurchase.stripe_payment_intent_id == payment_intent_id)
            .first()
        )
        if existing_purchase:
            # Already processed
            return {"ok": True}

        # Add credits
        user.credit_balance += credits_to_add
        purchase = CreditPurchase(
            user_id=user.id,
            stripe_payment_intent_id=payment_intent_id,
            credits=credits_to_add,
            amount_cents=amount_cents,
            status="completed",
            created_at=datetime.utcnow(),
        )
        db.add(purchase)
        db.commit()

    return {"ok": True}


@app.get("/billing/history")
def billing_history(
    token: Token = Depends(get_token_from_header),
    db: Session = Depends(get_db),
):
    """Return the last 100 usage log records for the authenticated user, newest first."""
    user = get_user_for_token(db, token)
    logs = (
        db.query(UsageLog)
        .filter(UsageLog.user_id == user.id)
        .order_by(UsageLog.timestamp.desc())
        .limit(100)
        .all()
    )
    return [
        {
            "id": log.id,
            "tool_name": log.tool_name,
            "credits_charged": log.credits_charged,
            "success": log.success,
            "latency_ms": log.latency_ms,
            "timestamp": log.timestamp.isoformat() if log.timestamp else None,
        }
        for log in logs
    ]


@app.get("/billing/balance")
def billing_balance(
    token: Token = Depends(get_token_from_header),
    db: Session = Depends(get_db),
):
    """Return the current credit balance for the authenticated user."""
    user = get_user_for_token(db, token)
    return {
        "credit_balance": user.credit_balance,
        "credits_value_usd": round(user.credit_balance * 0.01, 2),
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    """Health check endpoint. No auth required."""
    backend = os.environ.get("QULAB_BACKEND_URL", BACKEND_URL)
    return {"status": "ok", "backend": backend, "tools": 41}


# ---------------------------------------------------------------------------
# MCP proxy
# ---------------------------------------------------------------------------


@app.post("/mcp")
async def mcp_proxy_post(
    request: Request,
    authorization: str = Header(..., description="Bearer <token>"),
    db: Session = Depends(get_db),
):
    """
    Main MCP proxy endpoint. Validates Bearer token, deducts credits, and
    forwards the request to the backend MCP server.
    """
    # Validate token
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header.")
    raw_key = authorization[len("Bearer "):]
    token = validate_token(db, raw_key)
    if token is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    user = get_user_for_token(db, token)
    request_body = await request.body()
    request_headers = dict(request.headers)

    response_body, status_code, response_headers = await proxy_mcp(
        request_body=request_body,
        request_headers=request_headers,
        token=token,
        user=user,
        db=db,
    )

    if status_code == 402:
        # Insufficient credits — return structured JSON error
        try:
            error_data = json.loads(response_body)
        except Exception:
            error_data = {"error": "Insufficient credits"}
        return JSONResponse(content=error_data, status_code=402)

    content_type = response_headers.get("content-type", "application/json")
    return Response(
        content=response_body,
        status_code=status_code,
        media_type=content_type,
        headers={k: v for k, v in response_headers.items() if k.lower() != "content-type"},
    )


@app.get("/mcp")
async def mcp_proxy_get(
    request: Request,
    authorization: str = Header(..., description="Bearer <token>"),
    db: Session = Depends(get_db),
):
    """
    Forward GET requests to backend (for SSE handshake / capability negotiation).
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header.")
    raw_key = authorization[len("Bearer "):]
    token = validate_token(db, raw_key)
    if token is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    from qulab_billing.proxy import get_httpx_client, _build_backend_headers
    backend_url = f"{os.environ.get('QULAB_BACKEND_URL', BACKEND_URL)}/mcp"
    client = get_httpx_client()
    backend_headers = _build_backend_headers(dict(request.headers))

    try:
        response = await client.get(backend_url, headers=backend_headers)
        content_type = response.headers.get("content-type", "application/json")
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=content_type,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Backend unreachable: {exc}")
