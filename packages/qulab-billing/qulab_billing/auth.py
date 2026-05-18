"""
Token and user management for QuLab Billing.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from qulab_billing.db import Token, User


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------


def generate_token() -> str:
    """Return a fresh raw token string: qlb_ + 32 random hex chars."""
    return "qlb_" + secrets.token_hex(32)


def hash_token(raw: str) -> str:
    """Return the SHA-256 hex digest of a raw token string."""
    return hashlib.sha256(raw.encode()).hexdigest()


def create_token(
    db: Session,
    user_id: int,
    name: str = "default",
    expires_days: int | None = None,
) -> tuple[Token, str]:
    """
    Create a new token in the database.

    Returns (Token ORM object, raw_key_string).
    The raw key is returned ONCE and never stored — only the hash is persisted.
    """
    raw = generate_token()
    key_hash = hash_token(raw)
    # key_prefix: first 8 chars after the "qlb_" prefix
    key_prefix = raw[4:12]

    expires_at: datetime | None = None
    if expires_days is not None:
        expires_at = datetime.utcnow() + timedelta(days=expires_days)

    token = Token(
        key_hash=key_hash,
        key_prefix=key_prefix,
        user_id=user_id,
        name=name,
        is_active=True,
        expires_at=expires_at,
    )
    db.add(token)
    db.commit()
    db.refresh(token)
    return token, raw


def validate_token(db: Session, raw_key: str) -> Token | None:
    """
    Look up a token by its hash.

    Returns the Token if it exists, is active, and has not expired.
    Returns None otherwise.
    """
    key_hash = hash_token(raw_key)
    token: Token | None = db.query(Token).filter(Token.key_hash == key_hash).first()
    if token is None:
        return None
    if not token.is_active:
        return None
    if token.expires_at is not None and token.expires_at < datetime.utcnow():
        return None
    return token


def get_user_for_token(db: Session, token: Token) -> User:
    """Return the User associated with a token. Raises if missing (shouldn't happen)."""
    user: User | None = db.query(User).filter(User.id == token.user_id).first()
    if user is None:
        raise RuntimeError(f"Token {token.id} references missing user {token.user_id}")
    return user


# ---------------------------------------------------------------------------
# User helpers
# ---------------------------------------------------------------------------


def create_user(db: Session, email: str) -> User:
    """Create and persist a new user with zero credit balance."""
    user = User(email=email, credit_balance=0)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_email(db: Session, email: str) -> User | None:
    """Return the User with the given email, or None if not found."""
    return db.query(User).filter(User.email == email).first()
