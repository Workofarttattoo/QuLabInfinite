"""
Database models and session management for QuLab Billing.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Generator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

# ---------------------------------------------------------------------------
# Engine singleton
# ---------------------------------------------------------------------------

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        db_path = os.environ.get("QULAB_DB_PATH", "qulab_billing.db")
        _engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
    return _engine


# ---------------------------------------------------------------------------
# ORM base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    __allow_unmapped__ = True


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class User(Base):
    __tablename__ = "users"

    id: int = Column(Integer, primary_key=True, index=True)
    email: str = Column(String, unique=True, nullable=False, index=True)
    stripe_customer_id: str | None = Column(String, nullable=True)
    credit_balance: int = Column(Integer, default=0, nullable=False)
    is_admin: bool = Column(Boolean, default=False, nullable=False)
    created_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)

    tokens: list[Token] = relationship("Token", back_populates="user")
    usage_logs: list[UsageLog] = relationship("UsageLog", back_populates="user")
    credit_purchases: list[CreditPurchase] = relationship("CreditPurchase", back_populates="user")


class Token(Base):
    __tablename__ = "tokens"

    id: int = Column(Integer, primary_key=True, index=True)
    key_hash: str = Column(String, unique=True, nullable=False, index=True)
    key_prefix: str = Column(String(8), nullable=False)
    user_id: int = Column(Integer, ForeignKey("users.id"), nullable=False)
    name: str = Column(String, default="default", nullable=False)
    is_active: bool = Column(Boolean, default=True, nullable=False)
    created_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at: datetime | None = Column(DateTime, nullable=True)

    user: User = relationship("User", back_populates="tokens")
    usage_logs: list[UsageLog] = relationship("UsageLog", back_populates="token")


class UsageLog(Base):
    __tablename__ = "usage_logs"

    id: int = Column(Integer, primary_key=True, index=True)
    token_id: int = Column(Integer, ForeignKey("tokens.id"), nullable=False)
    user_id: int = Column(Integer, ForeignKey("users.id"), nullable=False)
    tool_name: str = Column(String, nullable=False)
    credits_charged: int = Column(Integer, nullable=False)
    success: bool = Column(Boolean, default=True, nullable=False)
    latency_ms: float | None = Column(Float, nullable=True)
    timestamp: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)

    token: Token = relationship("Token", back_populates="usage_logs")
    user: User = relationship("User", back_populates="usage_logs")


class CreditPurchase(Base):
    __tablename__ = "credit_purchases"

    id: int = Column(Integer, primary_key=True, index=True)
    user_id: int = Column(Integer, ForeignKey("users.id"), nullable=False)
    stripe_payment_intent_id: str = Column(String, unique=True, nullable=False)
    credits: int = Column(Integer, nullable=False)
    amount_cents: int = Column(Integer, nullable=False)
    status: str = Column(String, default="pending", nullable=False)  # pending | completed | failed
    created_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)

    user: User = relationship("User", back_populates="credit_purchases")


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------


def create_tables() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=get_engine())


# ---------------------------------------------------------------------------
# Session factory & FastAPI dependency
# ---------------------------------------------------------------------------

_SessionLocal: sessionmaker | None = None


def _get_session_factory() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session."""
    factory = _get_session_factory()
    db: Session = factory()
    try:
        yield db
    finally:
        db.close()
