"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Production Security Module
Implements OAuth2/JWT authentication, API keys, and rate limiting
"""
import json
import os
import secrets
import time
from collections import defaultdict
from datetime import datetime, timedelta
from threading import Lock
from typing import Optional, Dict, Any, Tuple

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from jose import JWTError, jwt
from passlib.context import CryptContext
from qulab_ai.production.logging_config import get_logger

logger = get_logger("security")

# Security configuration
SECRET_KEY = secrets.token_urlsafe(32)  # In production, load from environment
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
bearer_scheme = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# In-memory storage (replace with database in production)
API_KEYS: Dict[str, Dict[str, Any]] = {}
USERS: Dict[str, Dict[str, Any]] = {}
RATE_LIMITS: Dict[str, list] = {}

DEFAULT_RATE_LIMIT_TIERS: Dict[str, Dict[str, int]] = {
    "public": {"requests": 60, "window": 60},
    "standard": {"requests": 100, "window": 60},
    "pro": {"requests": 600, "window": 60},
    "enterprise": {"requests": 2000, "window": 60},
}


class SecurityManager:
    """Centralized security management"""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against hash"""
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """
        Create JWT access token

        Args:
            data: Token payload
            expires_delta: Token expiration time

        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire, "type": "access"})

        logger.info("Creating access token", user=data.get("sub"), expires=expire.isoformat())
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    @staticmethod
    def create_refresh_token(data: dict) -> str:
        """
        Create JWT refresh token

        Args:
            data: Token payload

        Returns:
            Encoded JWT refresh token
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})

        logger.info("Creating refresh token", user=data.get("sub"), expires=expire.isoformat())
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    @staticmethod
    def decode_token(token: str) -> Dict[str, Any]:
        """
        Decode and validate JWT token

        Args:
            token: JWT token string

        Returns:
            Decoded token payload

        Raises:
            HTTPException: If token is invalid
        """
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError as e:
            logger.error("Token validation failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @staticmethod
    def generate_api_key() -> str:
        """Generate a new API key"""
        return f"qlab_{secrets.token_urlsafe(32)}"

    @staticmethod
    def create_api_key(name: str, permissions: list = None, tier: str = "standard") -> Dict[str, Any]:
        """
        Create a new API key

        Args:
            name: API key name/description
            permissions: List of permissions
            tier: Rate limit tier for this key

        Returns:
            API key details including the key itself
        """
        api_key = SecurityManager.generate_api_key()
        key_data = {
            "key": api_key,
            "name": name,
            "permissions": permissions or ["read"],
            "created_at": datetime.utcnow(),
            "last_used": None,
            "active": True,
            "tier": tier or "standard",
        }

        API_KEYS[api_key] = key_data
        logger.info("Created API key", name=name, permissions=permissions, tier=tier)

        return key_data

    @staticmethod
    def validate_api_key(api_key: str) -> Dict[str, Any]:
        """
        Validate API key

        Args:
            api_key: API key to validate

        Returns:
            API key data

        Raises:
            HTTPException: If key is invalid
        """
        key_data = API_KEYS.get(api_key)

        if not key_data or not key_data["active"]:
            logger.warning("Invalid API key used", key_prefix=api_key[:10])
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )

        # Update last used
        key_data["last_used"] = datetime.utcnow()
        logger.debug("API key validated", name=key_data["name"])

        return key_data

    @staticmethod
    def get_api_key(api_key: str) -> Optional[Dict[str, Any]]:
        """Get API key data without raising HTTPException"""
        key_data = API_KEYS.get(api_key)
        if key_data and key_data.get("active"):
            return key_data
        return None

    @staticmethod
    def create_user(
        username: str,
        password: str,
        email: str,
        roles: list = None,
        tier: str = "standard"
    ) -> Dict[str, Any]:
        """
        Create a new user

        Args:
            username: Username
            password: Plain password (will be hashed)
            email: User email
            roles: List of user roles
            tier: Rate limit tier for this user

        Returns:
            User data (without password)
        """
        if username in USERS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )

        user_data = {
            "username": username,
            "email": email,
            "hashed_password": SecurityManager.hash_password(password),
            "roles": roles or ["user"],
            "created_at": datetime.utcnow(),
            "active": True,
            "tier": tier or "standard",
        }

        USERS[username] = user_data
        logger.info("Created user", username=username, roles=roles, tier=tier)

        # Return user data without password
        user_data_safe = user_data.copy()
        del user_data_safe["hashed_password"]
        return user_data_safe

    @staticmethod
    def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate user with username and password

        Args:
            username: Username
            password: Plain password

        Returns:
            User data if authenticated, None otherwise
        """
        user = USERS.get(username)

        if not user:
            logger.warning("Login attempt for non-existent user", username=username)
            return None

        if not SecurityManager.verify_password(password, user["hashed_password"]):
            logger.warning("Failed login attempt", username=username)
            return None

        logger.info("User authenticated", username=username)

        # Return user data without password
        user_safe = user.copy()
        del user_safe["hashed_password"]
        return user_safe

    @staticmethod
    def get_user(username: str) -> Optional[Dict[str, Any]]:
        """Get a user record without exposing password hash"""
        user = USERS.get(username)
        if not user:
            return None

        user_safe = user.copy()
        user_safe.pop("hashed_password", None)
        return user_safe

    @staticmethod
    def get_user_rate_limit_tier(username: Optional[str]) -> str:
        """Get the rate limit tier for a user"""
        if not username:
            return "standard"
        user = USERS.get(username)
        return user.get("tier", "standard") if user else "standard"

    @staticmethod
    def get_api_key_rate_limit_tier(api_key: Optional[str]) -> str:
        """Get the rate limit tier for an API key"""
        if not api_key:
            return "standard"
        key_data = API_KEYS.get(api_key)
        if key_data and key_data.get("active"):
            return key_data.get("tier", "standard")
        return "standard"


class RateLimitStore:
    """Abstract rate limit storage"""

    def record_request(self, identifier: str, window_seconds: int) -> Tuple[int, datetime]:
        raise NotImplementedError

    def get_request_count(self, identifier: str, window_seconds: int) -> Tuple[int, datetime]:
        raise NotImplementedError


class MemoryRateLimitStore(RateLimitStore):
    """In-memory rate limit backend"""

    def __init__(self, storage: Optional[Dict[str, list]] = None):
        self._lock = Lock()
        if storage is None:
            self._requests = defaultdict(list)
        else:
            self._requests = defaultdict(list, storage)

    def _prune(self, identifier: str, window_seconds: int) -> datetime:
        cutoff = datetime.utcnow() - timedelta(seconds=window_seconds)
        self._requests[identifier] = [
            ts for ts in self._requests[identifier]
            if ts > cutoff
        ]
        return cutoff

    def record_request(self, identifier: str, window_seconds: int) -> Tuple[int, datetime]:
        with self._lock:
            self._prune(identifier, window_seconds)
            now = datetime.utcnow()
            self._requests[identifier].append(now)
            earliest = self._requests[identifier][0]
            reset = earliest + timedelta(seconds=window_seconds)
            return len(self._requests[identifier]), reset

    def get_request_count(self, identifier: str, window_seconds: int) -> Tuple[int, datetime]:
        with self._lock:
            self._prune(identifier, window_seconds)
            if not self._requests[identifier]:
                now = datetime.utcnow()
                return 0, now + timedelta(seconds=window_seconds)
            earliest = self._requests[identifier][0]
            reset = earliest + timedelta(seconds=window_seconds)
            return len(self._requests[identifier]), reset


class RedisRateLimitStore(RateLimitStore):
    """Redis-backed rate limit backend"""

    def __init__(self, client, prefix: str = "qulab:rate_limit"):
        self.client = client
        self.prefix = prefix

    def _key(self, identifier: str) -> str:
        return f"{self.prefix}:{identifier}"

    def record_request(self, identifier: str, window_seconds: int) -> Tuple[int, datetime]:
        key = self._key(identifier)
        now_ms = int(time.time() * 1000)
        window_ms = window_seconds * 1000
        pipeline = self.client.pipeline()
        pipeline.zremrangebyscore(key, 0, now_ms - window_ms)
        pipeline.zadd(key, {str(now_ms): now_ms})
        pipeline.zcard(key)
        pipeline.expire(key, window_seconds)
        _, _, count, _ = pipeline.execute()

        earliest = self.client.zrange(key, 0, 0, withscores=True)
        if earliest:
            reset_ms = int(earliest[0][1]) + window_ms
        else:
            reset_ms = now_ms + window_ms

        reset_time = datetime.utcfromtimestamp(reset_ms / 1000.0)
        return int(count), reset_time

    def get_request_count(self, identifier: str, window_seconds: int) -> Tuple[int, datetime]:
        key = self._key(identifier)
        now_ms = int(time.time() * 1000)
        window_ms = window_seconds * 1000
        pipeline = self.client.pipeline()
        pipeline.zremrangebyscore(key, 0, now_ms - window_ms)
        pipeline.zcard(key)
        _, count = pipeline.execute()

        earliest = self.client.zrange(key, 0, 0, withscores=True)
        if earliest:
            reset_ms = int(earliest[0][1]) + window_ms
        else:
            reset_ms = now_ms + window_ms

        reset_time = datetime.utcfromtimestamp(reset_ms / 1000.0)
        return int(count), reset_time


class RateLimiter:
    """Rate limiting implementation with pluggable backends and tiered limits"""

    def __init__(
        self,
        requests_per_minute: Optional[int] = 60,
        store: Optional[RateLimitStore] = None,
        tier_limits: Optional[Dict[str, Dict[str, int]]] = None,
        default_tier: str = "standard"
    ):
        """
        Initialize rate limiter

        Args:
            requests_per_minute: Maximum requests per minute for default tier (kept for backward compatibility)
            store: Backend store (memory or Redis)
            tier_limits: Mapping of tier -> {"requests": int, "window": int}
            default_tier: Default rate limit tier
        """
        self.tier_limits = self._load_tier_limits(tier_limits, requests_per_minute)
        self.default_tier = default_tier if default_tier in self.tier_limits else "standard"
        self.store = store or self._init_store()

    def _init_store(self) -> RateLimitStore:
        backend = os.environ.get("QULAB_RATE_LIMIT_BACKEND", "memory").lower()
        if backend == "redis":
            try:
                import redis  # type: ignore

                redis_url = os.environ.get("QULAB_REDIS_URL", "redis://localhost:6379/0")
                client = redis.Redis.from_url(redis_url)
                client.ping()
                logger.info("Initialized Redis rate limit backend", redis_url=redis_url)
                return RedisRateLimitStore(client)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning(
                    "Redis backend unavailable, falling back to memory",
                    error=str(exc)
                )
        logger.info("Initialized in-memory rate limit backend")
        return MemoryRateLimitStore(storage=RATE_LIMITS)

    def _load_tier_limits(
        self,
        tier_limits: Optional[Dict[str, Dict[str, int]]],
        requests_per_minute: Optional[int]
    ) -> Dict[str, Dict[str, int]]:
        limits = {**DEFAULT_RATE_LIMIT_TIERS}

        if tier_limits:
            for tier, config in tier_limits.items():
                if not isinstance(config, dict):
                    continue
                requests = int(config.get("requests", limits.get(tier, {}).get("requests", 60)))
                window = int(config.get("window", limits.get(tier, {}).get("window", 60)))
                limits[tier] = {"requests": requests, "window": window}

        env_limits = os.environ.get("QULAB_RATE_LIMIT_TIERS")
        if env_limits:
            try:
                parsed = json.loads(env_limits)
                for tier, config in parsed.items():
                    requests = int(config.get("requests", limits.get(tier, {}).get("requests", 60)))
                    window = int(config.get("window", limits.get(tier, {}).get("window", 60)))
                    limits[tier] = {"requests": requests, "window": window}
            except json.JSONDecodeError as exc:
                logger.warning("Invalid QULAB_RATE_LIMIT_TIERS override", error=str(exc))

        if requests_per_minute is not None:
            limits["standard"] = {
                "requests": requests_per_minute,
                "window": limits.get("standard", {}).get("window", 60)
            }

        return limits

    def _get_tier_config(self, tier: Optional[str]) -> Tuple[str, Dict[str, int]]:
        tier_name = tier if tier in self.tier_limits else self.default_tier
        config = self.tier_limits.get(tier_name, self.tier_limits[self.default_tier])
        return tier_name, config

    def check_rate_limit(self, identifier: str, tier: Optional[str] = None) -> bool:
        """
        Check if request is within rate limit

        Args:
            identifier: Unique identifier (user ID, IP, API key)
            tier: Rate limit tier

        Returns:
            True if within limit, False otherwise
        """
        tier_name, config = self._get_tier_config(tier)
        count, reset_time = self.store.record_request(identifier, config["window"])
        allowed = count <= config["requests"]

        if not allowed:
            logger.warning(
                "Rate limit exceeded",
                identifier=identifier,
                requests=count,
                limit=config["requests"],
                tier=tier_name,
                reset=reset_time.isoformat() + "Z"
            )
        return allowed

    def get_rate_limit_info(self, identifier: str, tier: Optional[str] = None) -> Dict[str, Any]:
        """
        Get rate limit information for identifier

        Args:
            identifier: Unique identifier
            tier: Rate limit tier

        Returns:
            Rate limit info including remaining requests
        """
        tier_name, config = self._get_tier_config(tier)
        count, reset_time = self.store.get_request_count(identifier, config["window"])
        remaining = config["requests"] - count

        return {
            "limit": config["requests"],
            "remaining": max(0, remaining),
            "reset": reset_time.isoformat() + "Z",
            "window_seconds": config["window"],
            "tier": tier_name,
        }


# Dependencies for FastAPI
async def get_current_user_token(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)
) -> Dict[str, Any]:
    """
    Dependency to get current user from JWT token

    Args:
        credentials: Bearer token from request header

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid
    """
    token = credentials.credentials
    payload = SecurityManager.decode_token(token)

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type"
        )

    return payload


async def get_current_user_api_key(
    api_key: Optional[str] = Security(api_key_header)
) -> Dict[str, Any]:
    """
    Dependency to validate API key

    Args:
        api_key: API key from request header

    Returns:
        API key data

    Raises:
        HTTPException: If key is invalid
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )

    return SecurityManager.validate_api_key(api_key)


# Example usage
if __name__ == "__main__":
    # Create a test user
    user = SecurityManager.create_user(
        username="testuser",
        password="testpass123",
        email="test@example.com",
        roles=["user", "admin"]
    )
    print(f"Created user: {user}")

    # Authenticate user
    auth_user = SecurityManager.authenticate_user("testuser", "testpass123")
    print(f"Authenticated: {auth_user}")

    # Create tokens
    access_token = SecurityManager.create_access_token(data={"sub": "testuser"})
    refresh_token = SecurityManager.create_refresh_token(data={"sub": "testuser"})
    print(f"Access token: {access_token[:50]}...")
    print(f"Refresh token: {refresh_token[:50]}...")

    # Create API key
    api_key_data = SecurityManager.create_api_key(
        name="Test API Key",
        permissions=["read", "write"]
    )
    print(f"API Key: {api_key_data['key']}")

    # Test rate limiting
    rate_limiter = RateLimiter(requests_per_minute=10)
    for i in range(12):
        allowed = rate_limiter.check_rate_limit("test_user")
        print(f"Request {i+1}: {'✓ Allowed' if allowed else '✗ Rate limited'}")

    # Get rate limit info
    info = rate_limiter.get_rate_limit_info("test_user")
    print(f"Rate limit info: {info}")
