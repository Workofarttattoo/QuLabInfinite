"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Production Security Module
Implements OAuth2/JWT authentication, API keys, and rate limiting
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
import secrets
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
    def create_api_key(name: str, permissions: list = None) -> Dict[str, Any]:
        """
        Create a new API key

        Args:
            name: API key name/description
            permissions: List of permissions

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
            "active": True
        }

        API_KEYS[api_key] = key_data
        logger.info("Created API key", name=name, permissions=permissions)

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
    def create_user(username: str, password: str, email: str, roles: list = None) -> Dict[str, Any]:
        """
        Create a new user

        Args:
            username: Username
            password: Plain password (will be hashed)
            email: User email
            roles: List of user roles

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
            "active": True
        }

        USERS[username] = user_data
        logger.info("Created user", username=username, roles=roles)

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


class RateLimiter:
    """Rate limiting implementation"""

    def __init__(self, requests_per_minute: int = 60):
        """
        Initialize rate limiter

        Args:
            requests_per_minute: Maximum requests per minute
        """
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60

    def check_rate_limit(self, identifier: str) -> bool:
        """
        Check if request is within rate limit

        Args:
            identifier: Unique identifier (user ID, IP, API key)

        Returns:
            True if within limit, False otherwise
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.window_seconds)

        # Get request history for this identifier
        if identifier not in RATE_LIMITS:
            RATE_LIMITS[identifier] = []

        # Remove old requests outside the window
        RATE_LIMITS[identifier] = [
            ts for ts in RATE_LIMITS[identifier] if ts > cutoff
        ]

        # Check if under limit
        if len(RATE_LIMITS[identifier]) >= self.requests_per_minute:
            logger.warning(
                "Rate limit exceeded",
                identifier=identifier,
                requests=len(RATE_LIMITS[identifier]),
                limit=self.requests_per_minute
            )
            return False

        # Add current request
        RATE_LIMITS[identifier].append(now)
        return True

    def get_rate_limit_info(self, identifier: str) -> Dict[str, Any]:
        """
        Get rate limit information for identifier

        Args:
            identifier: Unique identifier

        Returns:
            Rate limit info including remaining requests
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.window_seconds)

        if identifier not in RATE_LIMITS:
            RATE_LIMITS[identifier] = []

        # Count requests in current window
        recent_requests = [
            ts for ts in RATE_LIMITS[identifier] if ts > cutoff
        ]

        remaining = self.requests_per_minute - len(recent_requests)
        reset_time = cutoff + timedelta(seconds=self.window_seconds)

        return {
            "limit": self.requests_per_minute,
            "remaining": max(0, remaining),
            "reset": reset_time.isoformat() + "Z",
            "window_seconds": self.window_seconds
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
