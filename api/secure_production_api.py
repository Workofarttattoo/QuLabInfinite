"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Secure Production API with Authentication
Enhanced version with OAuth2/JWT and API key support
"""
from fastapi import FastAPI, HTTPException, Request, Security, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import time

# Import QuLab AI components
from chemistry_lab.qulab_ai_integration import analyze_molecule_with_provenance
from frequency_lab.qulab_ai_integration import encode_spectrum_array
from qulab_ai.production import (
    get_logger,
    QuLabException,
    retry,
    timed_execution
)
from qulab_ai.production.security import (
    SecurityManager,
    RateLimiter,
    get_current_user_token,
    get_current_user_api_key
)

# Initialize logger
logger = get_logger("secure_api")

# Initialize FastAPI app
app = FastAPI(
    title="QuLab AI Secure Production API",
    description="Production API with OAuth2/JWT authentication and rate limiting",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://qulab.ai", "https://api.qulab.ai"],  # Configure for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Rate limiter
rate_limiter = RateLimiter(requests_per_minute=100)

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# Request/Response Models
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    email: EmailStr
    roles: Optional[List[str]] = ["user"]

class UserLogin(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class APIKeyCreate(BaseModel):
    name: str = Field(..., min_length=3)
    permissions: List[str] = ["read"]

class APIKeyResponse(BaseModel):
    key: str
    name: str
    permissions: List[str]
    created_at: datetime

class MoleculeRequest(BaseModel):
    smiles: str = Field(..., description="SMILES notation", example="CCO")
    citations: Optional[List[Dict[str, str]]] = None

# Middleware for rate limiting
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    # Extract identifier (API key, user, or IP)
    identifier = request.headers.get("X-API-Key") or \
                request.headers.get("Authorization") or \
                request.client.host

    # Check rate limit
    if not rate_limiter.check_rate_limit(identifier):
        rate_info = rate_limiter.get_rate_limit_info(identifier)
        logger.warning(
            "Rate limit exceeded",
            identifier=identifier[:20],
            limit=rate_info["limit"],
            reset=rate_info["reset"]
        )
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "rate_limit_exceeded",
                "message": "Too many requests",
                "rate_limit": rate_info
            },
            headers={
                "X-RateLimit-Limit": str(rate_info["limit"]),
                "X-RateLimit-Remaining": str(rate_info["remaining"]),
                "X-RateLimit-Reset": rate_info["reset"]
            }
        )

    # Add rate limit headers to response
    response = await call_next(request)
    rate_info = rate_limiter.get_rate_limit_info(identifier)
    response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
    response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
    response.headers["X-RateLimit-Reset"] = rate_info["reset"]

    return response

# Authentication endpoints
@app.post("/auth/register", response_model=Dict[str, Any], tags=["Authentication"])
async def register_user(user: UserCreate):
    """
    Register a new user

    Creates a new user account with email and password.
    Returns user data (without password).
    """
    try:
        user_data = SecurityManager.create_user(
            username=user.username,
            password=user.password,
            email=user.email,
            roles=user.roles
        )

        logger.info("User registered", username=user.username)

        return {
            "message": "User created successfully",
            "user": user_data
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("Registration failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@app.post("/auth/token", response_model=TokenResponse, tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 compatible token endpoint

    Authenticates user and returns access/refresh tokens.
    """
    user = SecurityManager.authenticate_user(form_data.username, form_data.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create tokens
    access_token = SecurityManager.create_access_token(data={"sub": user["username"]})
    refresh_token = SecurityManager.create_refresh_token(data={"sub": user["username"]})

    logger.info("User logged in", username=user["username"])

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": 1800  # 30 minutes
    }

@app.post("/auth/refresh", response_model=TokenResponse, tags=["Authentication"])
async def refresh_token(refresh_token: str):
    """
    Refresh access token using refresh token

    Args:
        refresh_token: Valid refresh token

    Returns:
        New access and refresh tokens
    """
    payload = SecurityManager.decode_token(refresh_token)

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type"
        )

    username = payload.get("sub")
    new_access_token = SecurityManager.create_access_token(data={"sub": username})
    new_refresh_token = SecurityManager.create_refresh_token(data={"sub": username})

    logger.info("Token refreshed", username=username)

    return {
        "access_token": new_access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer",
        "expires_in": 1800
    }

# API Key management
@app.post("/api-keys/create", response_model=APIKeyResponse, tags=["API Keys"])
async def create_api_key(
    key_request: APIKeyCreate,
    current_user: Dict = Depends(get_current_user_token)
):
    """
    Create a new API key (requires authentication)

    Args:
        key_request: API key configuration
        current_user: Authenticated user from JWT

    Returns:
        API key details including the key itself (shown only once)
    """
    key_data = SecurityManager.create_api_key(
        name=key_request.name,
        permissions=key_request.permissions
    )

    logger.info(
        "API key created",
        user=current_user.get("sub"),
        key_name=key_request.name
    )

    return {
        "key": key_data["key"],
        "name": key_data["name"],
        "permissions": key_data["permissions"],
        "created_at": key_data["created_at"]
    }

# Protected endpoints - JWT Authentication
@app.post("/api/v2/parse/molecule", tags=["Chemistry (JWT)"])
@timed_execution(log_threshold_ms=100.0)
@retry(max_attempts=2, delay_seconds=0.5)
async def parse_molecule_jwt(
    request: MoleculeRequest,
    current_user: Dict = Depends(get_current_user_token)
):
    """
    Parse molecule (JWT authenticated)

    Requires: Valid JWT access token
    """
    try:
        result = analyze_molecule_with_provenance(
            request.smiles,
            citations=request.citations
        )

        logger.log_operation(
            operation="parse_molecule",
            status="success",
            user=current_user.get("sub"),
            smiles=request.smiles
        )

        return result

    except Exception as e:
        logger.log_operation(
            operation="parse_molecule",
            status="error",
            user=current_user.get("sub"),
            error=str(e)
        )
        raise

# Protected endpoints - API Key Authentication
@app.post("/api/v2/parse/molecule-key", tags=["Chemistry (API Key)"])
@timed_execution(log_threshold_ms=100.0)
@retry(max_attempts=2, delay_seconds=0.5)
async def parse_molecule_apikey(
    request: MoleculeRequest,
    api_key_data: Dict = Depends(get_current_user_api_key)
):
    """
    Parse molecule (API Key authenticated)

    Requires: Valid API key in X-API-Key header
    """
    # Check permissions
    if "read" not in api_key_data["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        result = analyze_molecule_with_provenance(
            request.smiles,
            citations=request.citations
        )

        logger.log_operation(
            operation="parse_molecule",
            status="success",
            api_key_name=api_key_data["name"],
            smiles=request.smiles
        )

        return result

    except Exception as e:
        logger.log_operation(
            operation="parse_molecule",
            status="error",
            api_key_name=api_key_data["name"],
            error=str(e)
        )
        raise

# Public health endpoint (no auth required)
@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Public health check (no authentication required)"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "2.0.0",
        "authentication": "enabled"
    }

# Root endpoint
@app.get("/", tags=["General"])
async def root():
    """API root"""
    return {
        "message": "QuLab AI Secure Production API",
        "version": "2.0.0",
        "docs": "/api/docs",
        "authentication": {
            "jwt": "/auth/token",
            "api_key": "X-API-Key header"
        }
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Secure API starting", version="2.0.0")

    # Create default admin user for testing
    try:
        SecurityManager.create_user(
            username="admin",
            password="admin123",  # Change in production!
            email="admin@qulab.ai",
            roles=["admin", "user"]
        )
        logger.info("Default admin user created")
    except:
        logger.info("Admin user already exists")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        ssl_keyfile=None,  # Add SSL cert in production
        ssl_certfile=None
    )
