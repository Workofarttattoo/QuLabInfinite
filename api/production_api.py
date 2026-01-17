"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Production API for QuLab AI
Implements RESTful endpoints with health checks, monitoring, and error handling
"""
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import time
import psutil
from datetime import datetime

# Import QuLab AI components
from chemistry_lab.qulab_ai_integration import analyze_molecule_with_provenance
from core.validation_status import get_validation_status, get_validation_warnings
from frequency_lab.qulab_ai_integration import encode_spectrum_array
from qulab_ai.production import (
    get_logger,
    QuLabException,
    ParserException,
    ValidationException,
    retry,
    timed_execution
)

# Initialize logger
logger = get_logger("qulab_api")

# Initialize FastAPI app
app = FastAPI(
    title="QuLab AI Production API",
    description="Production-ready API for scientific computing with provenance tracking",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class MoleculeRequest(BaseModel):
    smiles: str = Field(..., description="SMILES notation", example="CCO")
    citations: Optional[List[Dict[str, str]]] = None

class MoleculeResponse(BaseModel):
    result: Dict[str, Any]
    digest: str
    timestamp_utc: str
    citations: List[Dict[str, Any]]
    units_checked: bool
    units_backend: str
    warnings: Optional[List[str]] = None

class SpectrumRequest(BaseModel):
    x: List[float] = Field(..., description="X-axis values")
    y: List[float] = Field(..., description="Y-axis values")
    caption: Optional[str] = Field("", description="Text caption for alignment")

class SpectrumResponse(BaseModel):
    ml_encoding: Dict[str, float]
    alignment: Optional[Dict[str, Any]] = None
    data_points: int
    warnings: Optional[List[str]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    system: Dict[str, Any]
    dependencies: Dict[str, bool]

class MetricsResponse(BaseModel):
    uptime_seconds: float
    total_requests: int
    total_errors: int
    avg_response_time_ms: float
    system_metrics: Dict[str, Any]

# Metrics tracking
class Metrics:
    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.total_errors = 0
        self.response_times = []
        self.max_response_times = 1000  # Keep last 1000 response times

    def record_request(self, response_time_ms: float, is_error: bool = False):
        self.total_requests += 1
        if is_error:
            self.total_errors += 1

        self.response_times.append(response_time_ms)
        if len(self.response_times) > self.max_response_times:
            self.response_times.pop(0)

    def get_avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)

    def get_uptime(self) -> float:
        return time.time() - self.start_time

metrics = Metrics()

# Middleware for request timing and logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    request_id = f"{int(start_time * 1000000)}"

    logger.info(
        "Request started",
        request_id=request_id,
        method=request.method,
        path=request.url.path
    )

    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000

        metrics.record_request(duration_ms, is_error=(response.status_code >= 400))

        logger.info(
            "Request completed",
            request_id=request_id,
            status_code=response.status_code,
            duration_ms=duration_ms
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        return response

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        metrics.record_request(duration_ms, is_error=True)

        logger.error(
            "Request failed",
            request_id=request_id,
            error=str(e),
            duration_ms=duration_ms
        )
        raise

# Exception handler
@app.exception_handler(QuLabException)
async def qulab_exception_handler(request: Request, exc: QuLabException):
    logger.error(
        "QuLab exception",
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        path=request.url.path
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "timestamp": exc.timestamp.isoformat() + "Z"
        }
    )

# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Health check endpoint

    Returns system health status including:
    - API status
    - System resources (CPU, memory, disk)
    - Dependency status
    """
    # Check dependencies
    dependencies = {
        "pint": True,  # Unit conversion
        "rdkit": True,  # Molecular parsing (fallback available)
        "logging": True,  # Always available
    }

    try:
        from qulab_ai.units import HAVE_PINT
        dependencies["pint"] = HAVE_PINT
    except:
        dependencies["pint"] = False

    # System metrics
    system = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
    }

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "1.0.0",
        "system": system,
        "dependencies": dependencies
    }

# Metrics endpoint
@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """
    Get API metrics

    Returns:
    - Uptime
    - Total requests
    - Error rate
    - Average response time
    - System metrics
    """
    system_metrics = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_available_mb": psutil.virtual_memory().available / (1024 * 1024),
        "disk_percent": psutil.disk_usage('/').percent,
    }

    return {
        "uptime_seconds": metrics.get_uptime(),
        "total_requests": metrics.total_requests,
        "total_errors": metrics.total_errors,
        "avg_response_time_ms": metrics.get_avg_response_time(),
        "system_metrics": system_metrics
    }

# Validation status endpoint
@app.get("/validation/status", tags=["Validation"])
async def validation_status():
    """
    Report current validation gates and coverage metadata.
    """
    return get_validation_status()

# Molecule parsing endpoint
@app.post("/api/v1/parse/molecule", response_model=MoleculeResponse, tags=["Chemistry"])
@timed_execution(log_threshold_ms=100.0)
@retry(max_attempts=2, delay_seconds=0.5)
async def parse_molecule(request: MoleculeRequest):
    """
    Parse molecule from SMILES notation with provenance tracking

    Args:
        request: Molecule request with SMILES string

    Returns:
        Parsed molecule with provenance (digest, timestamp, citations)
    """
    try:
        result = analyze_molecule_with_provenance(
            request.smiles,
            citations=request.citations
        )
        warnings = get_validation_warnings("chemistry", request.dict())

        logger.log_operation(
            operation="parse_molecule",
            status="success",
            smiles=request.smiles,
            n_atoms=result["result"].get("n_atoms", 0)
        )

        return {**result, "warnings": warnings or None}

    except Exception as e:
        logger.log_operation(
            operation="parse_molecule",
            status="error",
            smiles=request.smiles,
            error=str(e)
        )
        raise ParserException(
            f"Failed to parse molecule: {str(e)}",
            parser_type="smiles",
            input=request.smiles
        )

# Spectrum encoding endpoint
@app.post("/api/v1/encode/spectrum", response_model=SpectrumResponse, tags=["Spectroscopy"])
@timed_execution(log_threshold_ms=50.0)
@retry(max_attempts=2, delay_seconds=0.5)
async def encode_spectrum(request: SpectrumRequest):
    """
    Encode spectrum for machine learning

    Args:
        request: Spectrum data (x, y arrays) and optional caption

    Returns:
        ML encoding (peaks, centroid, variance, roughness) and alignment score
    """
    if len(request.x) != len(request.y):
        raise ValidationException(
            "X and Y arrays must have same length",
            field="x,y",
            x_length=len(request.x),
            y_length=len(request.y)
        )

    if len(request.x) < 2:
        raise ValidationException(
            "Spectrum must have at least 2 data points",
            field="x,y",
            length=len(request.x)
        )

    try:
        result = encode_spectrum_array(
            request.x,
            request.y,
            request.caption
        )
        warnings = get_validation_warnings(
            "spectroscopy",
            {"data_points": len(request.x)}
        )

        logger.log_operation(
            operation="encode_spectrum",
            status="success",
            data_points=len(request.x),
            peaks=result["ml_encoding"]["peaks"]
        )

        return {**result, "warnings": warnings or None}

    except Exception as e:
        logger.log_operation(
            operation="encode_spectrum",
            status="error",
            data_points=len(request.x),
            error=str(e)
        )
        raise ParserException(
            f"Failed to encode spectrum: {str(e)}",
            parser_type="spectrum",
            data_points=len(request.x)
        )

# Root endpoint
@app.get("/", tags=["General"])
async def root():
    """API root with links to documentation"""
    return {
        "message": "QuLab AI Production API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/health",
        "metrics": "/metrics"
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("QuLab AI Production API starting up", version="1.0.0")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info(
        "QuLab AI Production API shutting down",
        uptime_seconds=metrics.get_uptime(),
        total_requests=metrics.total_requests
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
