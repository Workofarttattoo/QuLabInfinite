"""
QuLabInfinite — Unified FastAPI Application.

Single consolidated API that replaces the 6+ separate API files.
Auto-discovers all labs and exposes them through a clean REST interface.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from qulab.api.auth import get_api_key
from qulab.core.simulator import UnifiedSimulator

logger = logging.getLogger(__name__)

# Global simulator instance
_simulator: UnifiedSimulator | None = None


def get_simulator() -> UnifiedSimulator:
    """Dependency injection for the simulator."""
    global _simulator
    if _simulator is None:
        _simulator = UnifiedSimulator()
    return _simulator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    global _simulator
    logger.info("Starting QuLabInfinite API...")
    _simulator = UnifiedSimulator()
    summary = _simulator.summary()
    logger.info(
        "Loaded %d labs across %d categories",
        summary["total_labs"],
        len(summary["categories"]),
    )
    yield
    logger.info("Shutting down QuLabInfinite API...")


app = FastAPI(
    title="QuLabInfinite API",
    description=(
        "Unified scientific simulation platform with 100+ laboratories "
        "spanning physics, chemistry, biology, medicine, engineering, "
        "quantum computing, and more."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Request/Response models
# ------------------------------------------------------------------


class SimulationRequest(BaseModel):
    lab_name: str = Field(..., description="Name of the lab to run the simulation in")
    experiment_spec: dict[str, Any] = Field(
        ..., description="Experiment specification parameters"
    )


class SimulationResponse(BaseModel):
    status: str
    lab: str
    duration_ms: float
    results: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    version: str
    total_labs: int
    categories: int


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------


@app.get("/", tags=["info"])
def root():
    """Welcome endpoint."""
    return {
        "service": "QuLabInfinite",
        "version": "1.0.0",
        "docs": "/docs",
        "copyright": "© 2025 Joshua Hendricks Cole (Corporation of Light). All Rights Reserved.",
    }


@app.get("/health", response_model=HealthResponse, tags=["info"])
def health(simulator: UnifiedSimulator = Depends(get_simulator)):
    """Health check endpoint."""
    summary = simulator.summary()
    return HealthResponse(
        status="healthy",
        version=summary.get("version", "1.0.0"),
        total_labs=summary["total_labs"],
        categories=len(summary["categories"]),
    )


@app.get("/labs", tags=["labs"])
def list_labs(simulator: UnifiedSimulator = Depends(get_simulator)):
    """List all available labs and their capabilities."""
    return simulator.list_labs()


@app.get("/labs/categories", tags=["labs"])
def list_categories(simulator: UnifiedSimulator = Depends(get_simulator)):
    """List all lab categories."""
    return {"categories": simulator.list_categories()}


@app.get("/labs/medical", tags=["labs", "medical"])
def list_medical_labs(simulator: UnifiedSimulator = Depends(get_simulator)):
    """List all medical-grade labs."""
    return {"medical_labs": simulator.list_medical_labs()}


@app.get("/labs/{lab_name}", tags=["labs"])
def get_lab_info(
    lab_name: str,
    simulator: UnifiedSimulator = Depends(get_simulator),
):
    """Get details about a specific lab."""
    try:
        status = simulator.get_lab_status(lab_name)
        return {"lab": lab_name, "status": status}
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Lab '{lab_name}' not found. Use /labs to see available labs.",
        ) from exc


@app.post(
    "/simulate",
    response_model=SimulationResponse,
    tags=["simulation"],
    dependencies=[Depends(get_api_key)],
)
def run_simulation(
    request: SimulationRequest,
    simulator: UnifiedSimulator = Depends(get_simulator),
):
    """Run a simulation in a specified lab."""
    start = time.time()
    try:
        results = simulator.run_simulation(request.lab_name, request.experiment_spec)
        duration_ms = (time.time() - start) * 1000
        return SimulationResponse(
            status="success",
            lab=request.lab_name,
            duration_ms=round(duration_ms, 2),
            results=results,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Lab '{request.lab_name}' not found.",
        ) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Simulation error in %s", request.lab_name)
        raise HTTPException(status_code=500, detail=f"Simulation error: {exc}") from exc


@app.get("/summary", tags=["info"])
def summary(simulator: UnifiedSimulator = Depends(get_simulator)):
    """Get complete platform summary."""
    return simulator.summary()


# ------------------------------------------------------------------
# Include sub-routers (for future expansion)
# ------------------------------------------------------------------
from qulab.api.routes import labs_router, medical_router, roof_hunter_router  # noqa: E402

app.include_router(labs_router, prefix="/api/v1/labs", tags=["v1"])
app.include_router(medical_router, prefix="/api/v1/medical", tags=["v1", "medical"])
app.include_router(
    roof_hunter_router,
    prefix="/api/v1/roof-hunter",
    tags=["v1", "roof-hunter", "climate"],
)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
