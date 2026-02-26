import sys
import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

# Ensure the project root is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_simulator import get_simulator
from api.auth import get_api_key
from api.v1.endpoints import api_router

# Optional: Stitch metrics for aios.is website (requires STITCH_API_KEY in env)
try:
    from api.stitch_metrics import fetch_stitch_metrics
except ImportError:
    fetch_stitch_metrics = None

app = FastAPI(
    title="QuLabInfinite API",
    description="A unified API for advanced scientific simulations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aios.is", "https://www.aios.is", "http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

simulator = get_simulator()

class SimulationRequest(BaseModel):
    lab_name: str
    experiment_spec: Dict[str, Any]

@app.get("/")
def read_root():
    return {"message": "Welcome to the QuLabInfinite API"}

@app.get("/labs", dependencies=[Depends(get_api_key)])
def list_labs():
    """List all available simulation labs and their capabilities."""
    return simulator.list_labs()

@app.post("/simulate", dependencies=[Depends(get_api_key)])
def run_simulation(request: SimulationRequest):
    """Run a simulation in a specified lab."""
    try:
        results = simulator.run_simulation(request.lab_name, request.experiment_spec)
        return {"status": "success", "results": results}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.get("/api/website/stitch-metrics")
def website_stitch_metrics():
    """
    Public endpoint for aios.is website: Stitch pipeline metrics.
    Uses STITCH_API_KEY server-side only; never exposes the key.
    """
    if fetch_stitch_metrics is None:
        return {"enabled": False, "reason": "Stitch integration not loaded", "sources": 0, "destinations": 0}
    return fetch_stitch_metrics()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
