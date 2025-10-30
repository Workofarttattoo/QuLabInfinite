from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.unified_simulator import UnifiedSimulator
from api.auth import get_api_key

api_router = APIRouter()
simulator = UnifiedSimulator()

class SimulationRequest(BaseModel):
    lab_name: str
    experiment_spec: Dict[str, Any]

@api_router.get("/health")
def health_check():
    return {"status": "ok"}

@api_router.get("/labs", dependencies=[Depends(get_api_key)])
def list_labs():
    """List all available simulation labs and their capabilities."""
    return simulator.list_labs()

@api_router.post("/simulate", dependencies=[Depends(get_api_key)])
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
