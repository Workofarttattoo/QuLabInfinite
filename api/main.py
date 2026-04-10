import sys
import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any

# Ensure the project root is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_simulator import UnifiedSimulator
from api.auth import get_api_key
from api.v1.endpoints import api_router

app = FastAPI(
    title="QuLabInfinite API",
    description="A unified API for advanced scientific simulations.",
    version="1.0.0",
)

app.include_router(api_router, prefix="/api/v1")

simulator = UnifiedSimulator()

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

@app.get("/validation/status", dependencies=[Depends(get_api_key)])
def validation_status():
    """Return current validation gates and coverage for all labs."""
    return simulator.get_validation_status()

@app.post("/simulate", dependencies=[Depends(get_api_key)])
def run_simulation(request: SimulationRequest):
    """Run a simulation in a specified lab."""
    try:
        results = simulator.run_simulation(request.lab_name, request.experiment_spec)
        response = {"status": "success", "results": results}
        response.update(simulator.get_validation_warnings(request.lab_name, request.experiment_spec))
        return response
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
