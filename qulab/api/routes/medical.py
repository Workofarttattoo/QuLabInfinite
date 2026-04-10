"""
Medical lab routes (v1).

Dedicated endpoints for clinical-grade medical labs.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from qulab.api.main import get_simulator
from qulab.core.simulator import UnifiedSimulator

router = APIRouter()


@router.get("/")
def list_medical_labs(simulator: UnifiedSimulator = Depends(get_simulator)):
    """List all medical-grade labs."""
    medical = simulator.list_medical_labs()
    labs_info = {}
    for name in medical:
        meta = simulator.registry.get_metadata(name)
        if meta:
            labs_info[name] = {
                "description": meta.description,
                "version": meta.version,
                "tags": list(meta.tags),
            }
    return {"count": len(medical), "labs": labs_info}


class MedicalSimulationRequest(BaseModel):
    lab_name: str = Field(..., description="Medical lab name")
    patient_data: Dict[str, Any] = Field(..., description="Patient input parameters")


@router.post("/simulate")
def run_medical_simulation(
    request: MedicalSimulationRequest,
    simulator: UnifiedSimulator = Depends(get_simulator),
):
    """Run a medical simulation with patient data."""
    medical_labs = simulator.list_medical_labs()
    if request.lab_name not in medical_labs:
        raise HTTPException(
            status_code=404,
            detail=f"'{request.lab_name}' is not a registered medical lab. "
            f"Available: {medical_labs}",
        )

    try:
        results = simulator.run_simulation(request.lab_name, request.patient_data)
        return {
            "status": "success",
            "lab": request.lab_name,
            "disclaimer": (
                "FOR RESEARCH PURPOSES ONLY. Not intended for clinical diagnosis "
                "or treatment decisions. Always consult qualified healthcare professionals."
            ),
            "results": results,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
