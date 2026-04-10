"""
Lab management routes (v1).

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from qulab.api.main import get_simulator
from qulab.core.simulator import UnifiedSimulator

router = APIRouter()


@router.get("/")
def list_all_labs(simulator: UnifiedSimulator = Depends(get_simulator)):
    """List all labs with metadata."""
    return simulator.list_labs()


@router.get("/by-category/{category}")
def labs_by_category(
    category: str,
    simulator: UnifiedSimulator = Depends(get_simulator),
):
    """List labs filtered by category."""
    labs = simulator.registry.list_by_category(category)
    if not labs:
        raise HTTPException(
            status_code=404,
            detail=f"No labs found in category '{category}'",
        )
    return {"category": category, "labs": labs}


@router.get("/{lab_name}/status")
def lab_status(
    lab_name: str,
    simulator: UnifiedSimulator = Depends(get_simulator),
):
    """Get detailed status of a specific lab."""
    try:
        return simulator.get_lab_status(lab_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Lab '{lab_name}' not found")


@router.get("/{lab_name}/capabilities")
def lab_capabilities(
    lab_name: str,
    simulator: UnifiedSimulator = Depends(get_simulator),
):
    """Get capabilities of a specific lab."""
    try:
        lab = simulator.registry.get(lab_name)
        return lab.get_capabilities()
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Lab '{lab_name}' not found")
