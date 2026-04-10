from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.auth import get_api_key

test_runs_router = APIRouter()


class TestRunSubmission(BaseModel):
    """Submission of a SIMTEST result record."""
    test_id: str
    test_version: str
    engine: Dict[str, str]
    run_config: Dict[str, Any]
    metrics: Dict[str, float]
    status: str
    provenance: Dict[str, Any]
    outputs: Optional[Dict[str, Any]] = None
    checks: Optional[List[Dict[str, Any]]] = None
    duration_s: Optional[float] = None
    artifacts: Optional[List[Dict[str, str]]] = None
    errors: Optional[List[str]] = None


class TestRunResponse(BaseModel):
    """Response for a test run submission."""
    id: int
    test_id: str
    status: str
    submitted_at: datetime
    engine_name: str
    engine_version: str


@test_runs_router.post("/test-runs", dependencies=[Depends(get_api_key)])
async def submit_test_run(submission: TestRunSubmission):
    """
    Submit a SIMTEST result record for storage and leaderboard inclusion.
    
    The result will be validated against the SIMTEST schema and stored
    in the database if valid.
    """
    try:
        # Validate submission
        if submission.status not in ["pass", "fail", "error"]:
            raise HTTPException(status_code=400, detail=f"Invalid status: {submission.status}")
        
        # In a real implementation, this would:
        # 1. Validate against JSON Schema
        # 2. Store in database
        # 3. Update leaderboard
        
        # Placeholder: return mock ID
        return {
            "id": 1,
            "test_id": submission.test_id,
            "status": submission.status,
            "submitted_at": datetime.utcnow().isoformat(),
            "engine_name": submission.engine.get("name", "unknown"),
            "engine_version": submission.engine.get("version", "unknown")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@test_runs_router.post("/test-runs/file", dependencies=[Depends(get_api_key)])
async def submit_test_run_file(file: UploadFile = File(...)):
    """
    Submit a SIMTEST result record as a JSON file upload.
    """
    try:
        content = await file.read()
        data = json.loads(content.decode("utf-8"))
        submission = TestRunSubmission(**data)
        return await submit_test_run(submission)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@test_runs_router.get("/test-runs", dependencies=[Depends(get_api_key)])
async def list_test_runs(
    test_id: Optional[str] = None,
    engine_name: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    List test runs with optional filtering.
    """
    # Placeholder: would query database
    return {
        "total": 0,
        "limit": limit,
        "offset": offset,
        "results": []
    }


@test_runs_router.get("/test-runs/{run_id}", dependencies=[Depends(get_api_key)])
async def get_test_run(run_id: int):
    """
    Get a specific test run by ID.
    """
    # Placeholder: would query database
    raise HTTPException(status_code=404, detail=f"Test run {run_id} not found")


@test_runs_router.get("/test-runs/leaderboard", dependencies=[Depends(get_api_key)])
async def get_leaderboard(
    domain: Optional[str] = None,
    limit: int = 50
):
    """
    Get the SIMTEST leaderboard, optionally filtered by domain.
    """
    # Placeholder: would query aggregated results
    return {
        "domain": domain or "all",
        "updated_at": datetime.utcnow().isoformat(),
        "entries": []
    }

