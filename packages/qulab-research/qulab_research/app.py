"""
ECH0 Autonomous Research Service — FastAPI application.

Endpoints:
  GET  /health
  POST /research/invention
  POST /research/materials
  POST /research/question
  POST /research/experiment_design
  GET  /research/domains

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

import os
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from qulab_research.research_engine import ResearchEngine, DOMAIN_MAP

# ---------------------------------------------------------------------------
# Engine singleton (shared across requests)
# ---------------------------------------------------------------------------

_LAB_ROOT_ENV = os.environ.get("QULAB_LAB_ROOT")
_lab_root = Path(_LAB_ROOT_ENV) if _LAB_ROOT_ENV else None

engine = ResearchEngine(lab_root=_lab_root)

BACKEND_URL = os.environ.get("QULAB_BACKEND_URL", "http://127.0.0.1:8000")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ECH0 Autonomous Research Service",
    description=(
        "Pay-per-use research API powered by ECH0's invention acceleration "
        "engine and QuLab Infinite scientific simulation labs."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class InventionRequest(BaseModel):
    goal: str = Field(..., description="Research goal or objective")
    domain: str = Field("materials", description="Scientific domain")
    concepts: list[str] = Field(..., description="List of invention concept names")
    max_concepts: int = Field(10, ge=1, le=50, description="Maximum concepts to process")


class MaterialsRequest(BaseModel):
    category: Optional[str] = Field(None, description="Material category (metal, ceramic, polymer, composite, nanomaterial)")
    max_density_kg_m3: Optional[float] = Field(None, description="Maximum density (kg/m³)")
    min_tensile_strength_MPa: Optional[float] = Field(None, description="Minimum tensile strength (MPa)")
    max_cost_per_kg: Optional[float] = Field(None, description="Maximum cost per kg (USD)")
    application: Optional[str] = Field(None, description="Application context (e.g. aerospace structural)")


class QuestionRequest(BaseModel):
    question: str = Field(..., description="Research question to answer")
    domain: str = Field("physics", description="Scientific domain")


class ExperimentDesignRequest(BaseModel):
    objective: str = Field(..., description="Experimental objective")
    variables: list[str] = Field(..., description="Variable names to explore")
    response: str = Field(..., description="Response variable being measured")
    n_runs: int = Field(16, ge=4, le=256, description="Number of experimental runs")
    variable_ranges: Optional[dict[str, list[float]]] = Field(
        None,
        description="Optional dict of {variable: [min, max]} ranges. Defaults to [0, 1] if not provided.",
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {
        "status": "ok",
        "lab_root": str(engine.lab_root) if engine.lab_root else None,
        "engine": engine.status,
        "backend_url": BACKEND_URL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ------------------------------------------------------------------
# POST /research/invention
# ------------------------------------------------------------------

@app.post("/research/invention")
def invention(req: InventionRequest):
    """
    Run ECH0 invention acceleration pipeline.

    Generates ranked inventions using quantum superposition scoring,
    physics/chemistry validation, and materials database lookup.
    """
    report = engine.run_invention_session(
        goal=req.goal,
        domain=req.domain,
        concepts=req.concepts,
        max_concepts=req.max_concepts,
    )
    return report


# ------------------------------------------------------------------
# POST /research/materials
# ------------------------------------------------------------------

@app.post("/research/materials")
def materials(req: MaterialsRequest):
    """
    Discover materials matching property requirements.

    Returns up to 10 materials ranked by fitness score against the
    specified constraints, drawn from the 1,619-material QuLab database.
    """
    requirements = {
        "category": req.category,
        "max_density_kg_m3": req.max_density_kg_m3,
        "min_tensile_strength_MPa": req.min_tensile_strength_MPa,
        "max_cost_per_kg": req.max_cost_per_kg,
        "application": req.application,
    }
    # Remove None values
    requirements = {k: v for k, v in requirements.items() if v is not None}

    result = engine.run_material_discovery(requirements)
    return result


# ------------------------------------------------------------------
# POST /research/question
# ------------------------------------------------------------------

@app.post("/research/question")
def research_question(req: QuestionRequest):
    """
    Answer a research question by mapping the domain to relevant QuLab labs.

    For domains that require simulation, this endpoint calls the qulab-mcp
    backend via JSON-RPC and aggregates results.
    """
    # Local analysis
    local = engine.run_research_question(req.question, req.domain)

    # Attempt to enrich from qulab-mcp backend
    remote_findings = _call_backend_for_domain(req.question, req.domain)
    if remote_findings:
        local["findings"].extend(remote_findings)
        local["synthesis"] += f" Remote simulations returned {len(remote_findings)} result(s)."

    return local


def _call_backend_for_domain(question: str, domain: str) -> list[dict]:
    """
    Call the qulab-mcp backend via JSON-RPC tools/call for the given domain.
    Returns a list of finding dicts. Empty list on any failure.
    """
    findings: list[dict] = []
    domain_info = DOMAIN_MAP.get(domain, {})
    labs = domain_info.get("labs", [])

    # Map domain labs to known MCP tool names
    tool_name_map = {
        "materials_lab.materials_database": "materials_recommend",
        "quantum_computing_lab": "quantum_circuit",
        "chemistry_lab.chemistry_lab": "molecular_properties",
        "pharma_pk_model": "pharma_pk_model",
        "nano_drug_release": "nano_drug_release",
        "physics_engine.mechanics": "mechanics_simulate",
        "physics_engine.thermodynamics": "thermodynamics",
    }

    for lab in labs:
        tool = tool_name_map.get(lab)
        if not tool:
            continue
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "tools/call",
                "params": {
                    "name": tool,
                    "arguments": {"question": question, "domain": domain},
                },
            }
            resp = httpx.post(
                f"{BACKEND_URL}/",
                json=payload,
                timeout=5.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                result_content = data.get("result", {}).get("content", [])
                for item in result_content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        findings.append({"source": tool, "summary": item["text"][:500]})
        except Exception:
            pass  # Backend unavailable — silently skip

    return findings


# ------------------------------------------------------------------
# POST /research/experiment_design
# ------------------------------------------------------------------

@app.post("/research/experiment_design")
def experiment_design(req: ExperimentDesignRequest):
    """
    Generate a Design of Experiments matrix.

    - ≤3 variables: full factorial design
    - 4+ variables: Latin Hypercube sampling
    """
    variables = req.variables
    n_vars = len(variables)
    n_runs = req.n_runs

    # Resolve ranges
    var_ranges: dict[str, list[float]] = {}
    for v in variables:
        if req.variable_ranges and v in req.variable_ranges:
            lo, hi = req.variable_ranges[v]
        else:
            lo, hi = 0.0, 1.0
        var_ranges[v] = [lo, hi]

    if n_vars <= 3:
        runs = _full_factorial(variables, var_ranges, n_runs)
        design_type = "full_factorial"
    else:
        runs = _latin_hypercube(variables, var_ranges, n_runs)
        design_type = "latin_hypercube"

    return {
        "objective": req.objective,
        "response": req.response,
        "design_type": design_type,
        "n_variables": n_vars,
        "n_runs": len(runs),
        "variables": variables,
        "variable_ranges": var_ranges,
        "runs": runs,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _full_factorial(
    variables: list[str],
    var_ranges: dict[str, list[float]],
    n_runs: int,
) -> list[dict]:
    """
    Full factorial design — 2-level for each variable.
    Returns all 2^k combinations, capped at n_runs.
    """
    n_vars = len(variables)
    levels_per_var = 2
    total = levels_per_var ** n_vars

    runs = []
    for run_idx in range(min(total, n_runs)):
        run: dict[str, Any] = {"run": run_idx + 1}
        for j, var in enumerate(variables):
            lo, hi = var_ranges[var]
            # Gray-code-like: bit j of run_idx → low (0) or high (1)
            level = (run_idx >> j) & 1
            run[var] = lo if level == 0 else hi
        runs.append(run)

    return runs


def _latin_hypercube(
    variables: list[str],
    var_ranges: dict[str, list[float]],
    n_runs: int,
) -> list[dict]:
    """
    Latin Hypercube Sampling.

    Divide each variable's range into n_runs equal intervals,
    randomly permute each variable's assignment, pick midpoint
    of each stratum.
    """
    # Build permutation for each variable
    lhs_matrix: dict[str, list[float]] = {}
    for var in variables:
        lo, hi = var_ranges[var]
        perm = list(range(n_runs))
        random.shuffle(perm)
        interval_width = (hi - lo) / n_runs
        values = [lo + (k + 0.5) * interval_width for k in perm]
        lhs_matrix[var] = values

    runs = []
    for i in range(n_runs):
        run: dict[str, Any] = {"run": i + 1}
        for var in variables:
            run[var] = round(lhs_matrix[var][i], 6)
        runs.append(run)

    return runs


# ------------------------------------------------------------------
# GET /research/domains
# ------------------------------------------------------------------

@app.get("/research/domains")
def domains():
    """
    List supported research domains with descriptions and mapped QuLab labs.
    """
    result = {}
    for domain, info in DOMAIN_MAP.items():
        result[domain] = {
            "description": info["description"],
            "labs": info["labs"],
            "keywords": info["keywords"],
        }
    return {"domains": result, "count": len(result)}
