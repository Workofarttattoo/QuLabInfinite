"""Canonical QuLabInfinite runtime entrypoint.

This module owns tool registration, name-based discovery, and reproducible JSON artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core.runtime import ArtifactWriter, Tool, ToolRegistry

# Core tool imports
from ech0_interface import ech0_analyze_material, ech0_design_selector
from ech0_invention_accelerator import ech0_quick_invention
from ech0_quantum_tools import ech0_filter_inventions, ech0_optimize_design
from materials_lab.qulab_ai_integration import (
    analyze_structure_with_provenance,
    batch_analyze_structures,
    get_materials_database_info,
)
from chemistry_lab.qulab_ai_integration import (
    analyze_molecule_with_provenance,
    batch_analyze_molecules,
    validate_smiles,
)
from chemistry_lab.molecular_dynamics import create_water_box
from physics_engine.physics_core import create_benchmark_simulation
from physics_engine.thermodynamics import get_element_properties
from qulab_ai.tools import calc


class MaterialsDataset:
    """Load the freshest Materials Project expansion dataset (mp-*) records."""

    def __init__(self, dataset_path: Optional[Path] = None):
        self.dataset_path = dataset_path or Path("materials_lab/data/materials_project_expansion.jsonl")
        self.records: Dict[str, Dict[str, Any]] = {}
        self.latest_timestamp: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Materials dataset not found at {self.dataset_path}. "
                "Please regenerate the expansion JSONL before starting the runtime."
            )

        with self.dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                material_id = record.get("material_id") or record.get("mp_id")
                if not material_id:
                    continue
                self.records[material_id] = record
                acquired = record.get("provenance", {}).get("acquired_at")
                if acquired and (self.latest_timestamp is None or acquired > self.latest_timestamp):
                    self.latest_timestamp = acquired

    def summary(self) -> Dict[str, Any]:
        return {
            "dataset_path": str(self.dataset_path),
            "material_count": len(self.records),
            "latest_timestamp": self.latest_timestamp,
            "sample_ids": sorted(list(self.records.keys()))[:5],
        }

    def get_material(self, mp_id: str) -> Dict[str, Any]:
        try:
            return self.records[mp_id]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Material '{mp_id}' not found") from exc


class ToolInvocationRequest(BaseModel):
    tool: str
    params: Dict[str, Any] = {}
    artifact_path: Optional[str] = None


class ExperimentRecord(BaseModel):
    name: str
    path: str
    description: str
    entry_point: Optional[str] = None


def build_registry(materials_dataset: MaterialsDataset) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool("materials.get_mp_material", "materials", "Return latest mp-* record.", materials_dataset.get_material, tags=["materials", "database", "mp"]))
    registry.register(Tool("materials.analyze_structure", "materials", "Analyze structure file with provenance.", analyze_structure_with_provenance))
    registry.register(Tool("materials.batch_analyze_structures", "materials", "Analyze a batch of structure files.", batch_analyze_structures))
    registry.register(Tool("materials.database_info", "materials", "Get materials database metadata.", get_materials_database_info))
    registry.register(Tool("chemistry.analyze_molecule", "chemistry", "Analyze a molecule from SMILES.", analyze_molecule_with_provenance))
    registry.register(Tool("chemistry.batch_analyze_molecules", "chemistry", "Run batched molecule analysis.", batch_analyze_molecules))
    registry.register(Tool("chemistry.validate_smiles", "chemistry", "Validate SMILES syntax.", validate_smiles))
    registry.register(Tool("chemistry.create_water_box", "chemistry", "Create a water box for MD simulations.", create_water_box, tags=["md", "simulation"]))
    registry.register(Tool("physics.get_element_properties", "physics", "Return thermodynamic element properties.", get_element_properties))
    registry.register(Tool("physics.create_benchmark_simulation", "physics", "Create a physics benchmark scenario.", create_benchmark_simulation, tags=["simulation"]))
    registry.register(Tool("ai.calc", "ai", "Lightweight calculator for numeric expressions.", calc))
    registry.register(Tool("ech0.analyze_material", "ech0", "Run Ech0 material analysis pipeline.", ech0_analyze_material, cost_tokens=300))
    registry.register(Tool("ech0.design_selector", "ech0", "Select candidate material designs.", ech0_design_selector, cost_tokens=300))
    registry.register(Tool("ech0.filter_inventions", "ech0", "Filter and rank inventions.", ech0_filter_inventions, cost_tokens=150))
    registry.register(Tool("ech0.optimize_design", "ech0", "Optimize invention/material designs.", ech0_optimize_design, cost_tokens=300))
    registry.register(Tool("ech0.quick_invention", "ech0", "Rapid invention generator.", ech0_quick_invention, cost_tokens=500))
    return registry


EXPERIMENTS: List[ExperimentRecord] = [
    ExperimentRecord(name="oncology.demo_experiment", path="demo_experiment.py", description="Calibrated tumor lab scenarios."),
    ExperimentRecord(name="materials.validation_suite", path="test_full_6_6m_materials.py", description="Validation harness for expanded materials dataset."),
    ExperimentRecord(name="chemistry.expanded_database", path="test_expanded_database.py", description="Smoke tests for chemistry ingestion and validation."),
    ExperimentRecord(name="physics.benchmarks", path="physics_engine/physics_core.py", description="Physics benchmark scenarios via runtime.", entry_point="create_benchmark_simulation"),
]


app = FastAPI(title="QuLabInfinite Runtime", version="3.0.0")
materials_dataset = MaterialsDataset()
registry = build_registry(materials_dataset)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "materials_dataset": materials_dataset.summary(),
        "tool_count": len(registry.list_tools()),
        "experiment_count": len(EXPERIMENTS),
    }


@app.get("/tools")
def list_tools() -> Dict[str, Any]:
    return {
        "tools": registry.list_tools(),
        "cartography": registry.cartography(),
        "experiments": [experiment.model_dump() for experiment in EXPERIMENTS],
    }


@app.get("/tools/{tool_name}")
def discover_tool(tool_name: str) -> Dict[str, Any]:
    if not registry.has(tool_name):
        raise HTTPException(status_code=404, detail=f"Unknown tool '{tool_name}'")
    return registry.discover(tool_name)


@app.post("/tools/call")
def call_tool(request: ToolInvocationRequest) -> Dict[str, Any]:
    if not registry.has(request.tool):
        raise HTTPException(status_code=404, detail=f"Unknown tool '{request.tool}'")

    result = registry.call(request.tool, **request.params)
    artifact = ArtifactWriter.canonical_payload(tool=request.tool, params=request.params, result=result)
    artifact_path = request.artifact_path
    if artifact_path:
        ArtifactWriter.write(Path(artifact_path), artifact)

    return {
        "result": result,
        "artifact": artifact,
        "artifact_path": artifact_path,
    }


def cli() -> int:
    parser = argparse.ArgumentParser(description="Canonical QuLabInfinite runtime entrypoint")
    parser.add_argument("command", choices=["list", "call"], help="Runtime command")
    parser.add_argument("--tool", help="Tool name for call command")
    parser.add_argument("--params", default="{}", help="JSON dict for tool call parameters")
    parser.add_argument("--artifact", help="Path to write deterministic JSON artifact")
    args = parser.parse_args()

    if args.command == "list":
        print(json.dumps({"tools": registry.list_tools()}, indent=2, sort_keys=True))
        return 0

    if not args.tool:
        parser.error("--tool is required for call")

    params = json.loads(args.params)
    result = registry.call(args.tool, **params)
    artifact = ArtifactWriter.canonical_payload(args.tool, params, result)
    if args.artifact:
        ArtifactWriter.write(Path(args.artifact), artifact)

    print(json.dumps({"result": result, "artifact": artifact}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8102)
