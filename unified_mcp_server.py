"""
Unified QuLabInfinite MCP server

This FastAPI-based server exposes a curated registry of QuLabInfinite tools so any
AI agent using the Model Context Protocol can call into the labs. It keeps a live
map of available tools/experiments and always serves material data from the
freshest Materials Project expansion (mp-*-style records).
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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


@dataclass
class Tool:
    """Metadata for a callable tool."""

    name: str
    func: Callable[..., Any]
    description: str
    module: str
    cost_tokens: int = 0
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        signature = inspect.signature(self.func)
        return {
            "name": self.name,
            "module": self.module,
            "description": self.description,
            "cost_tokens": self.cost_tokens,
            "tags": self.tags,
            "parameters": [
                {
                    "name": param_name,
                    "kind": str(param.kind),
                    "default": None if param.default is inspect._empty else param.default,
                    "annotation": str(param.annotation),
                }
                for param_name, param in signature.parameters.items()
            ],
        }


class MaterialsDataset:
    """Load the freshest Materials Project expansion dataset (mp-*) records."""

    def __init__(self, dataset_path: Optional[Path] = None):
        self.dataset_path = dataset_path or Path(
            "materials_lab/data/materials_project_expansion.jsonl"
        )
        self.records: Dict[str, Dict[str, Any]] = {}
        self.latest_timestamp: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Materials dataset not found at {self.dataset_path}. "
                "Please regenerate the expansion JSONL before starting the MCP server."
            )

        with self.dataset_path.open("r") as handle:
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
            raise HTTPException(
                status_code=404,
                detail=f"Material '{mp_id}' not found in the freshest mp dataset",
            ) from exc


class ToolRegistry:
    """Registry that maps tool names to callables and metadata."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def list_tools(self) -> List[Dict[str, Any]]:
        return [tool.to_dict() for tool in self._tools.values()]

    def cartography(self) -> Dict[str, List[Dict[str, Any]]]:
        mapped: Dict[str, List[Dict[str, Any]]] = {}
        for tool in self._tools.values():
            mapped.setdefault(tool.module, []).append(tool.to_dict())
        for module_tools in mapped.values():
            module_tools.sort(key=lambda entry: entry["name"])
        return mapped

    def call(self, tool_name: str, **kwargs: Any) -> Any:
        if tool_name not in self._tools:
            raise HTTPException(status_code=404, detail=f"Unknown tool '{tool_name}'")
        tool = self._tools[tool_name]
        return tool.func(**kwargs)


class ToolInvocationRequest(BaseModel):
    tool: str
    params: Dict[str, Any] = {}


class ExperimentRecord(BaseModel):
    name: str
    path: str
    description: str
    entry_point: Optional[str] = None


def build_registry(materials_dataset: MaterialsDataset) -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        Tool(
            name="materials.get_mp_material",
            func=materials_dataset.get_material,
            description="Return the freshest mp-* record from the Materials Project expansion dataset.",
            module="materials",
            tags=["materials", "database", "mp"],
        )
    )
    registry.register(
        Tool(
            name="materials.analyze_structure",
            func=analyze_structure_with_provenance,
            description="Analyze a structure file with provenance tracking.",
            module="materials",
        )
    )
    registry.register(
        Tool(
            name="materials.batch_analyze_structures",
            func=batch_analyze_structures,
            description="Analyze a batch of structure files.",
            module="materials",
        )
    )
    registry.register(
        Tool(
            name="materials.database_info",
            func=get_materials_database_info,
            description="Get metadata about the materials database (coverage, fields, stats).",
            module="materials",
        )
    )

    registry.register(
        Tool(
            name="chemistry.analyze_molecule",
            func=analyze_molecule_with_provenance,
            description="Analyze a molecule specified by SMILES, returning provenance-aware annotations.",
            module="chemistry",
        )
    )
    registry.register(
        Tool(
            name="chemistry.batch_analyze_molecules",
            func=batch_analyze_molecules,
            description="Run batched molecule analysis for SMILES lists.",
            module="chemistry",
        )
    )
    registry.register(
        Tool(
            name="chemistry.validate_smiles",
            func=validate_smiles,
            description="Validate SMILES syntax before downstream computations.",
            module="chemistry",
        )
    )
    registry.register(
        Tool(
            name="chemistry.create_water_box",
            func=create_water_box,
            description="Create a water box for MD simulations with configurable size.",
            module="chemistry",
            tags=["md", "simulation"],
        )
    )

    registry.register(
        Tool(
            name="physics.get_element_properties",
            func=get_element_properties,
            description="Return thermodynamic properties for an element symbol.",
            module="physics",
        )
    )
    registry.register(
        Tool(
            name="physics.create_benchmark_simulation",
            func=create_benchmark_simulation,
            description="Create a physics benchmark scenario for downstream simulation runs.",
            module="physics",
            tags=["simulation"],
        )
    )

    registry.register(
        Tool(
            name="ai.calc",
            func=calc,
            description="Lightweight calculator for quick numeric expressions.",
            module="ai",
        )
    )

    registry.register(
        Tool(
            name="ech0.analyze_material",
            func=ech0_analyze_material,
            description="Run the Ech0 engine's material analysis pipeline.",
            module="ech0",
            cost_tokens=300,
        )
    )
    registry.register(
        Tool(
            name="ech0.design_selector",
            func=ech0_design_selector,
            description="Select candidate material designs for a target application and budget.",
            module="ech0",
            cost_tokens=300,
        )
    )
    registry.register(
        Tool(
            name="ech0.filter_inventions",
            func=ech0_filter_inventions,
            description="Filter and rank inventions discovered across lab runs.",
            module="ech0",
            cost_tokens=150,
        )
    )
    registry.register(
        Tool(
            name="ech0.optimize_design",
            func=ech0_optimize_design,
            description="Optimize an invention or material design using Ech0 heuristics.",
            module="ech0",
            cost_tokens=300,
        )
    )
    registry.register(
        Tool(
            name="ech0.quick_invention",
            func=ech0_quick_invention,
            description="Rapid invention generator for proofs of concept.",
            module="ech0",
            cost_tokens=500,
        )
    )

    return registry


EXPERIMENTS: List[ExperimentRecord] = [
    ExperimentRecord(
        name="oncology.demo_experiment",
        path="demo_experiment.py",
        description="Demonstration of calibrated tumor lab scenarios (chemo vs Ech0 protocol).",
    ),
    ExperimentRecord(
        name="materials.validation_suite",
        path="test_full_6_6m_materials.py",
        description="Automated validation harness for the expanded materials dataset.",
    ),
    ExperimentRecord(
        name="chemistry.expanded_database",
        path="test_expanded_database.py",
        description="Smoke tests that exercise the chemistry ingestion and validation stack.",
    ),
    ExperimentRecord(
        name="physics.benchmarks",
        path="physics_engine/physics_core.py",
        description="Core physics benchmark scenarios callable via the MCP API.",
        entry_point="create_benchmark_simulation",
    ),
]


app = FastAPI(title="QuLabInfinite MCP Server", version="2.0.0")
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


@app.post("/tools/call")
def call_tool(request: ToolInvocationRequest) -> Any:
    return registry.call(request.tool, **request.params)


@app.get("/map")
def map_everything() -> Dict[str, Any]:
    return {
        "cartography": registry.cartography(),
        "experiments": [experiment.model_dump() for experiment in EXPERIMENTS],
        "materials_dataset": materials_dataset.summary(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8102)
