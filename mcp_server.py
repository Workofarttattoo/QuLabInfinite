"""
QuLabInfinite MCP Server

This module wraps QuLabInfinite labs, experiments, and materials databases so
they can be plugged directly into any AI assistant that supports the Model
Context Protocol (MCP). It keeps the application logic untouched while exposing
it through a small dispatcher, a tool manifest for discovery, and a
"cartographer" that inventories the available experiments and datasets.

Key features
------------
- Lite/paid access model with token-aware tool dispatch.
- Automatic tool manifest describing every callable MCP tool with pricing and
  provenance data.
- Cartography helpers to map labs/experiments on disk without importing heavy
  modules.
- Materials Project tracking so we always reference the freshest MP data by ID
  (e.g., "mp-149").
"""

import datetime
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

# Placeholder for user management, token counting, and payment logic.
# This will be implemented in subsequent steps.
class UserAccount:
    def __init__(self, user_id):
        self.user_id = user_id
        self.access_level = "lite"  # "lite" or "paid"
        self.tool_calls_remaining = 20
        self.tokens = 0
        self.trial_start_date = datetime.datetime.utcnow()

    @property
    def trial_days_remaining(self):
        elapsed = datetime.datetime.utcnow() - self.trial_start_date
        return max(0, 3 - elapsed.days)

    def has_access(self, tool_name: str, tool_cost_tokens: int = 0):
        """Check if user has access to a specific tool."""
        if self.access_level == "paid" and self.tokens >= tool_cost_tokens:
            return True, "Access granted."

        if self.access_level == "lite":
            if self.trial_days_remaining <= 0:
                return False, "Trial period has expired. Please upgrade to a paid plan."
            if self.tool_calls_remaining <= 0:
                return False, "Tool call limit reached. Please upgrade to a paid plan."
            if tool_name not in LITE_TIER_TOOLS:
                return False, f"Tool '{tool_name}' is not available on the lite plan. Please upgrade for full access."
            return True, f"Access granted. {self.tool_calls_remaining} calls remaining."

        return False, "Access denied. Please purchase tokens."

    def use_tool(self, tool_cost_tokens: int = 0):
        if self.access_level == "lite":
            self.tool_calls_remaining -= 1
        elif self.access_level == "paid":
            self.tokens -= tool_cost_tokens

# --- Lite Tier Definition ---
LITE_TIER_TOOLS = {
    # tool_name: calls_allowed
    "ech0.analyze_material": 1,
    "materials.analyze_structure": 2,
    "materials.get_database_info": 2,
    "chemistry.analyze_molecule": 2,
    "chemistry.validate_smiles": 5,
    "quantum.run_simulation": 1, # Assuming a quantum lab tool
    "ai.calc": 5,
    "physics.get_element_properties": 2,
}

# --- Token Pricing Model ---
TOKEN_COSTS = {
    # Tier 1: Data & Validation (1-5 tokens)
    "chemistry.validate_smiles": 1,
    "materials.get_database_info": 5,
    "physics.get_element_properties": 2,

    # Tier 2: Basic Analysis & Calculations (10-20 tokens)
    "materials.analyze_structure": 15,
    "chemistry.analyze_molecule": 15,
    "ai.calc": 10,

    # Tier 3: Advanced Analysis & Simple Simulations (50-100 tokens)
    "quantum.run_vqe_simulation": 75, # Placeholder for a quantum tool
    "chemistry.create_water_box": 50,
    "materials.batch_analyze_structures": 10, # Per file, logic to be handled in the tool

    # Tier 4: Complex Simulations & Ech0 Engine (200-500+ tokens)
    "ech0.optimize_design": 300,
    "ech0.quick_invention": 500, # Example cost
}

# --- Custom Exceptions ---
class PaymentRequiredException(Exception):
    def __init__(self, message, payment_url):
        super().__init__(message)
        self.payment_url = payment_url

# --- Payment Gateway (Placeholder) ---
def generate_payment_link(user_id: str, amount: int = 5000) -> str:
    """
    Generates a mock payment link for a user to purchase tokens.
    In a real implementation, this would call the Stripe API.
    """
    return f"https://example.com/pay?user_id={user_id}&amount={amount}"

# In a real application, you would have a persistent user database.
# For this example, we'll use a simple in-memory dictionary.
USER_ACCOUNTS = {
    "user_lite_1": UserAccount("user_lite_1"),
    "user_paid_1": UserAccount("user_paid_1"),
}
USER_ACCOUNTS["user_paid_1"].access_level = "paid"
USER_ACCOUNTS["user_paid_1"].tokens = 10000


def get_user(user_id: str) -> UserAccount:
    """Retrieves a user account."""
    return USER_ACCOUNTS.get(user_id)


def payment_webhook_handler(payload: dict):
    """
    Handles incoming webhooks from the payment provider.
    This function is called when a payment is successfully processed.
    """
    user_id = payload.get("user_id")
    amount_purchased = payload.get("amount") # This would be tokens or currency amount

    user = get_user(user_id)
    if not user:
        print(f"Webhook received for unknown user: {user_id}")
        return {"status": "error", "message": "User not found"}

    # In a real system, you'd convert currency to tokens.
    # Here, we'll just add the amount as tokens.
    user.tokens += amount_purchased
    user.access_level = "paid"
    
    print(f"User {user_id} purchased {amount_purchased} tokens. Account upgraded to paid.")
    
    return {"status": "success", "user_id": user_id, "new_token_balance": user.tokens}


# --- Function Imports from QuLabInfinite Application ---

# It's important to ensure that the QuLabInfinite project is in the PYTHONPATH.

# api
from api.ech0_bridge import main as ech0_bridge_main
from api.hardware_feasibility import *
from api.hardware_integration import *
from api.phase_bloch import *
from api.production_api import *
from api.qulab_api import *
from api.qulab_extended import *
from api.scaling_studies import *
from api.secure_production_api import *
from api.teleport import *

# ech0 interfaces
from ech0_interface import ech0_analyze_material, ech0_design_selector
from ech0_quantum_tools import ech0_filter_inventions, ech0_optimize_design
from ech0_qulab_ai_tools import call_ech0_with_tools, execute_tool_call, ech0_interactive_session
from ech0_invention_accelerator import ech0_quick_invention

# materials_lab
from materials_lab.qulab_ai_integration import (
    analyze_structure_with_provenance,
    batch_analyze_structures,
    validate_structure_file,
    get_materials_database_info,
)
from materials_lab.elemental_data_builder import create_elemental_database
from materials_lab.demo_materials_database import (
    demo_basic_lookup,
    demo_category_search,
    demo_property_search,
    demo_comparison,
    demo_best_for_application,
    demo_database_stats,
    demo_material_details,
    demo_cost_analysis,
)

# chemistry_lab
from chemistry_lab.qulab_ai_integration import (
    analyze_molecule_with_provenance,
    batch_analyze_molecules,
    validate_smiles,
)
from chemistry_lab.molecular_dynamics import create_water_box
from chemistry_lab.datasets.registry import list_datasets, get_dataset

# physics_engine
from physics_engine.mechanics import (
    spring_force,
    damped_spring_force
)
from physics_engine.thermodynamics import get_element_properties
from physics_engine.physics_core import create_benchmark_simulation

# qulab_ai
from qulab_ai.tools import calc
from qulab_ai.uq import conformal_interval, mc_dropout_like
from qulab_ai.parsers.structures import (
    parse_cif,
    parse_poscar,
    parse_xyz,
    parse_pdb,
    parse_structure,
)

# --- Tool Metadata & Cartography ---


@dataclass
class ToolDefinition:
    """Minimal metadata for an MCP-exposed tool."""

    name: str
    description: str
    category: str
    access: str = "paid"
    token_cost: int = 0
    handler: Optional[Callable] = None

    def to_dict(self) -> Dict[str, str]:
        """Serialize to a JSON-friendly dict without the handler."""
        data = asdict(self)
        data.pop("handler", None)
        return data


@dataclass
class ExperimentDefinition:
    """Lightweight description of a lab experiment module on disk."""

    name: str
    path: str
    category: str = "lab"

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def _load_latest_materials_project_entries():
    """Load the freshest Materials Project expansion file and expose mp IDs."""

    data_dir = Path(__file__).parent / "materials_lab" / "data"
    json_path = data_dir / "materials_project_expansion.json"
    jsonl_path = data_dir / "materials_project_expansion.jsonl"

    if json_path.exists():
        with json_path.open("r") as f:
            raw = json.load(f)
        entries = []
        for name, payload in raw.items():
            mp_id = payload.get("material_id") or payload.get("provenance", {}).get("extra", {}).get("material_id")
            entries.append({"name": name, "material_id": mp_id or "unknown"})
    elif jsonl_path.exists():
        entries = []
        with jsonl_path.open("r") as f:
            for line in f:
                try:
                    payload = json.loads(line)
                    entries.append({"name": payload.get("substance", "unknown"), "material_id": payload.get("material_id", "unknown")})
                except json.JSONDecodeError:
                    continue
    else:
        entries = []

    return {
        "count": len(entries),
        "entries": entries,
        "latest_mp_ids": [entry["material_id"] for entry in entries if entry.get("material_id")],
        "source": str(json_path if json_path.exists() else jsonl_path),
        "last_modified": max(
            (os.path.getmtime(p) for p in [json_path, jsonl_path] if Path(p).exists()),
            default=None,
        ),
    }


def discover_experiments(root: Path = Path(__file__).parent) -> List[ExperimentDefinition]:
    """
    Map available lab experiment modules without importing them.

    We scan for files ending in ``_lab.py`` as well as directories ending in
    ``_lab`` so MCP clients can expose them as runnable experiments.
    """

    experiments: List[ExperimentDefinition] = []

    for path in root.glob("*_lab.py"):
        experiments.append(ExperimentDefinition(name=path.stem, path=str(path)))

    for path in root.glob("*_lab"):
        experiments.append(ExperimentDefinition(name=path.name, path=str(path)))

    # Include dedicated experiment demos
    for path in root.glob("*experiment*.py"):
        experiments.append(ExperimentDefinition(name=path.stem, path=str(path), category="experiment"))

    # Deduplicate by name while keeping first occurrence
    seen = set()
    unique: List[ExperimentDefinition] = []
    for exp in experiments:
        if exp.name not in seen:
            unique.append(exp)
            seen.add(exp.name)

    return unique


# --- MCP Tool Call Definitions ---

# The following functions wrap the imported application logic, making them
# available as MCP tool calls. In a real implementation, these would be
# registered with the MCP server.

class Ech0EngineTools:
    """Tools related to the Ech0 Engine."""

    @staticmethod
    def analyze_material(material_name: str) -> str:
        """Analyzes a material using the Ech0 engine."""
        # User auth and payment logic would go here.
        return ech0_analyze_material(material_name)

    @staticmethod
    def design_selector(application: str, budget_per_kg: float = 100.0) -> str:
        """Selects a material design for an application based on budget."""
        return ech0_design_selector(application, budget_per_kg)

    @staticmethod
    def filter_inventions(inventions: list, top_n: int = 10) -> list:
        """Filters a list of inventions."""
        return ech0_filter_inventions(inventions, top_n)

    @staticmethod
    def optimize_design(design: dict) -> dict:
        """Optimizes a given design."""
        return ech0_optimize_design(design)

    # ... and so on for all other Ech0 functions.

class MaterialsLabTools:
    """Tools related to the Materials Lab."""

    @staticmethod
    def analyze_structure(file_path: str, citations: list = None) -> dict:
        """Analyzes a structure file and attaches provenance."""
        return analyze_structure_with_provenance(file_path, citations)

    @staticmethod
    def batch_analyze_structures(file_paths: list) -> list:
        """Analyzes a batch of structure files."""
        return batch_analyze_structures(file_paths)

    @staticmethod
    def get_database_info() -> dict:
        """Gets information about the materials database."""
        info = get_materials_database_info()
        mp_snapshot = _load_latest_materials_project_entries()

        info["materials_project"] = {
            "count": mp_snapshot["count"],
            "latest_mp_ids": mp_snapshot["latest_mp_ids"][:5],
            "source": mp_snapshot["source"],
            "last_modified": mp_snapshot["last_modified"],
        }
        return info

class ChemistryLabTools:
    """Tools related to the Chemistry Lab."""

    @staticmethod
    def analyze_molecule(smiles: str, citations: list = None) -> dict:
        """Analyzes a molecule from a SMILES string."""
        return analyze_molecule_with_provenance(smiles, citations)

    @staticmethod
    def validate_smiles(smiles: str) -> dict:
        """Validates a SMILES string."""
        return validate_smiles(smiles)

    @staticmethod
    def create_water_box(n_molecules: int, box_size: float = 30.0) -> tuple:
        """Creates a box of water molecules for simulation."""
        return create_water_box(n_molecules, box_size)

class PhysicsEngineTools:
    """Tools related to the Physics Engine."""

    @staticmethod
    def get_element_properties(element_symbol: str) -> dict:
        """Gets properties for a chemical element."""
        return get_element_properties(element_symbol)

class QulabAITools:
    """General purpose AI tools."""

    @staticmethod
    def calc(expr: str) -> float:
        """A simple calculator tool."""
        return calc(expr)


# Example of how the tool calls could be invoked through a unified dispatcher
# This is a conceptual example. The final implementation will depend on the MCP server framework.
def call_tool(user: UserAccount, tool_name: str, **kwargs):
    """
    Dispatcher to call the appropriate tool.
    Example: call_tool(user, "materials.analyze_structure", file_path="...")
    """
    try:
        # Look up the token cost for the tool. Default to 0 if not priced.
        token_cost = TOKEN_COSTS.get(tool_name, 0)

        has_access, message = user.has_access(tool_name, token_cost)
        if not has_access:
            # Generate a payment link and raise an exception.
            payment_url = generate_payment_link(user.user_id)
            raise PaymentRequiredException(message, payment_url)

        parts = tool_name.split('.')
        if len(parts) != 2:
            raise ValueError("Invalid tool name format. Use 'module.tool_name'.")
        
        module_name, function_name = parts
        
        tool_classes = {
            "ech0": Ech0EngineTools,
            "materials": MaterialsLabTools,
            "chemistry": ChemistryLabTools,
            "physics": PhysicsEngineTools,
            "ai": QulabAITools,
        }

        if module_name not in tool_classes:
            raise ValueError(f"Unknown tool module: {module_name}")
            
        tool_class = tool_classes[module_name]
        
        if not hasattr(tool_class, function_name):
            raise ValueError(f"Unknown tool '{function_name}' in module '{module_name}'")
        
        # Decrement user's remaining calls or tokens
        user.use_tool(token_cost)
        
        return getattr(tool_class, function_name)(**kwargs)
    except PaymentRequiredException as e:
        # The server would catch this and return a 402 Payment Required response
        # with the payment URL in the body.
        print(f"Payment required: {e}. Please visit {e.payment_url}")
        return {"error": str(e), "payment_url": e.payment_url}


def build_tool_manifest() -> Dict[str, Dict[str, ToolDefinition]]:
    """Create a discoverable manifest of all MCP-exposed tools."""

    manifest: Dict[str, Dict[str, ToolDefinition]] = {
        "ech0": {
            "analyze_material": ToolDefinition(
                name="ech0.analyze_material",
                description="Analyze a material using the Ech0 engine for rapid feasibility checks.",
                category="ech0",
                access="lite",
                token_cost=TOKEN_COSTS.get("ech0.analyze_material", 0),
                handler=Ech0EngineTools.analyze_material,
            ),
            "optimize_design": ToolDefinition(
                name="ech0.optimize_design",
                description="Optimize a proposed design with Ech0 heuristics.",
                category="ech0",
                token_cost=TOKEN_COSTS.get("ech0.optimize_design", 0),
                handler=Ech0EngineTools.optimize_design,
            ),
        },
        "materials": {
            "analyze_structure": ToolDefinition(
                name="materials.analyze_structure",
                description="Parse and analyze a structure file with provenance.",
                category="materials",
                access="lite",
                token_cost=TOKEN_COSTS.get("materials.analyze_structure", 0),
                handler=MaterialsLabTools.analyze_structure,
            ),
            "batch_analyze_structures": ToolDefinition(
                name="materials.batch_analyze_structures",
                description="Batch analyze multiple structures (token cost is per file).",
                category="materials",
                token_cost=TOKEN_COSTS.get("materials.batch_analyze_structures", 0),
                handler=MaterialsLabTools.batch_analyze_structures,
            ),
            "get_database_info": ToolDefinition(
                name="materials.get_database_info",
                description="Inspect the integrated materials database including Materials Project mp- IDs.",
                category="materials",
                access="lite",
                token_cost=TOKEN_COSTS.get("materials.get_database_info", 0),
                handler=MaterialsLabTools.get_database_info,
            ),
        },
        "chemistry": {
            "analyze_molecule": ToolDefinition(
                name="chemistry.analyze_molecule",
                description="Analyze a molecule from a SMILES string with provenance.",
                category="chemistry",
                token_cost=TOKEN_COSTS.get("chemistry.analyze_molecule", 0),
                handler=ChemistryLabTools.analyze_molecule,
            ),
            "validate_smiles": ToolDefinition(
                name="chemistry.validate_smiles",
                description="Validate a SMILES string and return basic properties.",
                category="chemistry",
                access="lite",
                token_cost=TOKEN_COSTS.get("chemistry.validate_smiles", 0),
                handler=ChemistryLabTools.validate_smiles,
            ),
            "create_water_box": ToolDefinition(
                name="chemistry.create_water_box",
                description="Create a water box for molecular dynamics simulations.",
                category="chemistry",
                token_cost=TOKEN_COSTS.get("chemistry.create_water_box", 0),
                handler=ChemistryLabTools.create_water_box,
            ),
        },
        "physics": {
            "get_element_properties": ToolDefinition(
                name="physics.get_element_properties",
                description="Lookup element properties from the physics engine.",
                category="physics",
                access="lite",
                token_cost=TOKEN_COSTS.get("physics.get_element_properties", 0),
                handler=PhysicsEngineTools.get_element_properties,
            ),
        },
        "ai": {
            "calc": ToolDefinition(
                name="ai.calc",
                description="Lightweight calculator exposed as an MCP tool.",
                category="utility",
                access="lite",
                token_cost=TOKEN_COSTS.get("ai.calc", 0),
                handler=QulabAITools.calc,
            ),
        },
    }

    return manifest


def manifest_as_dict(manifest: Dict[str, Dict[str, ToolDefinition]]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Serialize a tool manifest to plain dictionaries suitable for JSON."""

    serialized: Dict[str, Dict[str, Dict[str, str]]] = {}
    for module, tools in manifest.items():
        serialized[module] = {name: tool.to_dict() for name, tool in tools.items()}
    return serialized


def map_capabilities() -> Dict[str, object]:
    """High-level cartography of tools, experiments, and data sources."""

    manifest = build_tool_manifest()
    experiments = discover_experiments()
    materials_project = _load_latest_materials_project_entries()

    return {
        "tools": manifest_as_dict(manifest),
        "experiments": [exp.to_dict() for exp in experiments],
        "data_sources": {
            "materials_project": materials_project,
            "materials_database_files": [
                str(path)
                for path in (Path(__file__).parent / "materials_lab" / "data").glob("materials*_expansion*.json*")
            ],
        },
    }


class McpServer:
    """Lightweight MCP-style dispatcher wrapper for QuLabInfinite."""

    def __init__(self, user_id: str = "user_lite_1"):
        self.user = get_user(user_id) or UserAccount(user_id)
        self.tool_manifest = build_tool_manifest()
        self.experiments = discover_experiments()

    def call(self, tool_name: str, **kwargs):
        return call_tool(self.user, tool_name, **kwargs)

    def describe(self) -> Dict[str, object]:
        return {
            "tools": manifest_as_dict(self.tool_manifest),
            "experiments": [exp.to_dict() for exp in self.experiments],
            "user": {
                "id": self.user.user_id,
                "access_level": self.user.access_level,
            },
        }


# This is a representative subset of the tool call mappings.
# The full implementation will include wrappers for all 316+ functions.
