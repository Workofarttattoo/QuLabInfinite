"""
Master MCP Server for QuLabInfinite.

Exposes a unified tool registry spanning materials, chemistry, physics, and
general utility tools via a single HTTP interface usable by any LLM runtime
(OpenAI-compatible, Anthropic-compatible, or local runners). Designed for
experimentation with lightweight auth, quotas, and structured tool metadata.
"""

from __future__ import annotations

import importlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# --- Configuration -----------------------------------------------------------------
API_KEY = os.getenv("MASTER_MCP_API_KEY")  # If unset, no auth is enforced.
RATE_LIMIT_PER_MIN = int(os.getenv("MASTER_MCP_RATE_LIMIT", "60"))


# --- Models ------------------------------------------------------------------------
class ToolCall(BaseModel):
    tool: str
    args: Dict[str, Any] = {}


class ToolMetadata(BaseModel):
    name: str
    description: str
    category: str
    input_schema: Dict[str, Any] = {}
    cost_tokens: int = 0


@dataclass
class Tool:
    name: str
    description: str
    category: str
    import_path: str
    cost_tokens: int = 0
    input_schema: Dict[str, Any] = field(default_factory=dict)

    def load(self) -> Callable[..., Any]:
        module_name, func_name = self.import_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, func_name)


# --- Tool Registry -----------------------------------------------------------------
REGISTRY: Dict[str, Tool] = {
    # Materials
    "materials.get_database_info": Tool(
        name="materials.get_database_info",
        description="Summarize the materials database and stats.",
        category="materials",
        import_path="materials_lab.qulab_ai_integration.get_materials_database_info",
        cost_tokens=5,
        input_schema={"type": "object", "properties": {}, "required": []},
    ),
    # Chemistry
    "chemistry.validate_smiles": Tool(
        name="chemistry.validate_smiles",
        description="Validate a SMILES string and return parsed details.",
        category="chemistry",
        import_path="chemistry_lab.qulab_ai_integration.validate_smiles",
        cost_tokens=1,
        input_schema={
            "type": "object",
            "properties": {"smiles": {"type": "string"}},
            "required": ["smiles"],
        },
    ),
    # Physics
    "physics.get_element_properties": Tool(
        name="physics.get_element_properties",
        description="Retrieve element properties from the physics engine.",
        category="physics",
        import_path="physics_engine.thermodynamics.get_element_properties",
        cost_tokens=2,
        input_schema={
            "type": "object",
            "properties": {"element": {"type": "string"}},
            "required": ["element"],
        },
    ),
    # Utility / Math
    "ai.calc": Tool(
        name="ai.calc",
        description="Deterministic calculator for simple math expressions.",
        category="utility",
        import_path="qulab_ai.tools.calc",
        cost_tokens=1,
        input_schema={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    ),
}


# --- Auth & Rate Limit -------------------------------------------------------------
def require_api_key(request: Request):
    if not API_KEY:
        return
    provided = request.headers.get("x-api-key")
    if provided != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


class RateLimiter:
    def __init__(self, max_per_min: int):
        self.max_per_min = max_per_min
        self.tokens: Dict[str, List[float]] = {}

    def check(self, identifier: str):
        if self.max_per_min <= 0:
            return
        now = time.time()
        window_start = now - 60
        bucket = self.tokens.setdefault(identifier, [])
        # Drop old
        while bucket and bucket[0] < window_start:
            bucket.pop(0)
        if len(bucket) >= self.max_per_min:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        bucket.append(now)


rate_limiter = RateLimiter(RATE_LIMIT_PER_MIN)


def rate_limit_dep(request: Request):
    key = request.headers.get("x-api-key") or "anon"
    rate_limiter.check(key)


# --- FastAPI App -------------------------------------------------------------------
app = FastAPI(title="Master MCP Server", version="0.1.0")


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.time() - start_time:.4f}s"
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tools", response_model=List[ToolMetadata], dependencies=[Depends(require_api_key)])
def list_tools():
    return [
        ToolMetadata(
            name=tool.name,
            description=tool.description,
            category=tool.category,
            input_schema=tool.input_schema,
            cost_tokens=tool.cost_tokens,
        )
        for tool in REGISTRY.values()
    ]


@app.post("/invoke", dependencies=[Depends(require_api_key), Depends(rate_limit_dep)])
def invoke_tool(payload: ToolCall):
    tool = REGISTRY.get(payload.tool)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    try:
        func = tool.load()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to load tool: {exc}")

    try:
        result = func(**payload.args)
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid arguments: {exc}")
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Tool execution error: {exc}")

    return {"tool": tool.name, "result": result}


# --- Entry point -------------------------------------------------------------------
def main():
    import uvicorn

    port = int(os.getenv("MASTER_MCP_PORT", "8001"))
    uvicorn.run("master_mcp_server:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
"""
Master MCP Server for QuLabInfinite

Goals
- Expose a single MCP-compatible HTTP surface that fronts lab tools across domains
  (materials, chemistry, physics, ech0, generic AI utilities).
- Keep auth/rate-limits/paywall concerns in one place.
- Be LLM-agnostic: any LLM that can hit HTTP+JSON can drive these tools.

Notes
- This is a pragmatic FastAPI-based implementation (already a dependency).
- Tool registry is explicit and typed; extend `TOOL_SPECS` to add more labs.
- Lightweight rate limiting and optional API key auth are included.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

# --- Domain imports (kept narrow to avoid heavy imports) ---
from qulab_ai.tools import calc
from physics_engine.thermodynamics import get_element_properties
from chemistry_lab.qulab_ai_integration import (
    analyze_molecule_with_provenance,
    batch_analyze_molecules,
    validate_smiles,
)
from materials_lab.qulab_ai_integration import (
    analyze_structure_with_provenance,
    batch_analyze_structures,
    validate_structure_file,
    get_materials_database_info,
)


# --- Auth / Rate limit helpers ---
API_KEY_ENV = "MCP_API_KEY"
RATE_LIMIT_PER_MIN = int(os.getenv("MCP_RATE_LIMIT_PER_MIN", "120"))


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> str:
    """Simple header-based API key check (optional if env not set)."""
    required = os.getenv(API_KEY_ENV)
    if required and x_api_key != required:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key or "anon"


_rate_buckets: Dict[str, deque] = defaultdict(deque)


def enforce_rate_limit(api_key: str) -> None:
    bucket = _rate_buckets[api_key]
    now = time.time()
    # drop entries older than 60s
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({RATE_LIMIT_PER_MIN}/min). Try again later.",
        )
    bucket.append(now)


# --- Tool Registry ---
@dataclass
class ToolSpec:
    name: str
    func: Callable[..., Any]
    description: str
    cost_tokens: int = 0
    lite_allowed: bool = False
    schema: Dict[str, str] = field(default_factory=dict)


TOOL_SPECS: List[ToolSpec] = [
    # AI utility
    ToolSpec(
        name="ai.calc",
        func=calc,
        description="Safe numeric calculator for simple expressions.",
        cost_tokens=1,
        lite_allowed=True,
        schema={"expr": "str"},
    ),
    # Physics
    ToolSpec(
        name="physics.get_element_properties",
        func=get_element_properties,
        description="Lookup element properties from physics_engine.",
        cost_tokens=2,
        lite_allowed=True,
        schema={"element_symbol": "str"},
    ),
    # Chemistry
    ToolSpec(
        name="chemistry.validate_smiles",
        func=validate_smiles,
        description="Validate a SMILES string.",
        cost_tokens=1,
        lite_allowed=True,
        schema={"smiles": "str"},
    ),
    ToolSpec(
        name="chemistry.analyze_molecule",
        func=analyze_molecule_with_provenance,
        description="Analyze molecule with provenance metadata.",
        cost_tokens=15,
        lite_allowed=False,
        schema={"smiles": "str", "citations": "Optional[list[str]]"},
    ),
    ToolSpec(
        name="chemistry.batch_analyze_molecules",
        func=batch_analyze_molecules,
        description="Batch analyze molecules from a list of SMILES strings.",
        cost_tokens=25,
        lite_allowed=False,
        schema={"smiles_list": "List[str]", "citations": "Optional[list[str]]"},
    ),
    # Materials
    ToolSpec(
        name="materials.analyze_structure",
        func=analyze_structure_with_provenance,
        description="Analyze structure file with provenance.",
        cost_tokens=15,
        lite_allowed=False,
        schema={"file_path": "str", "citations": "Optional[list[str]]"},
    ),
    ToolSpec(
        name="materials.batch_analyze_structures",
        func=batch_analyze_structures,
        description="Batch analyze multiple structure files.",
        cost_tokens=30,
        lite_allowed=False,
        schema={"file_paths": "List[str]"},
    ),
    ToolSpec(
        name="materials.validate_structure_file",
        func=validate_structure_file,
        description="Validate structure file content.",
        cost_tokens=5,
        lite_allowed=True,
        schema={"file_path": "str"},
    ),
    ToolSpec(
        name="materials.get_database_info",
        func=get_materials_database_info,
        description="Return materials database metadata/stats.",
        cost_tokens=2,
        lite_allowed=True,
        schema={},
    ),
]

TOOL_REGISTRY: Dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}


# --- FastAPI app ---
app = FastAPI(
    title="QuLabInfinite Master MCP Server",
    version="1.0.0",
    description="Unified MCP surface for lab tools (materials, chemistry, physics, AI utilities).",
)


class ToolCall(BaseModel):
    tool: str
    args: Dict[str, Any] = {}
    lite: bool = False


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/tools")
def list_tools() -> List[Dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "cost_tokens": spec.cost_tokens,
            "lite_allowed": spec.lite_allowed,
            "schema": spec.schema,
        }
        for spec in TOOL_SPECS
    ]


@app.post("/call")
def call_tool(payload: ToolCall, api_key: str = Depends(require_api_key)):
    enforce_rate_limit(api_key)

    if payload.tool not in TOOL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {payload.tool}")

    spec = TOOL_REGISTRY[payload.tool]
    if payload.lite and not spec.lite_allowed:
        raise HTTPException(status_code=402, detail="Tool not available on lite plan.")

    try:
        result = spec.func(**payload.args)
        return {"tool": payload.tool, "result": result, "cost_tokens": spec.cost_tokens}
    except Exception as exc:  # pragma: no cover - runtime guardrail
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    import uvicorn

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        print("[warn] MCP_API_KEY not set; server will allow anonymous access.")

    uvicorn.run(
        "master_mcp_server:app",
        host=os.getenv("MCP_HOST", "0.0.0.0"),
        port=int(os.getenv("MCP_PORT", "8000")),
        reload=os.getenv("MCP_RELOAD", "false").lower() == "true",
    )


if __name__ == "__main__":
    main()
