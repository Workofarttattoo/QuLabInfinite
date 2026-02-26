"""Canonical runtime primitives for QuLabInfinite tool execution."""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List


@dataclass(frozen=True)
class Tool:
    """Interface every runtime module uses for registration."""

    name: str
    module: str
    description: str
    func: Callable[..., Any]
    cost_tokens: int = 0
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        signature = inspect.signature(self.func)
        return {
            "name": self.name,
            "module": self.module,
            "description": self.description,
            "cost_tokens": self.cost_tokens,
            "tags": sorted(self.tags),
            "parameters": [
                {
                    "name": name,
                    "kind": str(param.kind),
                    "default": None if param.default is inspect._empty else param.default,
                    "annotation": str(param.annotation),
                }
                for name, param in signature.parameters.items()
            ],
        }


class ToolRegistry:
    """Name-addressable registry for all runtime tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def discover(self, name: str) -> Dict[str, Any]:
        return self.get(name).to_dict()

    def list_tools(self) -> List[Dict[str, Any]]:
        return [self._tools[name].to_dict() for name in sorted(self._tools)]

    def cartography(self) -> Dict[str, List[Dict[str, Any]]]:
        by_module: Dict[str, List[Dict[str, Any]]] = {}
        for tool in self.list_tools():
            by_module.setdefault(tool["module"], []).append(tool)
        return by_module

    def call(self, name: str, **params: Any) -> Any:
        return self.get(name).func(**params)


class ArtifactWriter:
    """Persist deterministic JSON artifacts for runtime calls."""

    @staticmethod
    def canonical_payload(tool: str, params: Dict[str, Any], result: Any) -> Dict[str, Any]:
        base = {
            "tool": tool,
            "params": params,
            "result": result,
        }
        canonical = json.dumps(base, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return {
            **base,
            "artifact_id": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "schema_version": "1.0",
        }

    @staticmethod
    def write(path: Path, payload: Dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False)
            handle.write("\n")
        return path
