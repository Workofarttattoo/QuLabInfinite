import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Protocol


class Tool(Protocol):
    """Contract for runtime tools that can execute named simulations."""

    @property
    def name(self) -> str:
        ...

    def describe(self) -> Mapping[str, Any]:
        ...

    def run(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class RuntimeArtifact:
    """Canonical artifact format emitted for every runtime execution."""

    tool: str
    payload: Dict[str, Any]
    result: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "payload": self.payload,
            "result": self.result,
        }

    def to_json(self) -> str:
        """Deterministic JSON for reproducible artifacts."""
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))


class RuntimeRegistry:
    """Canonical runtime registry for discoverable named tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def list_tools(self) -> Dict[str, Dict[str, Any]]:
        return {name: dict(tool.describe()) for name, tool in self._tools.items()}

    def get_tool(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._tools))
            raise ValueError(f"Unknown tool: {name}. Available tools: [{available}]") from exc

    def run(self, name: str, payload: Mapping[str, Any]) -> RuntimeArtifact:
        result = self.get_tool(name).run(payload)
        return RuntimeArtifact(tool=name, payload=dict(payload), result=dict(result))


_runtime_singleton: RuntimeRegistry | None = None


def get_runtime_registry() -> RuntimeRegistry:
    global _runtime_singleton
    if _runtime_singleton is None:
        _runtime_singleton = RuntimeRegistry()
    return _runtime_singleton
