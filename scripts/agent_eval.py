"""
Lightweight agent/tool eval harness.

This script exercises a small battery of tasks against the Master MCP server
and reports pass/fail with basic assertions on the returned payloads.

Usage:
  MCP_API_URL=http://localhost:8000 MCP_API_KEY=... python scripts/agent_eval.py
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Callable

from scripts.master_mcp_client import call_tool


Test = Dict[str, Any]


def assert_key(result: Dict[str, Any], key: str) -> bool:
    return isinstance(result.get("result", {}), dict) and key in result["result"]


def tests() -> Dict[str, Callable[[], bool]]:
    return {
        "ai.calc": lambda: call_tool("ai.calc", {"expr": "10/2"})["result"] == 5,
        "chemistry.validate_smiles": lambda: call_tool("chemistry.validate_smiles", {"smiles": "CCO"})[
            "result"
        ].get("valid", True),
        "materials.get_database_info": lambda: assert_key(call_tool("materials.get_database_info", {}), "total_materials"),
    }


def run() -> int:
    failures = []
    for name, fn in tests().items():
        try:
            ok = fn()
            if not ok:
                failures.append(f"{name}: assertion failed")
        except Exception as exc:
            failures.append(f"{name}: {exc}")
    if failures:
        print("[eval] failures:")
        for f in failures:
            print(f"- {f}")
        return 1
    print("[eval] all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(run())
