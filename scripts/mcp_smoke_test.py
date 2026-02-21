"""
Smoke tests for the Master MCP Server.

Runs a minimal set of calls to ensure the server is reachable and core tools work.

Usage:
  MCP_API_URL=http://localhost:8000 MCP_API_KEY=... python scripts/mcp_smoke_test.py
"""

from __future__ import annotations

import sys
from typing import Any, Dict

from master_mcp_server import API_KEY_ENV
from scripts.master_mcp_client import call_tool, list_tools


def run() -> int:
    print("[smoke] listing tools...")
    tools = list_tools()
    print(f"[smoke] found {len(tools)} tools")

    tests: Dict[str, Dict[str, Any]] = {
        "ai.calc": {"expr": "2+2*2"},
        "chemistry.validate_smiles": {"smiles": "CCO"},
        "physics.get_element_properties": {"element_symbol": "Fe"},
    }

    failures = 0
    for tool, args in tests.items():
        try:
            print(f"[smoke] calling {tool}({args})")
            result = call_tool(tool, args)
            print(f"[smoke] ok -> {result}")
        except Exception as exc:
            failures += 1
            print(f"[smoke] FAILED {tool}: {exc}")

    if failures:
        print(f"[smoke] completed with {failures} failure(s)")
        return 1

    print("[smoke] all core calls succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(run())
