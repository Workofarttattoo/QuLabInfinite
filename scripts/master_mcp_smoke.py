"""
Smoke tests for Master MCP Server.

Run:
  MASTER_MCP_BASE=http://localhost:8001 MASTER_MCP_API_KEY=... python scripts/master_mcp_smoke.py
"""

from __future__ import annotations

import os
from typing import Dict

from scripts.master_mcp_clients import invoke, list_tools


def check_tool(name: str, args: Dict):
    print(f"[smoke] invoking {name} with {args}")
    resp = invoke(name, args)
    print(f"[smoke] result: {resp}")


def main():
    base = os.getenv("MASTER_MCP_BASE", "http://localhost:8001")
    print(f"[info] Master MCP base: {base}")
    print("[info] Listing tools...")
    tools = list_tools()
    print(f"[info] Found {len(tools)} tools")

    # Minimal representative calls
    check_tool("ai.calc", {"expression": "3*7+1"})
    check_tool("physics.get_element_properties", {"element": "Fe"})

    # Optional calls; skip if tool not present
    available = {t["name"] for t in tools}
    if "chemistry.validate_smiles" in available:
        check_tool("chemistry.validate_smiles", {"smiles": "CCO"})
    if "materials.get_database_info" in available:
        check_tool("materials.get_database_info", {})

    print("[info] Smoke tests completed.")


if __name__ == "__main__":
    main()
