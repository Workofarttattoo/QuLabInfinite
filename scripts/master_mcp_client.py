"""
Client helpers for the Master MCP Server.

Supports:
- Direct HTTP calls
- OpenAI-compatible chat models
- Anthropic-compatible chat models
- Local/Ollama/Llama.cpp via HTTP

These helpers keep payloads small and assume the MCP server exposes /tools and /call.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


API_URL = os.getenv("MCP_API_URL", "http://localhost:8000")
API_KEY = os.getenv("MCP_API_KEY")


def _headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


def list_tools() -> List[Dict[str, Any]]:
    resp = requests.get(f"{API_URL}/tools", headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def call_tool(tool: str, args: Dict[str, Any], lite: bool = False) -> Dict[str, Any]:
    payload = {"tool": tool, "args": args, "lite": lite}
    resp = requests.post(f"{API_URL}/call", headers=_headers(), json=payload, timeout=120)
    if not resp.ok:
        raise RuntimeError(f"Tool call failed ({resp.status_code}): {resp.text}")
    return resp.json()


# --- OpenAI-compatible helper ---
def openai_tool_prompt(tool: str, args: Dict[str, Any]) -> str:
    return f"Call MCP tool `{tool}` with args {json.dumps(args)} and return the JSON result."


# --- Anthropic-compatible helper ---
def anthropic_tool_prompt(tool: str, args: Dict[str, Any]) -> str:
    return f"Use the MCP tool `{tool}` with args {json.dumps(args)}; return only JSON."


# --- Local model helper (generic HTTP completion) ---
def local_tool_prompt(tool: str, args: Dict[str, Any]) -> str:
    return f"[MCP] tool={tool} args={json.dumps(args)} -> reply with JSON only."


def demo_calls() -> None:
    print("[demo] available tools:")
    for t in list_tools():
        print(f"- {t['name']} (cost={t['cost_tokens']}, lite={t['lite_allowed']})")

    print("[demo] ai.calc 2+2")
    print(call_tool("ai.calc", {"expr": "2+2"}))

    print("[demo] chemistry.validate_smiles 'CCO'")
    print(call_tool("chemistry.validate_smiles", {"smiles": "CCO"}))


if __name__ == "__main__":
    demo_calls()
