"""
Thin client helpers for the Master MCP Server.

Supports:
- OpenAI-compatible LLMs (tool-use payloads)
- Anthropic-compatible LLMs
- Local/Ollama-style direct HTTP calls
"""

from __future__ import annotations

import os
import requests
from typing import Any, Dict, List

MASTER_MCP_BASE = os.getenv("MASTER_MCP_BASE", "http://localhost:8001")
MASTER_MCP_KEY = os.getenv("MASTER_MCP_API_KEY", "")


def _headers():
    headers = {"Content-Type": "application/json"}
    if MASTER_MCP_KEY:
        headers["x-api-key"] = MASTER_MCP_KEY
    return headers


def list_tools() -> List[Dict[str, Any]]:
    resp = requests.get(f"{MASTER_MCP_BASE}/tools", headers=_headers(), timeout=15)
    resp.raise_for_status()
    return resp.json()


def invoke(tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(
        f"{MASTER_MCP_BASE}/invoke",
        headers=_headers(),
        json={"tool": tool, "args": args},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# --- OpenAI-compatible example -----------------------------------------------------
def build_openai_tools_payload(messages: List[Dict[str, str]], model: str = "gpt-4o") -> Dict[str, Any]:
    """
    Build a Chat Completions payload that includes MCP tool metadata for models
    that support tool calls. The caller can attach `list_tools()` results to the
    `tools` field.
    """
    tools = []
    for tool in list_tools():
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool.get("input_schema") or {"type": "object"},
                },
            }
        )
    return {"model": model, "messages": messages, "tools": tools}


# --- Anthropic-compatible example --------------------------------------------------
def build_anthropic_tools_payload(messages: List[Dict[str, str]], model: str = "claude-3-opus-20240229") -> Dict[str, Any]:
    tools = []
    for tool in list_tools():
        tools.append(
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool.get("input_schema") or {"type": "object"},
            }
        )
    return {"model": model, "messages": messages, "tools": tools}


# --- Local/Ollama example ----------------------------------------------------------
def run_local_example():
    print("Tools:", list_tools())
    print("calc:", invoke("ai.calc", {"expression": "2+2"}))
    print("materials:", invoke("materials.get_database_info", {}))


if __name__ == "__main__":
    run_local_example()
