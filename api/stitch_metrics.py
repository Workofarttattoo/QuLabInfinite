"""
Stitch Data API integration for aios.is website metrics.
Uses STITCH_API_KEY from environment only (never exposed to frontend).
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

STITCH_BASE = "https://api.stitchdata.com"


def _get_token() -> str | None:
    return os.environ.get("STITCH_API_KEY") or os.environ.get("STITCH_ACCESS_TOKEN")


def _get(url: str, token: str) -> tuple[int, Any]:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:
        return -1, None


def fetch_stitch_metrics() -> dict[str, Any]:
    """
    Call Stitch Connect API to get sources and destinations for dashboard.
    Returns a safe summary; never exposes raw API key or sensitive config.
    """
    token = _get_token()
    if not token:
        return {"enabled": False, "reason": "STITCH_API_KEY not set", "sources": 0, "destinations": 0}

    out: dict[str, Any] = {"enabled": True, "sources": 0, "destinations": 0, "source_names": [], "destination_names": []}

    try:
        status_src, data_src = _get(f"{STITCH_BASE}/v4/sources", token)
        if status_src == 200 and data_src is not None:
            if isinstance(data_src, list):
                out["sources"] = len(data_src)
                out["source_names"] = [s.get("display_name") or s.get("name") or "Source" for s in data_src[:10] if isinstance(s, dict)]
            elif isinstance(data_src, dict) and "data" in data_src:
                arr = data_src["data"]
                out["sources"] = len(arr)
                out["source_names"] = [s.get("display_name") or s.get("name") or "Source" for s in arr[:10] if isinstance(s, dict)]
        elif status_src == 401:
            out["enabled"] = False
            out["reason"] = "Stitch API key invalid or expired"
            return out

        status_dst, data_dst = _get(f"{STITCH_BASE}/v4/destinations", token)
        if status_dst == 200 and data_dst is not None:
            if isinstance(data_dst, list):
                out["destinations"] = len(data_dst)
                out["destination_names"] = [d.get("display_name") or d.get("name") or "Destination" for d in data_dst[:10] if isinstance(d, dict)]
            elif isinstance(data_dst, dict) and "data" in data_dst:
                arr = data_dst["data"]
                out["destinations"] = len(arr)
                out["destination_names"] = [d.get("display_name") or d.get("name") or "Destination" for d in arr[:10] if isinstance(d, dict)]
    except Exception as e:
        out["enabled"] = False
        out["reason"] = "Stitch API unreachable"
        out["error"] = str(e)[:200]

    return out
