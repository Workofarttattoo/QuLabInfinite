from __future__ import annotations

import os
from typing import Dict, Any, List, Optional
import requests


class MaterialsProjectClient:
    """
    Minimal client for Materials Project v2 API.
    Set MP_API_KEY in environment.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, timeout_s: int = 20) -> None:
        self.api_key = api_key or os.environ.get("MP_API_KEY")
        if not self.api_key:
            raise RuntimeError("MP_API_KEY not set. Obtain an API key from Materials Project and set env var.")
        self.base_url = base_url or "https://api.materialsproject.org"
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {self.api_key}"

    def search_structures(self, formula: str, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/v2/materials/summary/search"
        payload = {
            "formula": formula,
            "fields": fields or ["material_id", "formula_pretty", "structure", "energy_above_hull"]
        }
        resp = self.session.post(url, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()
