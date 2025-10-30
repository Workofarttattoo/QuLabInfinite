from __future__ import annotations

import os
from typing import Dict, Any, Optional
import requests


class OQMDClient:
    """
    Minimal client for OQMD (Open Quantum Materials Database) REST API.
    Docs: https://oqmd.org/
    """

    def __init__(self, base_url: Optional[str] = None, timeout_s: int = 30) -> None:
        self.base_url = base_url or os.environ.get("OQMD_BASE_URL", "http://oqmd.org")
        self.timeout_s = timeout_s
        self.session = requests.Session()

    def search(self, query: str) -> Dict[str, Any]:
        """
        Query OQMD using the REST interface (SQL-like query string per OQMD docs).
        Example query: "select * from materials where formula=\"Si\""
        """
        url = f"{self.base_url}/materials/"
        params = {"q": query}
        resp = self.session.get(url, params=params, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()
