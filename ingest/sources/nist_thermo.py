from __future__ import annotations

import os
from typing import Dict, Any, Optional
import requests


class NISTThermoClient:
    """
    Minimal NIST Chemistry WebBook client.
    Note: NIST does not provide an open, stable public API for full thermodynamic tables.
    This client uses a placeholder endpoint and is designed to be swapped with a
    project-approved data provider or a sanctioned scraping service.
    """

    def __init__(self, base_url: Optional[str] = None, timeout_s: int = 20) -> None:
        self.base_url = base_url or os.environ.get("NIST_WEBBOOK_BASE_URL", "https://webbook.nist.gov")
        self.timeout_s = timeout_s
        self.session = requests.Session()

    def get_thermo_by_inchi(self, inchi: str) -> Dict[str, Any]:
        """
        Fetch thermodynamic metadata by InChI identifier.
        Returns a normalized dict with units attached to values when possible.
        """
        # Placeholder endpoint; you will likely replace this with a sanctioned data service
        # in your environment, or a pre-cached mirror. The method is structured to make that swap trivial.
        url = f"{self.base_url}/cgi/cbook.cgi"
        params = {"InChI": inchi, "Units": "SI"}
        resp = self.session.get(url, params=params, timeout=self.timeout_s)
        resp.raise_for_status()
        # The WebBook returns HTML. In practice, parse and normalize here.
        # For now, return the raw text payload as a stubbed structure.
        return {
            "source": "NIST Chemistry WebBook",
            "query": {"inchi": inchi},
            "raw_payload": resp.text,
        }

    def get_thermo_by_formula(self, formula: str) -> Dict[str, Any]:
        url = f"{self.base_url}/cgi/cbook.cgi"
        params = {"Formula": formula, "Units": "SI"}
        resp = self.session.get(url, params=params, timeout=self.timeout_s)
        resp.raise_for_status()
        return {
            "source": "NIST Chemistry WebBook",
            "query": {"formula": formula},
            "raw_payload": resp.text,
        }
