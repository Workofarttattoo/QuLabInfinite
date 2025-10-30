from __future__ import annotations
from typing import Iterable, List
import pandas as pd
import requests
from .base import DataSource


class NistThermoClient(DataSource):
    """Fetch thermodynamic data from NIST Chemistry WebBook (HTML scrape endpoint).

    Note: The WebBook does not offer an official JSON API for all data. This client
    implements a minimal fetch that retrieves constant-pressure heat capacity (Cp)
    values when available by parsing the HTML page. For broader coverage, extend
    the parser or integrate curated endpoints if available.
    """

    BASE_URL = "https://webbook.nist.gov/cgi/cbook.cgi"

    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()

    def fetch(self, identifiers: Iterable[str]) -> pd.DataFrame:
        rows: List[dict] = []
        for ident in identifiers:
            formula = ident
            try:
                # Query by formula (Name=on requests the species page)
                resp = self.session.get(self.BASE_URL, params={"Name": formula, "Units": "SI"}, timeout=30)
                resp.raise_for_status()
                html = resp.text

                # Minimal heuristic extraction: look for Cp at 298 K patterns
                # This is intentionally conservative; refine as needed.
                cp_value = self._extract_cp_298k(html)
                if cp_value is not None:
                    rows.append({
                        "material_id": formula,
                        "formula": formula,
                        "property_name": "Cp",
                        "value": cp_value,
                        "units": "J/mol-K",
                        "temperature_k": 298.15,
                        "source": "NIST Chemistry WebBook",
                        "source_ref": resp.url,
                    })
            except Exception:
                # Skip identifier on any failure (network, parse, etc.)
                continue

        return pd.DataFrame(rows, columns=[
            "material_id", "formula", "property_name", "value", "units",
            "temperature_k", "source", "source_ref",
        ])

    @staticmethod
    def _extract_cp_298k(html: str) -> float | None:
        # Extremely simple pattern search; replace with robust parser (BeautifulSoup) if desired.
        # Examples might contain text like: "Cp,gas = 33.58 J/mol*K (at 298 K)"
        import re
        m = re.search(r"Cp[^\n]*?=\s*([0-9]+\.?[0-9]*)\s*J/mol\*K[^\n]*?\(at\s*298\s*K\)", html, flags=re.IGNORECASE)
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None
