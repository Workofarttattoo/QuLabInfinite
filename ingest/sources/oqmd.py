from __future__ import annotations
from typing import Iterable, List
import pandas as pd
import requests
from .base import DataSource


class OQMDClient(DataSource):
    """Client for the Open Quantum Materials Database (OQMD) API.

    Reference: http://oqmd.org/oqmdapi (availability may vary)
    """

    BASE_URL = "http://oqmd.org/oqmdapi/formationenergy"

    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()

    def fetch(self, identifiers: Iterable[str]) -> pd.DataFrame:
        rows: List[dict] = []
        for ident in identifiers:
            try:
                # Simple query by chemical formula
                resp = self.session.get(self.BASE_URL, params={"formula": ident}, timeout=30)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                results = data.get("data", []) if isinstance(data, dict) else []
                for item in results:
                    formula = item.get("composition") or ident
                    fe = item.get("delta_e") or item.get("formation_energy")
                    if fe is None:
                        continue
                    rows.append({
                        "material_id": formula,
                        "formula": formula,
                        "property_name": "formation_energy",
                        "value": float(fe),
                        "units": "eV/atom",
                        "temperature_k": None,
                        "source": "OQMD",
                        "source_ref": self.BASE_URL,
                    })
            except Exception:
                continue

        return pd.DataFrame(rows, columns=[
            "material_id", "formula", "property_name", "value", "units",
            "temperature_k", "source", "source_ref",
        ])
