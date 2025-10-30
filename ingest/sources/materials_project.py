from __future__ import annotations
from typing import Iterable, List
import os
import pandas as pd
import requests
from .base import DataSource


class MaterialsProjectClient(DataSource):
    """Client for the Materials Project REST API v2.

    Requires an API key in environment variable MP_API_KEY.
    """

    BASE_URL = os.getenv("MP_BASE_URL", "https://api.materialsproject.org/v2/materials/summary")

    def __init__(self, api_key: str | None = None, session: requests.Session | None = None) -> None:
        self.api_key = api_key or os.getenv("MP_API_KEY")
        self.session = session or requests.Session()
        if not self.api_key:
            # Operate in dry mode without calls
            self.session = None

    def fetch(self, identifiers: Iterable[str]) -> pd.DataFrame:
        rows: List[dict] = []
        if not self.session:
            return pd.DataFrame(columns=[
                "material_id", "formula", "property_name", "value", "units",
                "temperature_k", "source", "source_ref",
            ])

        headers = {"X-API-KEY": self.api_key}
        for ident in identifiers:
            try:
                # Query summaries by chemical formula
                resp = self.session.get(self.BASE_URL, params={"formula": ident}, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                results = data.get("data", []) if isinstance(data, dict) else []
                for item in results:
                    mpid = item.get("material_id") or item.get("material_id_deprecated")
                    formula = item.get("formula_pretty") or ident
                    energy = item.get("energy_per_atom")
                    if energy is not None:
                        rows.append({
                            "material_id": mpid or formula,
                            "formula": formula,
                            "property_name": "energy_per_atom",
                            "value": float(energy),
                            "units": "eV/atom",
                            "temperature_k": None,
                            "source": "Materials Project",
                            "source_ref": self.BASE_URL,
                        })
            except Exception:
                continue

        return pd.DataFrame(rows, columns=[
            "material_id", "formula", "property_name", "value", "units",
            "temperature_k", "source", "source_ref",
        ])
