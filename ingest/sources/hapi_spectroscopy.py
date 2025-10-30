from __future__ import annotations
from typing import Iterable, List
import pandas as pd

try:
    # HAPI is typically provided by HITRAN; install/availability may vary.
    from hapi import db_begin, fetch, partition, absorptionCoefficient_Voigt
    HAPI_AVAILABLE = True
except Exception:
    HAPI_AVAILABLE = False

from .base import DataSource


class HapiSpectroscopyClient(DataSource):
    """Fetch spectroscopic line data from HITRAN using HAPI if available.

    This client returns a canonical dataframe with a subset of columns
    (e.g., line intensity at reference temperature). Extend as needed.
    """

    def __init__(self, cache_dir: str = ".hapi_db") -> None:
        self.cache_dir = cache_dir
        if HAPI_AVAILABLE:
            db_begin(cache_dir)

    def fetch(self, identifiers: Iterable[str]) -> pd.DataFrame:
        if not HAPI_AVAILABLE:
            # Return empty dataframe if HAPI is not installed
            return pd.DataFrame(columns=[
                "material_id", "formula", "property_name", "value", "units",
                "temperature_k", "source", "source_ref",
            ])

        rows: List[dict] = []
        for formula in identifiers:
            try:
                # Example: fetch lines for a species symbol around 2000-2100 cm^-1
                table_name = f"{formula}_2000_2100"
                fetch(table_name, formula, 2000, 2100)
                # For simplicity, use partition data as a proxy property
                part = partition(formula, 296)  # at 296 K
                rows.append({
                    "material_id": formula,
                    "formula": formula,
                    "property_name": "spectral_partition_fn",
                    "value": float(part),
                    "units": "dimensionless",
                    "temperature_k": 296.0,
                    "source": "HITRAN HAPI",
                    "source_ref": f"HAPI:{table_name}",
                })
            except Exception:
                continue

        return pd.DataFrame(rows, columns=[
            "material_id", "formula", "property_name", "value", "units",
            "temperature_k", "source", "source_ref",
        ])
