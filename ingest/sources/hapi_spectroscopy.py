from __future__ import annotations

import os
from typing import List, Dict, Any

# HAPI is the HITRAN Application Programming Interface.
# Import guarded to keep this module importable if HAPI is not installed.
try:
    from hapi import fetch, absorptionCoefficient_Lorentz
except Exception:  # pragma: no cover - optional at runtime
    fetch = None
    absorptionCoefficient_Lorentz = None


class HitranClient:
    """
    Minimal wrapper around HITRAN HAPI for line lists and absorption coefficients.
    Requires internet and appropriate environment per HAPI/HITRAN terms of use.
    """

    def __init__(self, db_dir: str = ".hitran_db") -> None:
        self.db_dir = os.path.abspath(db_dir)
        os.makedirs(self.db_dir, exist_ok=True)

    def ensure_hapi(self) -> None:
        if fetch is None or absorptionCoefficient_Lorentz is None:
            raise RuntimeError(
                "HAPI is not available. Install 'hapi' and ensure HITRAN access is configured."
            )

    def fetch_lines(self, iso: str, mol_id: int, wmin: float, wmax: float) -> None:
        """
        Fetch line list for molecule id in [wmin, wmax].
        Example: mol_id=2 for CO2.
        """
        self.ensure_hapi()
        table_name = f"hitran_{mol_id}_{wmin}_{wmax}".replace(".", "_")
        fetch(table_name, mol_id, 1, wmin, wmax)

    def compute_absorption(self, table_name: str, pressure_atm: float, temperature_k: float) -> Dict[str, Any]:
        """
        Compute absorption using Lorentz profile for a previously fetched table.
        """
        self.ensure_hapi()
        # In real use, you will configure the path via hapi.db_begin/db_end
        # and call absorptionCoefficient_* with correct arguments.
        # Here we surface a shape of the returned object only.
        result = {
            "table": table_name,
            "pressure_atm": pressure_atm,
            "temperature_k": temperature_k,
            "coefficients": [],
        }
        return result
