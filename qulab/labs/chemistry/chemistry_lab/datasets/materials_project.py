"""
Specialized loader for fetching data from the Materials Project.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from mp_api.client import MPRester

def load_materials_project_data(
    chemsys: str,
    fields: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Fetch materials data from the Materials Project for a given chemical system.

    Args:
        chemsys: A chemical system string (e.g., "Fe-O", "Si-C").
        fields: A list of properties to fetch from the Materials Project.
                If None, a default set of fields will be used.

    Returns:
        A pandas DataFrame containing the fetched materials data.
    """
    if fields is None:
        fields = [
            "material_id", "formula_pretty", "symmetry", "band_gap",
            "formation_energy_per_atom", "e_above_hull"
        ]

    with MPRester() as mpr:
        docs = mpr.summary.search(chemsys=chemsys, fields=fields)

    data = [doc.dict() for doc in docs]
    return pd.DataFrame(data)
