"""
Specialized loader for fetching data from the Crystallography Open Database (COD).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import requests

def load_cod_data(
    formula: str,
    max_results: int = 100
) -> pd.DataFrame:
    """
    Fetch crystal structure data from the Crystallography Open Database (COD)
    for a given chemical formula.

    Args:
        formula: A chemical formula (e.g., "SiO2", "Fe2O3").
        max_results: The maximum number of results to return.

    Returns:
        A pandas DataFrame containing the fetched crystal structure data.
    """
    base_url = "http://www.crystallography.net/cod/result"
    params = {
        "text": formula,
        "format": "json",
        "page": 1,
        "count": max_results
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        
        if "entries" in data:
            return pd.DataFrame(data["entries"])
        else:
            return pd.DataFrame()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from COD: {e}")
        return pd.DataFrame()
    except ValueError:  # Handles JSON decoding errors
        print("Error decoding JSON from COD response.")
        return pd.DataFrame()
