"""
Specialized loader for fetching nuclear data from the BNL National Nuclear Data Center (NNDC).
"""

from __future__ import annotations

from typing import Optional
import io

import pandas as pd
import requests

def load_nndc_wallet_card(nuclide: str) -> Optional[pd.DataFrame]:
    """
    Fetch Nuclear Wallet Card data for a given nuclide from the NNDC.

    Args:
        nuclide: The nuclide to query (e.g., "60co", "235u").

    Returns:
        A pandas DataFrame containing the Nuclear Wallet Card data, or None if
        the nuclide is not found or an error occurs.
    """
    base_url = "https://www.nndc.bnl.gov/wallet/wallet.jsp"
    params = {"nuc": nuclide}

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        # The response is a fixed-width text file, so we need to parse it.
        # This is a simplified parser based on the NWC format.
        col_specs = [
            (0, 5), (6, 8), (9, 11), (12, 14), (15, 24), (25, 39), (40, 53),
            (54, 66), (67, 78), (79, 94), (95, 105)
        ]
        col_names = [
            "NZ", "N", "Z", "A", "Isotope", "Mass_Excess_MeV", "Decay_Mode",
            "Half_life_sec", "Spin_Parity", "Abundance", "ENSDF_Date"
        ]

        # Use a string IO buffer to treat the text response as a file
        text_buffer = io.StringIO(response.text)
        
        # Skip the header lines to get to the data
        lines = text_buffer.readlines()
        data_lines = [line for line in lines if line.strip() and not line.startswith(('=', ' '))]

        if not data_lines:
            return None

        df = pd.read_fwf(io.StringIO("".join(data_lines)), colspecs=col_specs, header=None)
        df.columns = col_names

        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from NNDC: {e}")
        return None
    except Exception as e:
        print(f"Error parsing NNDC data: {e}")
        return None
