"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

hitran - Part of Chemistry Lab
"""

"""
Specialized loader for fetching spectroscopic data from the HITRAN database.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from hapi import fetch

def load_hitran_data(
    table_name: str,
    molecule_id: int,
    isotopologue_id: int,
    wavenumber_range: Tuple[float, float],
) -> pd.DataFrame:
    """
    Fetch spectroscopic data from the HITRAN database for a specific molecule
    and wavenumber range.

    Args:
        table_name: The name of the HITRAN table to query (e.g., 'H2O').
        molecule_id: The HITRAN molecule ID (e.g., 1 for H2O).
        isotopologue_id: The HITRAN isotopologue ID (e.g., 1 for H2^16O).
        wavenumber_range: A tuple specifying the min and max wavenumber (cm^-1).

    Returns:
        A pandas DataFrame containing the fetched spectroscopic data.
    """
    min_wavenumber, max_wavenumber = wavenumber_range
    
    # The hitran-api fetch function requires a list of parameters to retrieve.
    # We will fetch a default set of common parameters.
    parameters = [
        'nu', 'sw', 'a', 'gamma_air', 'gamma_self', 'n_air', 'delta_air',
        'elower', 'gpp', 'gp'
    ]

    fetch(
        table_name,
        molecule_id,
        isotopologue_id,
        min_wavenumber,
        max_wavenumber,
        ParameterGroups=parameters
    )

    # The fetch function saves the data to a file with the same name as the table.
    # We can then load this file into a pandas DataFrame.
    # The file is typically a .data file, which is space-delimited.
    try:
        df = pd.read_csv(f"{table_name}.data", delim_whitespace=True, header=None)
        # The column names are not included in the file, so we have to add them manually.
        # The order of columns corresponds to the order of parameters requested.
        df.columns = parameters
        return df
    except FileNotFoundError:
        return pd.DataFrame()
