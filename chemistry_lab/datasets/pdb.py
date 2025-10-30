"""
Specialized loader for fetching data from the Protein Data Bank (PDB).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from pypdb import Query, get_pdb_file

def search_pdb(query_text: str, max_results: int = 100) -> pd.DataFrame:
    """
    Search the Protein Data Bank for structures matching a given query.

    Args:
        query_text: The search query (e.g., "hemoglobin", "cancer").
        max_results: The maximum number of results to return.

    Returns:
        A pandas DataFrame containing metadata for the matching PDB entries.
    """
    q = Query(query_text)
    results = q.search(limit=max_results)
    
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame()

def download_pdb_structure(pdb_id: str, directory: str = ".") -> Optional[str]:
    """
    Download a PDB file for a given PDB ID.

    Args:
        pdb_id: The 4-character PDB ID (e.g., "1A1W").
        directory: The directory where the PDB file should be saved.

    Returns:
        The path to the downloaded PDB file, or None if the download failed.
    """
    try:
        pdb_file = get_pdb_file(pdb_id, filetype='pdb', dest_dir=directory)
        return pdb_file
    except Exception as e:
        print(f"Error downloading PDB file {pdb_id}: {e}")
        return None
