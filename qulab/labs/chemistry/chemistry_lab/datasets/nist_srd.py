"""Specialized loader for NIST Standard Reference Database 101."""

from __future__ import annotations
import io
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List
import pandas as pd

def _extract_table(lines: List[str]) -> List[Dict[str, str]]:
    """Extracts a multi-line, fixed-width table from a stream of lines."""
    if not lines:
        return []

    header_line = lines[0]
    # Rough column boundaries based on visual inspection of the data format
    boundaries = [0, 15, 30, 45, 60, 75]
    headers = [header_line[i:j].strip() for i, j in zip(boundaries, boundaries[1:])]

    data = []
    for line in lines[1:]:
        if line.strip() and "---" not in line:
            row = {headers[i]: line[boundaries[i]:boundaries[i+1]].strip() for i in range(len(headers))}
            data.append(row)
    return data

def load_nist_srd_101(path: Path) -> Dict[str, pd.DataFrame]:
    """
    Loader for the NIST SRD 101 thermochemical database.

    Extracts tables from the zipped archive and returns a dictionary of
    DataFrames, where each key is a species name.
    """
    tables: Dict[str, pd.DataFrame] = {}
    if not path.exists() or not path.is_file():
        return tables

    with zipfile.ZipFile(path, "r") as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".txt"):
                continue

            with archive.open(name, "r") as handle:
                lines = [line.decode("latin-1").rstrip() for line in handle]
                
                # Find tables within the file
                in_table = False
                table_lines: List[str] = []
                species_name = "unknown"

                for line in lines:
                    if "Ideal Gas Thermochemical Data" in line:
                        # Reset for a new species table
                        if table_lines:
                            table_data = _extract_table(table_lines)
                            if table_data:
                                tables[species_name] = pd.DataFrame(table_data)
                        table_lines = []
                        in_table = False

                        # Extract species name
                        parts = line.split()
                        if len(parts) > 1:
                           species_name = parts[0]

                    if line.startswith("T(K)"):
                        in_table = True
                        table_lines = [line]
                    elif in_table and line.strip():
                        table_lines.append(line)
                    elif not line.strip():
                        if in_table and table_lines:
                            table_data = _extract_table(table_lines)
                            if table_data:
                                tables[species_name] = pd.DataFrame(table_data)
                        in_table = False
                        table_lines = []

                # Capture the last table in the file
                if table_lines:
                    table_data = _extract_table(table_lines)
                    if table_data:
                        tables[species_name] = pd.DataFrame(table_data)

    return tables
