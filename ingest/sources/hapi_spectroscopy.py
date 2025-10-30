from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

try:
    # HAPI is the official HITRAN Python interface
    from hapi import db_begin, fetch, molecularName, ISO_ID
except Exception as e:  # pragma: no cover - optional dependency not always present
    db_begin = None  # type: ignore
    fetch = None  # type: ignore
    molecularName = None  # type: ignore
    ISO_ID = None  # type: ignore


class HitranClient:
    """
    Thin wrapper around HITRAN's HAPI to fetch spectroscopic lines.

    Usage pattern:
      client = HitranClient()
      client.set_cache_directory("/path/to/cache")  # optional, defaults to temp dir
      lines = client.fetch_lines(
          molecule="H2O",  # formula or HITRAN molecule name
          isotopologue_id=1,
          wavenumber_min=500.0,
          wavenumber_max=4000.0,
          intensity_min=1e-27,
      )
    """

    def __init__(self) -> None:
        self._db_path: Optional[str] = None

    def set_cache_directory(self, path: str) -> None:
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        self._db_path = path

    def _ensure_db(self) -> str:
        if self._db_path is None:
            self._db_path = tempfile.mkdtemp(prefix="hitran_hapi_")
        # Initialize HAPI local database
        if db_begin is None:
            raise RuntimeError(
                "HAPI (HITRAN Python Interface) is not installed. Please install 'hapi'."
            )
        db_begin(self._db_path)
        return self._db_path

    @staticmethod
    def _normalize_molecule(molecule: str) -> str:
        # HAPI accepts standard names or formulas; keep passthrough but strip
        return molecule.strip()

    def fetch_lines(
        self,
        molecule: str,
        isotopologue_id: int = 1,
        wavenumber_min: float = 0.0,
        wavenumber_max: float = 99999.0,
        intensity_min: float = 0.0,
        table_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch line list for a molecule via HAPI.

        - molecule: chemical formula or HITRAN molecule name (e.g., "H2O")
        - isotopologue_id: HITRAN ISO_ID (1 = most abundant)
        - wavenumber_min/max: spectral window in cm^-1
        - intensity_min: minimum line intensity at 296 K (cm^-1/(molecule*cm^-2))
        - table_name: optional HAPI table label to reuse cache between calls

        Returns a list of dicts for portability without hard dep on pandas.
        """
        self._ensure_db()

        mol = self._normalize_molecule(molecule)
        label = table_name or self._make_label(mol, isotopologue_id, wavenumber_min, wavenumber_max, intensity_min)

        if fetch is None:
            raise RuntimeError(
                "HAPI (HITRAN Python Interface) is not installed. Please install 'hapi'."
            )

        # Perform fetch into local HAPI SQLite DB; HAPI stores a table named `label`
        fetch(
            label,
            mol,
            isotopologue_id,
            wavenumber_min,
            wavenumber_max,
            ParameterDefinitions={"Intensity": ("S", "Intensity")},  # ensure intensity field
            Destination=None,
            # Intensity threshold filtering is applied post-fetch for reliability
        )

        # Read back using HAPI's built-in accessor via sqlite (avoid pandas hard-dep)
        # HAPI exposes the table via global variables in its API; we will query through sqlite directly.
        import sqlite3  # local import to avoid global dependency if unused

        db_file = os.path.join(self._db_path or "", "data.sqlite")
        if not os.path.isfile(db_file):
            # Newer HAPI may store per-table dbs; fallback to searching directory
            candidate = self._locate_sqlite(self._db_path or "")
            if candidate is None:
                raise RuntimeError("Could not locate HAPI SQLite database after fetch.")
            db_file = candidate

        rows: List[Dict[str, Any]] = []
        with sqlite3.connect(db_file) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(f"SELECT * FROM '{label}'")
            for r in cur.fetchall():
                rec = {k: r[k] for k in r.keys()}
                if intensity_min > 0.0:
                    # HAPI intensities are typically in column 'S'
                    s_val = rec.get("S")
                    if s_val is None or float(s_val) < float(intensity_min):
                        continue
                rows.append(rec)

        return rows

    @staticmethod
    def _make_label(
        molecule: str,
        isotopologue_id: int,
        wmin: float,
        wmax: float,
        smin: float,
    ) -> str:
        # Compact but deterministic label for cache table name
        return (
            f"{molecule}_iso{isotopologue_id}_" \
            f"{int(wmin)}-{int(wmax)}cm1_smin{str(smin).replace('.', 'p')}"
        )

    @staticmethod
    def _locate_sqlite(path: str) -> Optional[str]:
        # Try common filenames HAPI uses
        candidates = [
            os.path.join(path, "data.sqlite"),
            os.path.join(path, "HITRAN.sqlite"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        # Fallback: scan for any .sqlite in path
        for fname in os.listdir(path or "."):
            if fname.endswith(".sqlite"):
                return os.path.join(path, fname)
        return None


__all__ = ["HitranClient"]
