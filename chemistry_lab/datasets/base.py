"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

base - Part of Chemistry Lab
"""

"""Dataset descriptor definitions for chemistry machine learning resources."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence
from pandas import DataFrame, read_csv
import zipfile


def _flatten_json(data: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten a nested JSON object.
    """
    items = {}
    for k, v in data.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_json(v, new_key, sep=sep))
        elif isinstance(v, list):
            for i, elem in enumerate(v):
                if isinstance(elem, dict):
                    items.update(_flatten_json(elem, f"{new_key}_{i}", sep=sep))
                else:
                    items[f"{new_key}_{i}"] = elem
        else:
            items[new_key] = v
    return items

@dataclass
class DatasetDescriptor:
    """Metadata and loader hooks for a chemistry dataset."""

    name: str
    category: str
    description: str
    url: str
    citation: Optional[str] = None
    local_hint: Optional[str] = None
    file_extensions: Iterable[str] = field(default_factory=lambda: [".csv"])
    requires_conversion: bool = False
    notes: Optional[str] = None
    env_var: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "url": self.url,
            "citation": self.citation,
            "local_hint": self.local_hint,
            "file_extensions": list(self.file_extensions),
            "requires_conversion": self.requires_conversion,
            "notes": self.notes,
            "env_var": self.env_var,
        }

    # --- Loader stubs -------------------------------------------------

    def resolve_base_dir(self, base_dir: Optional[Path] = None) -> Path:
        """
        Resolve dataset directory using explicit base_dir or environment hint.

        Returns the resolved path (may not exist).
        """
        if base_dir is not None:
            return Path(base_dir)
        if self.env_var:
            value = Path(os.environ[self.env_var]) if self.env_var in os.environ else None
            if value:
                return value
        if self.local_hint:
            return Path(self.local_hint)
        return Path(".")

    def discover_files(self, base_dir: Optional[Path] = None) -> List[Path]:
        """
        Discover files in the dataset directory matching allowed extensions.

        Returns a list (possibly empty) of paths.
        """
        root = self.resolve_base_dir(base_dir)
        if not root.exists():
            return []

        extensions = tuple(ext.lower() for ext in self.file_extensions)
        candidates: List[Path] = []

        if root.is_file():
            if root.suffix.lower() in extensions:
                candidates.append(root)
            return candidates

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() in extensions:
                candidates.append(path)
        return candidates

    def load_csv_rows(self, path: Path) -> Iterator[Dict[str, Any]]:
        """
        Stream rows from a CSV file for quick inspection.

        This loader is intentionally lightweight; callers can override it for
        dataset-specific parsing logic.
        """
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path, "r") as archive:
                for name in archive.namelist():
                    if name.lower().endswith(".csv"):
                        with archive.open(name) as handle:
                            text = (line.decode("utf-8") for line in handle)
                            reader = csv.DictReader(text)
                            for row in reader:
                                yield row
                        return
            return

        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield row

    def load_metadata(self, path: Path) -> Dict[str, Any]:
        """
        Load metadata stored alongside the dataset.

        Expected file names:
          - <dataset>.json (preferred)
          - meta.json
        """
        if path.suffix.lower() in {".json"}:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        candidate = path if path.is_file() else path / "meta.json"
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        raise FileNotFoundError(f"No metadata JSON found for dataset {self.name}")

    def load_jsonl(self, path: Path) -> Iterator[Dict[str, Any]]:
        """
        Stream rows from a JSON Lines file.
        """
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    def load_dataframe(self, path: Path) -> DataFrame:
        """
        Load a CSV or JSONL file into a Pandas DataFrame.

        Handles both plain and zip-archived CSV files.
        """
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path, "r") as archive:
                for name in archive.namelist():
                    if name.lower().endswith(".csv"):
                        with archive.open(name) as handle:
                            return read_csv(handle)
        
        if path.suffix.lower() == ".jsonl":
            return DataFrame(self.load_jsonl(path))

        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                return DataFrame([_flatten_json(row) for row in data])
            else:
                return DataFrame([_flatten_json(data)])

        return read_csv(path)

    def load_sample_rows(
        self,
        base_dir: Optional[Path] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Load a small sample of rows from the dataset for inspection.

        Returns an empty list if no files are present.
        """
        candidates = self.discover_files(base_dir)
        for candidate in candidates:
            try:
                sample = []
                for row in self.load_csv_rows(candidate):
                    sample.append(row)
                    if len(sample) >= limit:
                        break
                if sample:
                    return sample
            except FileNotFoundError:
                continue

        return []
