from __future__ import annotations

from typing import Dict, Any


def validate_record(record: Dict[str, Any]) -> None:
    """
    Minimal validation: ensure required keys and types exist.
    Extend this with domain-specific checks and unit validations.
    """
    if not isinstance(record, dict):
        raise ValueError("record must be a dict")
    for key in ("source", "key", "data"):
        if key not in record:
            raise ValueError(f"record missing required key: {key}")
    if not isinstance(record["data"], dict):
        raise ValueError("record.data must be a dict")
