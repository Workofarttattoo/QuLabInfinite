#!/usr/bin/env python3
from __future__ import annotations
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

safety - Part of Materials Lab

Safety (MSDS) metadata access for materials.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SafetyData:
    """Minimal MSDS-style metadata."""

    hazards: List[str]
    signal_word: str
    pictograms: List[str]
    ppe: List[str]
    storage: str
    first_aid: Dict[str, str]
    notes: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "hazards": self.hazards,
            "signal_word": self.signal_word,
            "pictograms": self.pictograms,
            "ppe": self.ppe,
            "storage": self.storage,
            "first_aid": self.first_aid,
            "notes": self.notes,
        }


class SafetyManager:
    """Simple dictionary-backed safety database."""

    def __init__(self, payload: Dict[str, Dict[str, object]]):
        self._entries: Dict[str, SafetyData] = {}
        for name, record in payload.items():
            self._entries[name.lower()] = SafetyData(
                hazards=record.get("hazards", []),
                signal_word=record.get("signal_word", "None"),
                pictograms=record.get("pictograms", []),
                ppe=record.get("ppe", []),
                storage=record.get("storage", "Store in a cool, dry place."),
                first_aid=record.get("first_aid", {}),
                notes=record.get("notes", ""),
            )

    def get(self, material_name: str) -> Optional[SafetyData]:
        return self._entries.get(material_name.lower())

    def to_dict(self, material_name: str) -> Optional[Dict[str, object]]:
        entry = self.get(material_name)
        return entry.to_dict() if entry else None
