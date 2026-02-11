#!/usr/bin/env python3
from __future__ import annotations
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

calibration - Part of Materials Lab
"""

"""
Simple experimental calibration bookkeeping for the materials laboratory.
"""


from dataclasses import dataclass, field
from typing import Dict, Tuple
from datetime import datetime, timezone


@dataclass
class CalibrationRecord:
    """Running statistics for a calibrated property."""

    material: str
    test_type: str
    property_name: str
    bias: float = 0.0
    samples: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def update(self, reference_value: float, measured_value: float) -> None:
        """Update the running bias using a simple exponential moving average."""
        measurement_bias = reference_value - measured_value
        if self.samples == 0:
            self.bias = measurement_bias
        else:
            alpha = min(0.5, 1.0 / (self.samples + 1))
            self.bias = (1 - alpha) * self.bias + alpha * measurement_bias
        self.samples += 1
        self.last_updated = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, float]:
        return {
            "bias": self.bias,
            "samples": self.samples,
            "last_updated": self.last_updated.isoformat(),
        }


class CalibrationManager:
    """Lightweight manager that keeps track of calibration corrections."""

    def __init__(self) -> None:
        self._records: Dict[Tuple[str, str, str], CalibrationRecord] = {}

    def register(self,
                 material: str,
                 test_type: str,
                 property_name: str,
                 reference_value: float,
                 measured_value: float) -> CalibrationRecord:
        """Record a new calibration measurement."""
        key = (material.lower(), test_type.lower(), property_name.lower())
        record = self._records.get(key)
        if record is None:
            record = CalibrationRecord(material, test_type, property_name)
            self._records[key] = record
        record.update(reference_value, measured_value)
        return record

    def apply(self,
              material: str,
              test_type: str,
              values: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Apply calibration bias to provided values in-place.

        Returns a dict describing the corrections that were applied.
        """
        corrections: Dict[str, Dict[str, float]] = {}
        if not values:
            return corrections

        lower_material = material.lower()
        lower_test = test_type.lower()

        for prop, value in list(values.items()):
            key = (lower_material, lower_test, prop.lower())
            record = self._records.get(key)
            if record and record.samples > 0:
                corrected_value = value + record.bias
                values[prop] = corrected_value
                corrections[prop] = {
                    "bias": record.bias,
                    "corrected_value": corrected_value,
                    "samples": record.samples,
                }

        return corrections

    def summary(self, material: str, test_type: str) -> Dict[str, Dict[str, float]]:
        """Return calibration summary for the requested material/test."""
        lower_material = material.lower()
        lower_test = test_type.lower()
        summary: Dict[str, Dict[str, float]] = {}
        for (mat, test, prop), record in self._records.items():
            if mat == lower_material and test == lower_test:
                summary[prop] = record.to_dict()
        return summary
