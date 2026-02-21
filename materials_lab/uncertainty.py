#!/usr/bin/env python3
from __future__ import annotations
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

uncertainty - Part of Materials Lab
"""

"""
Helpers for estimating uncertainties on simulated measurements.
"""


from typing import Dict

from materials_database import MaterialProperties


_BASE_FRACTIONS: Dict[str, float] = {
    "tensile": 0.05,
    "compression": 0.05,
    "fatigue": 0.08,
    "impact": 0.06,
    "hardness": 0.04,
    "thermal": 0.03,
    "corrosion": 0.1,
    "environmental_extreme_cold": 0.07,
}


def _confidence_modifier(confidence: float) -> float:
    """Convert material confidence (0-1) to a multiplier."""
    confidence = max(0.0, min(confidence, 1.0))
    return 1.2 - 0.6 * confidence  # 0.6 for high confidence, 1.2 for low


def estimate_property_uncertainty(material: MaterialProperties,
                                  property_name: str,
                                  value: float,
                                  test_type: str) -> float:
    """
    Estimate one-sigma uncertainty for a measured property.
    """
    if value == 0 or not isinstance(value, (int, float)):
        return 0.0

    base_fraction = _BASE_FRACTIONS.get(test_type, 0.05)
    modifier = _confidence_modifier(getattr(material, "confidence", 0.8))

    # Some properties are inherently noisier.
    if "fatigue" in property_name or "cycles" in property_name:
        base_fraction *= 1.5
    elif "thermal_conductivity" in property_name:
        base_fraction *= 0.8
    elif "corrosion" in property_name:
        base_fraction *= 1.8

    return abs(value) * base_fraction * modifier
