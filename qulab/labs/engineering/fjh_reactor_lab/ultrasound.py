"""
Ultrasound hypothesis module for FJH reactor.

Effect on Au single-atom retention is currently UNVALIDATED.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import UltrasoundStage


@dataclass
class UltrasoundConfig:
    """Optional ultrasound research hypothesis parameters."""

    enabled: bool = False
    stage: UltrasoundStage = UltrasoundStage.DISABLED
    frequency_Hz: float | None = None
    amplitude: float | None = None  # relative 0-1
    duration_s: float | None = None
    coupling_efficiency: float | None = None  # 0-1, UNKNOWN if not calibrated

    validation_status: str = (
        "Effect on Au single-atom retention currently unvalidated."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "stage": self.stage.value,
            "frequency_Hz": self.frequency_Hz,
            "amplitude": self.amplitude,
            "duration_s": self.duration_s,
            "coupling_efficiency": self.coupling_efficiency,
            "validation_status": self.validation_status,
            "provenance": "HYPOTHESIS — no assumed benefit",
        }


def compare_ultrasound_hypothesis(
    simulate_fn,
    base_config,
    ultrasound: UltrasoundConfig,
) -> dict[str, Any]:
    """
    Compare ultrasound ON vs OFF without hard-coding benefit.
    """
    result_off = simulate_fn(base_config, ultrasound_enabled=False)
    result_on = simulate_fn(base_config, ultrasound_enabled=True, ultrasound=ultrasound)

    return {
        "ultrasound_off": result_off,
        "ultrasound_on": result_on,
        "validation_status": ultrasound.validation_status,
        "note": (
            "Comparison reports simulation outputs with ultrasound ON vs OFF. "
            "No validated coupling/material model exists; scores are unchanged "
            "by ultrasound until experimental calibration."
        ),
        "assumed_benefit": False,
    }
