"""
Calibration utilities for the deterministic chemistry models.

The functions in this module compare model outputs against the hand-curated
reference datasets defined in ``chemistry_lab.reference_data`` and return error
metrics (mean absolute errors, maximum deviations, etc.).  They are designed to
run quickly as part of the automated test suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .reference_data import SPECTROSCOPY_REFERENCE, SYNTHESIS_REFERENCE
from .spectroscopy_predictor import SpectroscopyPredictor
from .synthesis_planner import SynthesisPlanner, Compound


@dataclass
class SpectroscopyCalibrationResult:
    molecule: str
    nmr_1h_mae: float
    ir_mae: float
    mass_spec_mae: float


@dataclass
class SynthesisCalibrationResult:
    target: str
    total_steps_error: int
    yield_abs_error: float


def _mean_absolute_difference(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(mean(abs(v) for v in values))


def _match_peak_set(predicted: List[float], expected: List[float]) -> float:
    if not expected:
        return 0.0
    differences = []
    for ref in expected:
        closest = min(predicted, key=lambda x: abs(x - ref))
        differences.append(abs(closest - ref))
    return _mean_absolute_difference(differences)


def calibrate_spectroscopy(predictor: SpectroscopyPredictor | None = None) -> Dict[str, object]:
    predictor = predictor or SpectroscopyPredictor()
    per_molecule: List[SpectroscopyCalibrationResult] = []

    for name, data in SPECTROSCOPY_REFERENCE.items():
        molecule = data["molecule"]

        # 1H NMR
        nmr_pred = predictor.predict_nmr_1h(molecule)
        nmr_positions = [peak.position for peak in nmr_pred.peaks]
        nmr_mae = _match_peak_set(nmr_positions, data.get("nmr_1h", []))

        # IR
        ir_pred = predictor.predict_ir(molecule)
        ir_positions = [peak.position for peak in ir_pred.peaks]
        ir_mae = _match_peak_set(ir_positions, data.get("ir", []))

        # Mass spec
        mass_pred = predictor.predict_mass_spec(molecule)
        mass_positions = [peak.position for peak in mass_pred.peaks]
        mass_mae = _match_peak_set(mass_positions, [mz for mz, _ in data.get("mass_spec", [])])

        per_molecule.append(
            SpectroscopyCalibrationResult(
                molecule=name,
                nmr_1h_mae=nmr_mae,
                ir_mae=ir_mae,
                mass_spec_mae=mass_mae,
            )
        )

    overall = {
        "nmr_1h_mae": _mean_absolute_difference(result.nmr_1h_mae for result in per_molecule),
        "ir_mae": _mean_absolute_difference(result.ir_mae for result in per_molecule),
        "mass_spec_mae": _mean_absolute_difference(result.mass_spec_mae for result in per_molecule),
    }

    return {
        "overall": overall,
        "per_molecule": per_molecule,
    }


def calibrate_synthesis(planner: SynthesisPlanner | None = None) -> Dict[str, object]:
    planner = planner or SynthesisPlanner()
    per_target: List[SynthesisCalibrationResult] = []

    for name, data in SYNTHESIS_REFERENCE.items():
        target_data = data["target"]
        target = Compound(**target_data)
        expected = data["expected"]

        route = planner.plan_route(target)
        if route is None:
            per_target.append(
                SynthesisCalibrationResult(
                    target=name,
                    total_steps_error=expected["total_steps"],
                    yield_abs_error=expected["overall_yield"],
                )
            )
            continue

        total_steps_error = route.total_steps - expected["total_steps"]
        yield_abs_error = abs(route.overall_yield - expected["overall_yield"])

        per_target.append(
            SynthesisCalibrationResult(
                target=name,
                total_steps_error=total_steps_error,
                yield_abs_error=yield_abs_error,
            )
        )

    overall = {
        "mean_steps_error": _mean_absolute_difference(result.total_steps_error for result in per_target),
        "mean_yield_abs_error": _mean_absolute_difference(result.yield_abs_error for result in per_target),
    }

    return {
        "overall": overall,
        "per_target": per_target,
    }
