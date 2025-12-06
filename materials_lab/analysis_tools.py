#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

analysis_tools - Part of Materials Lab
"""

"""
Materials analysis helpers for QuLab Infinite.

Provides:
    * Pareto ranking and outlier detection
    * Material comparison tables and normalized scores
    * Batch experiment runner built on MaterialsLab tests
    * Optional visualization helpers (matplotlib)
    * CLI-style summary utilities for integrations

These helpers do not modify the underlying database; they provide structured
results suitable for ECH0 workflows, command-line tools, or web endpoints.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

try:  # pragma: no cover - support script execution
    from materials_database import MaterialProperties  # type: ignore
except ImportError:  # package-relative
    from .materials_database import MaterialProperties  # type: ignore

if TYPE_CHECKING:  # only for static type checking
    from materials_lab import MaterialsLab  # type: ignore


# ---------------------------------------------------------------------------
# Data classes


@dataclass
class ComparisonRecord:
    """Structured result for materials comparison."""

    property_name: str
    unit: str
    values: Dict[str, float]
    best_material: Optional[str] = None
    normalized_scores: Optional[Dict[str, float]] = None


@dataclass
class BatchExperimentResult:
    """Container for batch experiment output."""

    experiment_id: str
    material: str
    test_type: str
    success: bool
    payload: Dict[str, Any]
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers


def _ensure_lab(lab: Optional["MaterialsLab"]) -> "MaterialsLab":
    if lab is not None:
        return lab
    from materials_lab import MaterialsLab  # type: ignore

    return MaterialsLab()


# ---------------------------------------------------------------------------
# Ranking / filtering helpers


def pareto_front(materials: Iterable[MaterialProperties],
                 objectives: Dict[str, Literal["max", "min"]]) -> List[MaterialProperties]:
    """Compute the Pareto front for the provided objectives."""

    mats = list(materials)
    pareto: List[MaterialProperties] = []

    for candidate in mats:
        dominated = False
        c_values = {prop: getattr(candidate, prop, None) for prop in objectives}

        if any(v is None for v in c_values.values()):
            continue

        for other in mats:
            if other is candidate:
                continue
            o_values = {prop: getattr(other, prop, None) for prop in objectives}
            if any(v is None for v in o_values.values()):
                continue

            better_or_equal = True
            strictly_better = False
            for prop, direction in objectives.items():
                cv = c_values[prop]
                ov = o_values[prop]
                if direction == "max":
                    if ov < cv:
                        better_or_equal = False
                        break
                    if ov > cv:
                        strictly_better = True
                else:
                    if ov > cv:
                        better_or_equal = False
                        break
                    if ov < cv:
                        strictly_better = True
            if better_or_equal and strictly_better:
                dominated = True
                break

        if not dominated:
            pareto.append(candidate)

    return pareto


def detect_outliers(materials: Iterable[MaterialProperties],
                    property_name: str,
                    z_threshold: float = 2.5) -> Tuple[List[MaterialProperties], List[MaterialProperties]]:
    """Detect outliers using Z-score. Returns (inliers, outliers)."""

    values: List[Tuple[MaterialProperties, float]] = []
    for mat in materials:
        value = getattr(mat, property_name, None)
        if isinstance(value, (int, float)) and not math.isnan(value):
            values.append((mat, float(value)))

    if not values:
        return [], []

    numbers = [v for _, v in values]
    mean_val = statistics.mean(numbers)
    std_val = statistics.pstdev(numbers) or 1e-12

    inliers, outliers = [], []
    for mat, value in values:
        z_score = abs((value - mean_val) / std_val)
        (outliers if z_score > z_threshold else inliers).append(mat)

    return inliers, outliers


# ---------------------------------------------------------------------------
# Comparison helpers


def compare_materials(material_names: Sequence[str],
                      properties: Sequence[Tuple[str, str]],
                      lab: Optional["MaterialsLab"] = None) -> List[ComparisonRecord]:
    """Compare materials across properties and produce normalized scores."""

    lab = _ensure_lab(lab)
    records: List[ComparisonRecord] = []

    mats = {name: lab.get_material(name) for name in material_names}

    for prop, unit in properties:
        values: Dict[str, float] = {}
        for name, mat in mats.items():
            if mat is None:
                continue
            value = getattr(mat, prop, None)
            if isinstance(value, (int, float)):
                values[name] = float(value)

        if not values:
            continue

        best = max(values.items(), key=lambda kv: kv[1])[0]
        max_val = max(values.values())
        min_val = min(values.values())
        span = max(max_val - min_val, 1e-9)
        normalized = {name: (val - min_val) / span for name, val in values.items()}

        records.append(
            ComparisonRecord(
                property_name=prop,
                unit=unit,
                values=values,
                best_material=best,
                normalized_scores=normalized,
            )
        )

    return records


# ---------------------------------------------------------------------------
# Batch experiments


def run_batch_experiments(batch: Sequence[Dict[str, Any]],
                          lab: Optional["MaterialsLab"] = None) -> List[BatchExperimentResult]:
    """Execute a sequence of experiments (tensile, fatigue, etc.)."""

    lab = _ensure_lab(lab)
    results: List[BatchExperimentResult] = []

    for index, spec in enumerate(batch, start=1):
        material = spec.get("material")
        test_type = spec.get("test")
        kwargs = spec.get("kwargs", {})
        experiment_id = f"batch-{index:04d}"

        if not material or not test_type:
            results.append(
                BatchExperimentResult(
                    experiment_id=experiment_id,
                    material=material or "unknown",
                    test_type=test_type or "unknown",
                    success=False,
                    payload={},
                    error="Material or test type missing",
                )
            )
            continue

        try:
            if test_type == "tensile":
                data = lab.tensile_test(material, **kwargs).data
            elif test_type == "compression":
                data = lab.compression_test(material, **kwargs).data
            elif test_type == "fatigue":
                data = lab.fatigue_test(material, **kwargs).data
            elif test_type == "impact":
                data = lab.impact_test(material, **kwargs).data
            elif test_type == "hardness":
                data = lab.hardness_test(material, **kwargs).data
            elif test_type == "thermal":
                data = lab.thermal_test(material, **kwargs)
            elif test_type == "corrosion":
                data = lab.corrosion_test(material, **kwargs)
            elif test_type == "environment":
                data = lab.environmental_test(material, **kwargs).data
            else:
                raise ValueError(f"Unsupported test type: {test_type}")

            results.append(
                BatchExperimentResult(
                    experiment_id=experiment_id,
                    material=material,
                    test_type=test_type,
                    success=True,
                    payload=data,
                )
            )
        except Exception as exc:  # pragma: no cover
            results.append(
                BatchExperimentResult(
                    experiment_id=experiment_id,
                    material=material,
                    test_type=test_type,
                    success=False,
                    payload={},
                    error=str(exc),
                )
            )

    return results


# ---------------------------------------------------------------------------
# Visualization (optional matplotlib dependency)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for visualization functions") from exc
    return plt


def plot_stress_strain(material: str,
                       lab: Optional["MaterialsLab"] = None,
                       ax=None):
    """Plot stress vs strain for the given material."""

    plt = _require_matplotlib()
    lab = _ensure_lab(lab)
    profile = lab.get_material_profile(material)
    curve = profile["mechanical"]["stress_strain"]

    if ax is None:
        fig, ax = plt.subplots()
        fig.suptitle(f"{material} – Stress/Strain")

    ax.plot(curve["abscissa"], curve["ordinate"], label=material)
    ax.set_xlabel("Strain (ε)")
    ax.set_ylabel("Stress (MPa)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    return ax


def plot_property_comparison(materials: Sequence[str],
                             property_name: str,
                             lab: Optional["MaterialsLab"] = None,
                             ax=None,
                             unit: str = ""):
    """Bar chart comparing a property across materials."""

    plt = _require_matplotlib()
    lab = _ensure_lab(lab)
    values = []
    for name in materials:
        mat = lab.get_material(name)
        if mat is None:
            continue
        value = getattr(mat, property_name, None)
        if isinstance(value, (int, float)):
            values.append((name, float(value)))

    if not values:
        raise ValueError("No numeric values found for comparison")

    labels, data = zip(*values)
    if ax is None:
        fig, ax = plt.subplots()
        fig.suptitle(f"{property_name} comparison")

    ax.bar(labels, data)
    ax.set_ylabel(f"{property_name} {f'({unit})' if unit else ''}")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    return ax


# ---------------------------------------------------------------------------
# Convenience helpers


def cli_summary(material_names: Sequence[str],
                properties: Sequence[str],
                lab: Optional["MaterialsLab"] = None) -> Dict[str, Any]:
    """Produce a JSON-friendly summary of selected materials."""

    lab = _ensure_lab(lab)
    payload: Dict[str, Any] = {"materials": []}

    for name in material_names:
        mat = lab.get_material(name)
        if mat is None:
            continue
        entry = {"name": name, "properties": {}}
        for prop in properties:
            entry["properties"][prop] = getattr(mat, prop, None)
        payload["materials"].append(entry)

    return payload
