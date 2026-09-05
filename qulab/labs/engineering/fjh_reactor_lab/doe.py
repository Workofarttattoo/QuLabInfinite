"""
Design of Experiments engine for FJH virtual experiments.
"""

from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from qulab.hive_mind.crystalline_intent import (
    ExperimentDesigner,
    Parameter,
)


@dataclass
class FJHExperimentFactor:
    """Scientific variable for FJH DOE."""

    name: str
    param_type: str  # continuous, discrete, categorical
    min_value: float | None = None
    max_value: float | None = None
    levels: list[Any] | None = None
    unit: str = ""
    description: str = ""


@dataclass
class FJHObjective:
    """Competing optimization objective."""

    name: str
    direction: str  # maximize, minimize
    weight: float = 1.0


@dataclass
class FJHExperimentRun:
    """Single virtual experiment run specification."""

    run_id: str
    parameters: dict[str, Any]
    design_method: str


@dataclass
class ParetoPoint:
    """Point on Pareto front."""

    run_id: str
    objectives: dict[str, float]
    parameters: dict[str, Any]


DEFAULT_FJH_FACTORS: list[FJHExperimentFactor] = [
    FJHExperimentFactor("relative_pulse_duration", "continuous", 0.1, 2.0, unit="relative"),
    FJHExperimentFactor("relative_stored_energy", "continuous", 0.5, 1.0, unit="relative"),
    FJHExperimentFactor("sample_resistance_ohm", "continuous", 0.05, 0.5, unit="ohm"),
    FJHExperimentFactor("contact_resistance_ohm", "continuous", 0.001, 0.05, unit="ohm"),
    FJHExperimentFactor("precursor_loading_wt_percent", "continuous", 0.1, 5.0, unit="wt%"),
    FJHExperimentFactor("precursor_uniformity", "continuous", 0.2, 1.0, unit="score"),
    FJHExperimentFactor("atmosphere", "categorical", levels=["vacuum", "argon"]),
    FJHExperimentFactor("chamber_pressure_Pa", "continuous", 100, 101325, unit="Pa"),
    FJHExperimentFactor("ambient_temperature_K", "continuous", 273, 323, unit="K"),
    FJHExperimentFactor("ultrasound_enabled", "categorical", levels=[False, True]),
]

DEFAULT_OBJECTIVES: list[FJHObjective] = [
    FJHObjective("graphene_conversion_score", "maximize"),
    FJHObjective("au_single_atom_retention_score", "maximize"),
    FJHObjective("au_cluster_risk", "minimize"),
    FJHObjective("au_loss_risk", "minimize"),
    FJHObjective("electrical_stress", "minimize"),
    FJHObjective("thermal_stress", "minimize"),
    FJHObjective("confidence", "maximize"),
]


def one_factor_at_a_time(
    factors: list[FJHExperimentFactor],
    baseline: dict[str, Any],
    n_levels: int = 3,
) -> list[FJHExperimentRun]:
    """OFAT exploration around baseline."""
    runs = []
    for factor in factors:
        if factor.param_type == "categorical" and factor.levels:
            values = factor.levels
        elif factor.min_value is not None and factor.max_value is not None:
            values = np.linspace(factor.min_value, factor.max_value, n_levels).tolist()
        else:
            continue
        for val in values:
            params = dict(baseline)
            params[factor.name] = val
            runs.append(FJHExperimentRun(
                run_id=str(uuid.uuid4())[:8],
                parameters=params,
                design_method="one_factor_at_a_time",
            ))
    return runs


def factorial_doe(
    factors: list[FJHExperimentFactor],
    levels: int = 2,
) -> list[FJHExperimentRun]:
    """Full factorial over selected factors (limited to 4 factors for practicality)."""
    selected = factors[:4]
    param_objs = []
    level_values = []
    for f in selected:
        param_objs.append(Parameter(
            name=f.name,
            param_type=f.param_type,
            min_value=f.min_value,
            max_value=f.max_value,
            discrete_values=f.levels,
        ))
        if f.levels:
            level_values.append(f.levels[:levels])
        else:
            level_values.append(
                np.linspace(f.min_value or 0, f.max_value or 1, levels).tolist()
            )

    runs = []
    for combo in itertools.product(*level_values):
        params = {f.name: v for f, v in zip(selected, combo)}
        runs.append(FJHExperimentRun(
            run_id=str(uuid.uuid4())[:8],
            parameters=params,
            design_method="full_factorial",
        ))
    return runs


def latin_hypercube_doe(
    factors: list[FJHExperimentFactor],
    n_samples: int = 20,
    seed: int = 42,
) -> list[FJHExperimentRun]:
    """Latin Hypercube Sampling."""
    continuous = [f for f in factors if f.param_type == "continuous"]
    if not continuous:
        return []

    designer = ExperimentDesigner()
    params = [
        Parameter(
            name=f.name,
            param_type="continuous",
            min_value=f.min_value,
            max_value=f.max_value,
        )
        for f in continuous
    ]
    matrix = designer.design_latin_hypercube(params, n_samples)
    rng = np.random.default_rng(seed)

    runs = []
    for row in matrix:
        pdict = {f.name: float(row[i]) for i, f in enumerate(continuous)}
        for f in factors:
            if f.param_type == "categorical" and f.levels:
                pdict[f.name] = rng.choice(f.levels)
        runs.append(FJHExperimentRun(
            run_id=str(uuid.uuid4())[:8],
            parameters=pdict,
            design_method="latin_hypercube",
        ))
    return runs


def compute_pareto_front(
    results: list[dict[str, Any]],
    objectives: list[FJHObjective],
) -> list[ParetoPoint]:
    """Compute non-dominated Pareto front from experiment results."""
    points = []
    for r in results:
        obj_vals = {}
        for o in objectives:
            val = r.get("objectives", {}).get(o.name, 0.0)
            obj_vals[o.name] = val
        points.append(ParetoPoint(
            run_id=r.get("run_id", ""),
            objectives=obj_vals,
            parameters=r.get("parameters", {}),
        ))

    def dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
        better_or_equal = True
        strictly_better = False
        for o in objectives:
            av, bv = a.objectives.get(o.name, 0), b.objectives.get(o.name, 0)
            if o.direction == "maximize":
                if av < bv:
                    better_or_equal = False
                if av > bv:
                    strictly_better = True
            else:
                if av > bv:
                    better_or_equal = False
                if av < bv:
                    strictly_better = True
        return better_or_equal and strictly_better

    front = []
    for p in points:
        if not any(dominates(other, p) for other in points if other.run_id != p.run_id):
            front.append(p)
    return front
