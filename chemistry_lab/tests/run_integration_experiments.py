"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

run_integration_experiments - Part of Chemistry Lab
"""

"""Utility to generate integration experiment data for regression testing."""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chemistry_lab.integration import apply_environmental_adjustments, apply_material_updates
from chemistry_lab.reaction_simulator import (
    ReactionConditions,
    ReactionKinetics,
    ReactionSimulator,
)
from environmental_sim import EnvironmentalSimulator
from materials_lab.materials_lab import MaterialsLab


OUTPUT_PATH = Path(__file__).parent / "data" / "integration_experiments.json"
DASHBOARD_PATH = Path(__file__).parent / "data" / "integration_dashboard.json"
GAS_CONSTANT_KCAL = 0.00198720425864083  # kcal/(mol*K)
EXPERIMENT_COUNT = 100
RANDOM_SEED = 42


def _ensure_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _compute_rate_constant(metadata, temperature_k: float) -> float:
    params = metadata.kinetics
    pre_exp = float(params.arrhenius_A)
    exponent = float(params.temperature_exponent)
    activation = float(params.activation_energy_kcal_per_mol)

    temperature_term = temperature_k ** exponent if exponent else 1.0
    barrier_term = math.exp(-activation / (GAS_CONSTANT_KCAL * temperature_k))
    return pre_exp * temperature_term * barrier_term


def _derive_half_life(rate_constant: float) -> float:
    if rate_constant <= 0:
        return float("inf")
    return math.log(2.0) / rate_constant


def _select_solvent(metadata) -> Optional[str]:
    if not metadata.solvent_effects:
        return None

    weights = [max(effect.rate_factor, 0.1) for effect in metadata.solvent_effects]
    total = sum(weights)
    if total <= 0:
        return random.choice(metadata.solvent_effects).solvent

    pick = random.uniform(0, total)
    cumulative = 0.0
    for effect, weight in zip(metadata.solvent_effects, weights):
        cumulative += weight
        if pick <= cumulative:
            return effect.solvent

    return metadata.solvent_effects[-1].solvent


def _summarise_materials(materials_payload: Dict[str, Any], lab: MaterialsLab) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    effects: Dict[str, Dict[str, Any]] = materials_payload.get("effects", {})

    for material_name, adjustments in effects.items():
        material = lab.get_material(material_name)
        if material is None:
            continue

        entry: Dict[str, Any] = {}
        for prop in adjustments:
            if not hasattr(material, prop):
                continue
            value = getattr(material, prop)
            if isinstance(value, (int, float)):
                entry[prop] = float(value)
            else:
                entry[prop] = value

        notes = material.notes.splitlines() if material.notes else []
        entry["note_tail"] = notes[-1] if notes else ""
        summary[material_name] = entry

    return summary


def _summarise_corrosion_state(controller) -> Dict[str, Any]:
    if hasattr(controller, "get_corrosion_state"):
        state = controller.get_corrosion_state()
    elif hasattr(controller, "corrosion_impacts"):
        state = controller.corrosion_impacts  # type: ignore[attr-defined]
    else:
        return {}

    summary: Dict[str, Any] = {}
    if isinstance(state, dict):
        for material_id, payload in state.items():
            entry: Dict[str, Any] = {}
            if isinstance(payload, dict):
                for key in ("active_multiplier", "multiplier"):
                    if key in payload:
                        entry["multiplier"] = _ensure_float(payload[key])
                        break
                if "total_exposure_hours" in payload:
                    entry["exposure_hours"] = _ensure_float(payload["total_exposure_hours"])
                elif "exposure_hours" in payload:
                    entry["exposure_hours"] = _ensure_float(payload["exposure_hours"])
                if "metadata" in payload and isinstance(payload["metadata"], dict):
                    meta = payload["metadata"]
                    if "peak_ppm" in meta:
                        entry["peak_ppm"] = _ensure_float(meta["peak_ppm"])
                if "peak_ppm" in payload:
                    entry["peak_ppm"] = _ensure_float(payload["peak_ppm"])

                sources: Iterable[Dict[str, Any]] = payload.get("sources", []) if isinstance(payload.get("sources"), list) else []
                entry["sources"] = [
                    {
                        "material_id": source.get("material_id"),
                        "multiplier": _ensure_float(source.get("multiplier", 1.0)),
                        "peak_ppm": _ensure_float(source.get("peak_ppm", 0.0)),
                    }
                    for source in list(sources)[:3]
                ]

            summary[str(material_id)] = entry

    return summary


def _summarise_emissions(controller) -> Dict[str, Any]:
    profiles = getattr(controller, "chemistry_emission_profiles", {})
    if not isinstance(profiles, dict):
        return {}

    summary: Dict[str, Any] = {}
    for contaminant, entries in profiles.items():
        if not entries:
            continue
        max_peak = 0.0
        samples: List[Dict[str, float]] = []
        total_entries = 0
        for entry in entries:
            total_entries += 1
            peak = _ensure_float(entry.get("peak_ppm"))
            max_peak = max(max_peak, peak)
            if not samples and isinstance(entry, dict):
                profile = entry.get("profile", [])
                for point in profile[:5]:
                    if isinstance(point, dict):
                        samples.append({
                            "time_hours": _ensure_float(point.get("time_hours")),
                            "ppm": _ensure_float(point.get("ppm")),
                        })

        summary[str(contaminant)] = {
            "entries": total_entries,
            "max_peak_ppm": max_peak,
            "profile_samples": samples,
        }

    return summary


def run_experiments(count: int = EXPERIMENT_COUNT) -> Dict[str, Any]:
    random.seed(RANDOM_SEED)

    reaction_sim = ReactionSimulator()
    reactions = reaction_sim.list_database_reactions()
    if not reactions:
        raise RuntimeError("No reactions available in the catalog.")

    experiments: List[Dict[str, Any]] = []

    for index in range(1, count + 1):
        reaction_name = random.choice(reactions)
        metadata = reaction_sim.get_reaction_metadata(reaction_name)
        if metadata is None:
            continue

        temperature = random.uniform(280.0, 360.0)
        pressure = random.uniform(0.8, 5.0)
        solvent = _select_solvent(metadata)
        concentration = random.uniform(0.05, 2.0)

        product_labels = metadata.products or ["product"]
        conditions = ReactionConditions(
            temperature=temperature,
            pressure=pressure,
            solvent=solvent,
        )
        selectivity = reaction_sim._derive_selectivity_profile(metadata, conditions, product_labels)  # pylint: disable=protected-access

        rate_constant = _compute_rate_constant(metadata, temperature)
        half_life = _derive_half_life(rate_constant)
        equilibrium_constant = metadata.thermodynamics.equilibrium_constant(temperature, GAS_CONSTANT_KCAL)

        kinetics = ReactionKinetics(
            rate_constant=rate_constant,
            activation_energy=float(metadata.kinetics.activation_energy_kcal_per_mol),
            pre_exponential_factor=float(metadata.kinetics.arrhenius_A),
            reaction_order=int(metadata.kinetics.reaction_order),
            half_life=half_life,
            equilibrium_constant=equilibrium_constant,
            product_selectivity=dict(selectivity),
        )

        payload = reaction_sim.build_integration_payload(metadata, kinetics, conditions, initial_concentration=concentration)

        buffer = StringIO()
        with redirect_stdout(buffer):
            materials_lab = MaterialsLab()

        env_sim = EnvironmentalSimulator()

        apply_material_updates(materials_lab, payload.get("materials", {}))
        apply_environmental_adjustments(env_sim, payload.get("environment", []))

        experiment_record = {
            "id": index,
            "reaction": metadata.name,
            "conditions": {
                "temperature_K": round(temperature, 3),
                "pressure_bar": round(pressure, 3),
                "solvent": solvent,
                "initial_concentration": round(concentration, 4),
            },
            "kinetics": {
                "rate_constant": rate_constant,
                "half_life": half_life,
                "equilibrium_constant": equilibrium_constant,
                "selectivity": selectivity,
            },
            "materials": _summarise_materials(payload.get("materials", {}), materials_lab),
            "environment": {
                "corrosion": _summarise_corrosion_state(env_sim.controller),
                "emissions": _summarise_emissions(env_sim.controller),
            },
            "raw_payload": {
                "materials": {
                    "links": payload.get("materials", {}).get("links", {}),
                    "effects": payload.get("materials", {}).get("effects", {}),
                },
                "environment": payload.get("environment", []),
            },
        }

        experiments.append(experiment_record)

    return {
        "count": len(experiments),
        "requested": count,
        "seed": RANDOM_SEED,
        "experiments": experiments,
    }


def build_dashboard(results: Dict[str, Any]) -> Dict[str, Any]:
    experiments: List[Dict[str, Any]] = list(results.get("experiments", []))

    reaction_distribution = Counter(exp.get("reaction") for exp in experiments)

    corrosion_totals: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "occurrences": 0.0,
            "total_multiplier": 0.0,
            "max_multiplier": 0.0,
            "total_exposure_hours": 0.0,
        }
    )

    for experiment in experiments:
        corrosion = experiment.get("environment", {}).get("corrosion", {})
        if not isinstance(corrosion, dict):
            continue
        for material_id, payload in corrosion.items():
            stats = corrosion_totals[material_id]
            multiplier = _ensure_float(payload.get("multiplier") or payload.get("active_multiplier") or 1.0)
            exposure = _ensure_float(payload.get("exposure_hours") or payload.get("total_exposure_hours") or 0.0)
            stats["occurrences"] += 1.0
            stats["total_multiplier"] += multiplier
            stats["total_exposure_hours"] += exposure
            stats["max_multiplier"] = max(stats["max_multiplier"], multiplier)

    corrosion_summary: Dict[str, Dict[str, float]] = {}
    for material_id, stats in corrosion_totals.items():
        occurrences = int(stats["occurrences"])
        avg_multiplier = stats["total_multiplier"] / occurrences if occurrences else 0.0
        avg_exposure = stats["total_exposure_hours"] / occurrences if occurrences else 0.0
        corrosion_summary[material_id] = {
            "occurrences": occurrences,
            "avg_multiplier": avg_multiplier,
            "max_multiplier": stats["max_multiplier"],
            "avg_exposure_hours": avg_exposure,
        }

    emission_counts: Counter[str] = Counter()
    emission_peaks: Dict[str, float] = {}

    for experiment in experiments:
        emissions = experiment.get("environment", {}).get("emissions", {})
        if not isinstance(emissions, dict):
            continue
        for contaminant, entry in emissions.items():
            emission_counts[contaminant] += int(entry.get("entries", 0))
            emission_peaks[contaminant] = max(
                emission_peaks.get(contaminant, 0.0),
                _ensure_float(entry.get("max_peak_ppm")),
            )

    return {
        "seed": results.get("seed"),
        "experiments": results.get("count"),
        "reactions": dict(reaction_distribution),
        "reactions_top5": reaction_distribution.most_common(5),
        "corrosion": corrosion_summary,
        "emissions": {
            "entries": dict(emission_counts),
            "max_peaks": emission_peaks,
            "top5": emission_counts.most_common(5),
        },
    }


def main() -> None:
    results = run_experiments(EXPERIMENT_COUNT)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    dashboard = build_dashboard(results)
    DASHBOARD_PATH.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
    print(f"[info] Generated {results['count']} experiments -> {OUTPUT_PATH}")
    print(f"[info] Wrote dashboard summary -> {DASHBOARD_PATH}")


if __name__ == "__main__":
    main()
