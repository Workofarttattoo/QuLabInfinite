"""Utilities to propagate chemistry results into other laboratory modules."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from materials_lab.materials_lab import MaterialsLab
    from environmental_sim.environmental_sim import EnvironmentalSimulator


def _format_material_note(reaction_name: str, kinetics: Dict[str, Any], thermodynamics: Dict[str, Any]) -> str:
    """Create a concise annotation string for material records."""
    rate_constant = kinetics.get("rate_constant")
    activation_energy = kinetics.get("activation_energy")
    delta_h = thermodynamics.get("delta_h_kcal_per_mol")
    delta_g = thermodynamics.get("delta_g_kcal_per_mol")

    fragments: List[str] = [f"Reaction: {reaction_name}"]
    if rate_constant is not None:
        fragments.append(f"k={rate_constant:.2e}")
    if activation_energy is not None:
        fragments.append(f"Ea={activation_energy:.2f} kcal/mol")
    if delta_h is not None:
        fragments.append(f"ΔH={delta_h:.2f} kcal/mol")
    if delta_g is not None:
        fragments.append(f"ΔG={delta_g:.2f} kcal/mol")

    return "[ChemLab] " + ", ".join(fragments)


def apply_material_updates(
    materials_lab: Optional["MaterialsLab"],
    materials_payload: Dict[str, Any]
) -> None:
    """Attach chemistry insights to materials referenced in the payload."""
    if materials_lab is None or not materials_payload:
        return

    links: Dict[str, Iterable[str]] = materials_payload.get("links", {})
    reaction_name: str = materials_payload.get("reaction_name", "unknown")
    kinetics: Dict[str, Any] = materials_payload.get("kinetics", {})
    thermodynamics: Dict[str, Any] = materials_payload.get("thermodynamics", {})

    note = _format_material_note(reaction_name, kinetics, thermodynamics)

    hazards: set[str] = {
        str(h).lower()
        for h in materials_payload.get("hazards", [])
        if isinstance(h, str)
    }
    delta_h = thermodynamics.get("delta_h_kcal_per_mol")

    adjustments: Dict[str, Dict[str, Dict[str, float]]] = materials_payload.get("effects", {})
    explicit_adjustments = {key for key in adjustments}
    processed_materials: set[str] = set()
    hazard_applied: set[str] = set()

    def _append_note(material: Any, message: str) -> None:
        """Add contextual note to material without duplicating entries."""
        if not message:
            return
        existing = material.notes or ""
        notes = existing.splitlines()
        if message in notes:
            return
        material.notes = f"{existing}\n{message}".strip() if existing else message

    def _resolve_property_update(current: Any, config: Any) -> Any:
        """Resolve new property value according to adjustment config."""
        if isinstance(config, dict):
            if "set" in config:
                return config["set"], config["set"] != current

            if isinstance(current, bool):
                return current, False

            if isinstance(current, Real):
                value = float(current)
                changed = False

                if "multiplier" in config:
                    try:
                        value *= float(config["multiplier"])
                        changed = True
                    except (TypeError, ValueError):
                        pass

                if "delta" in config:
                    try:
                        value += float(config["delta"])
                        changed = True
                    except (TypeError, ValueError):
                        pass

                if "min" in config:
                    try:
                        min_val = float(config["min"])
                        new_val = max(value, min_val)
                        if new_val != value:
                            changed = True
                        value = new_val
                    except (TypeError, ValueError):
                        pass

                if "max" in config:
                    try:
                        max_val = float(config["max"])
                        new_val = min(value, max_val)
                        if new_val != value:
                            changed = True
                        value = new_val
                    except (TypeError, ValueError):
                        pass

                if not changed:
                    return current, False

                if not math.isfinite(value):
                    return current, False

                if isinstance(current, int):
                    return type(current)(round(value)), True

                return value, True

            return current, False

        if config == current:
            return current, False

        return config, True

    rating_order = ["excellent", "good", "moderate", "poor"]
    base_corrosion_rates = {
        "excellent": 0.05,
        "good": 0.2,
        "moderate": 1.0,
        "poor": 5.0,
    }

    def _degrade_rating(value: Any, steps: int = 1) -> Any:
        if not isinstance(value, str):
            return value
        key = value.strip().lower()
        if key not in rating_order:
            return value
        idx = rating_order.index(key)
        idx = min(idx + max(steps, 0), len(rating_order) - 1)
        return rating_order[idx]

    def _apply_adjustments(material_name: str, effect: Dict[str, Any]) -> None:
        material = materials_lab.get_material(material_name)
        if material is None:
            return
        applied = False
        extra_notes: List[str] = []

        for prop, config in effect.items():
            if prop == "notes":
                if isinstance(config, str) and config.strip():
                    extra_notes.append(config.strip())
                elif isinstance(config, (list, tuple, set)):
                    for entry in config:
                        if isinstance(entry, str) and entry.strip():
                            extra_notes.append(entry.strip())
                continue

            if not hasattr(material, prop):
                continue
            current = getattr(material, prop)
            new_value, changed = _resolve_property_update(current, config)
            if not changed:
                continue

            setattr(material, prop, new_value)
            applied = True

        if applied or extra_notes:
            combined_note = note
            if extra_notes:
                combined_note = f"{note} | {' | '.join(extra_notes)}"
            _append_note(material, combined_note)

    def _apply_hazard_adjustment(material_name: str) -> None:
        if material_name in hazard_applied:
            return
        material = materials_lab.get_material(material_name)
        if material is None:
            return

        rating = getattr(material, "corrosion_resistance", "moderate")
        base_rate = base_corrosion_rates.get(str(rating).lower(), 1.0)

        multiplier = 1.0
        degrade_steps = 0

        if "corrosive" in hazards:
            multiplier *= 1.6
            degrade_steps += 1
        if "oxidizer" in hazards:
            multiplier *= 1.25
        if "acid" in hazards or "acidic" in hazards:
            multiplier *= 1.1
        if "alkaline" in hazards or "base" in hazards:
            multiplier *= 1.05
        if isinstance(delta_h, (int, float)):
            if delta_h < -40.0:
                multiplier *= 1.15
            elif delta_h > 25.0:
                multiplier *= 0.95

        if degrade_steps > 0:
            new_rating = rating
            for _ in range(degrade_steps):
                candidate = _degrade_rating(new_rating)
                if candidate == new_rating:
                    break
                new_rating = candidate
            try:
                setattr(material, "corrosion_resistance", new_rating)
                rating = new_rating
            except AttributeError:
                pass

        corrosion_rate = base_corrosion_rates.get(str(rating).lower(), base_rate) * multiplier
        prev_rate = getattr(material, "chemistry_corrosion_rate_mm_per_year", None)
        try:
            if prev_rate is not None:
                corrosion_rate = max(corrosion_rate, float(prev_rate))
        except (TypeError, ValueError):
            pass

        setattr(material, "chemistry_corrosion_rate_mm_per_year", corrosion_rate)
        setattr(material, "chemistry_last_reaction", reaction_name)
        _append_note(material, note)
        hazard_applied.add(material_name)

    for material_name, effect in adjustments.items():
        _apply_adjustments(material_name, effect)
        processed_materials.add(material_name)
        _apply_hazard_adjustment(material_name)

    for collection_key in ("product_ids", "reactant_ids", "byproduct_ids"):
        for material_id in links.get(collection_key, []):
            if material_id in explicit_adjustments:
                continue
            _apply_hazard_adjustment(material_id)


def apply_environmental_adjustments(
    environment_simulator: Optional["EnvironmentalSimulator"],
    environment_payload: List[Dict[str, Any]]
) -> None:
    """
    Annotate the environmental simulator with chemistry outputs.

    Adds gaseous by-products as contaminants and records the payload for
    later inspection by the environment team.
    """
    if environment_simulator is None or not environment_payload:
        return

    controller = getattr(environment_simulator, "controller", None)
    if controller is None:
        return

    if not hasattr(controller, "chemistry_feeds"):
        controller.chemistry_feeds = []  # type: ignore[attr-defined]

    controller.chemistry_feeds.extend(environment_payload)  # type: ignore[attr-defined]

    atmosphere = getattr(controller, "atmosphere", None)
    record_corrosion_effect = getattr(controller, "record_corrosion_effect", None)
    set_corrosion_baseline = getattr(controller, "set_corrosion_baseline", None)

    if record_corrosion_effect is None and not hasattr(controller, "corrosion_impacts"):
        controller.corrosion_impacts = {}  # type: ignore[attr-defined]

    if not hasattr(controller, "chemistry_emission_profiles"):
        controller.chemistry_emission_profiles = {}  # type: ignore[attr-defined]

    def _parse_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _build_concentration_profile(rate_ppm_per_hour: float, duration_hours: float, decay_constant: float) -> (float, List[Dict[str, float]]):
        if rate_ppm_per_hour <= 0 or duration_hours <= 0:
            return 0.0, [{"time_hours": 0.0, "ppm": 0.0}]

        if decay_constant > 0:
            steady_value = (rate_ppm_per_hour / decay_constant) * (1.0 - math.exp(-decay_constant * duration_hours))
        else:
            steady_value = rate_ppm_per_hour * duration_hours

        def concentration_at(time_hours: float) -> float:
            if time_hours <= 0:
                return 0.0
            if time_hours <= duration_hours:
                if decay_constant > 0:
                    return (rate_ppm_per_hour / decay_constant) * (1.0 - math.exp(-decay_constant * time_hours))
                return rate_ppm_per_hour * time_hours

            if decay_constant > 0:
                tail_time = time_hours - duration_hours
                return steady_value * math.exp(-decay_constant * tail_time)

            return steady_value

        step_count = max(3, min(24, int(math.ceil(duration_hours * 2.0))))
        step = duration_hours / step_count if step_count else duration_hours
        profile: List[Dict[str, float]] = [{"time_hours": 0.0, "ppm": 0.0}]

        for index in range(1, step_count + 1):
            time_point = round(step * index, 6)
            ppm_value = max(0.0, concentration_at(time_point))
            profile.append({"time_hours": time_point, "ppm": round(ppm_value, 6)})

        if decay_constant > 0:
            half_life = math.log(2.0) / decay_constant
            for multiplier in (1.0, 2.0):
                tail_time = duration_hours + half_life * multiplier
                ppm_value = steady_value * (0.5 ** multiplier)
                profile.append({
                    "time_hours": round(tail_time, 6),
                    "ppm": round(max(0.0, ppm_value), 6)
                })

        return steady_value, profile

    for emission in environment_payload:
        target_material = emission.get("target_material")
        corrosion_multiplier = _parse_float(emission.get("corrosion_rate_multiplier"))

        release_rate = max(_parse_float(emission.get("estimated_release_rate"), 0.0) or 0.0, 0.0)
        duration_hours = max(_parse_float(emission.get("exposure_hours"), 1.0) or 1.0, 0.0)
        removal_eff = _parse_float(emission.get("removal_efficiency"), 0.0) or 0.0
        removal_factor = max(0.0, min(1.0, 1.0 - removal_eff))
        decay_constant = 0.0
        decay_half_life = _parse_float(emission.get("decay_half_life_hours"))
        if decay_half_life and decay_half_life > 0:
            decay_constant = math.log(2.0) / decay_half_life

        rate_ppm_per_hour = release_rate * removal_factor * 1e3
        final_ppm, profile = _build_concentration_profile(rate_ppm_per_hour, duration_hours, decay_constant)

        baseline_rate = _parse_float(emission.get("baseline_corrosion_rate_mm_per_year"))
        if baseline_rate is None:
            baseline_rate = _parse_float(emission.get("baseline_corrosion_rate"))
        if target_material and baseline_rate is not None and set_corrosion_baseline is not None:
            extra_metadata = {
                "source_material": emission.get("material_id"),
                "source_name": emission.get("name"),
                "phase": emission.get("phase"),
            }
            set_corrosion_baseline(  # type: ignore[misc]
                target_material,
                baseline_rate,
                metadata={k: v for k, v in extra_metadata.items() if v is not None}
            )

        if target_material and corrosion_multiplier is not None:
            if record_corrosion_effect is not None:
                source_info = {
                    "material_id": emission.get("material_id"),
                    "name": emission.get("name"),
                    "duration_hours": duration_hours,
                    "peak_ppm": final_ppm,
                }
                metadata = {
                    "phase": emission.get("phase"),
                    "release_rate": release_rate,
                    "peak_ppm": final_ppm,
                }
                record_corrosion_effect(  # type: ignore[misc]
                    target_material,
                    corrosion_multiplier,
                    duration_hours,
                    source=source_info,
                    metadata={k: v for k, v in metadata.items() if v is not None}
                )
            else:
                record = controller.corrosion_impacts.setdefault(target_material, {  # type: ignore[attr-defined]
                    "multiplier": 1.0,
                    "exposure_hours": 0.0,
                    "sources": [],
                    "peak_ppm": 0.0
                })
                record["multiplier"] *= corrosion_multiplier
                record["exposure_hours"] += duration_hours
                record["peak_ppm"] = max(record.get("peak_ppm", 0.0), final_ppm)
                record["sources"].append({
                    "material_id": emission.get("material_id"),
                    "name": emission.get("name"),
                    "multiplier": corrosion_multiplier,
                    "duration_hours": duration_hours,
                    "peak_ppm": final_ppm,
                })

        contaminant = emission.get("material_id") or emission.get("name") or "anon_gas"
        profile_entry = {
            "material_id": emission.get("material_id"),
            "name": emission.get("name"),
            "phase": emission.get("phase"),
            "release_rate": release_rate,
            "duration_hours": duration_hours,
            "decay_half_life_hours": decay_half_life,
            "removal_efficiency": removal_eff,
            "peak_ppm": final_ppm,
            "profile": profile,
        }
        controller.chemistry_emission_profiles.setdefault(contaminant, []).append(profile_entry)  # type: ignore[attr-defined]

        if emission.get("phase") != "gas" or atmosphere is None or final_ppm <= 0:
            continue

        atmosphere.add_contaminant(
            contaminant,
            final_ppm,
            half_life_hours=decay_half_life,
            removal_efficiency=removal_eff,
        )
