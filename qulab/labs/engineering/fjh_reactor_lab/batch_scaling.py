"""
Virtual capacitor-count scaling for larger FJH batch masses.

Uses the same 900 uF / 450 V parts. Does not recommend firing a larger bank.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .config import ReactorConfiguration
from .thermal import VULCAN_XC72_THERMAL, _effective_cp


@dataclass
class BatchScaleCase:
    """One scaling scenario."""

    name: str
    target_description: str
    energy_J: float
    energy_density_J_per_g: float
    capacitor_count: int
    stored_energy_at_count_J: float
    adiabatic_peak_temperature_K: float
    graphene_relevant: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_description": self.target_description,
            "energy_J": self.energy_J,
            "energy_density_J_per_g": self.energy_density_J_per_g,
            "capacitor_count": self.capacitor_count,
            "stored_energy_at_count_J": self.stored_energy_at_count_J,
            "adiabatic_peak_temperature_K": self.adiabatic_peak_temperature_K,
            "graphene_relevant": self.graphene_relevant,
            "notes": self.notes,
        }


def energy_per_capacitor_J(config: ReactorConfiguration | None = None) -> float:
    cfg = config or ReactorConfiguration.default_fjh_bank()
    c_each_F = cfg.effective_capacitance_each_uF() * 1e-6
    v = cfg.initial_voltage_V or cfg.capacitor_nominal_voltage_V
    return 0.5 * c_each_F * v ** 2


def capacitors_for_energy_J(energy_J: float, e_each_J: float) -> int:
    if e_each_J <= 0:
        raise ValueError("Energy per capacitor must be positive")
    return int(math.ceil(energy_J / e_each_J))


def adiabatic_energy_for_temperature_J(
    mass_g: float,
    target_T_K: float,
    t0_K: float = 298.15,
    cp_J_kg_K: float = 710.0,
    sample_energy_fraction: float = 1.0,
) -> float:
    """Bank energy needed so m*cp*dT is met after the given sample-energy fraction."""
    mass_kg = mass_g / 1000.0
    delta_T = max(target_T_K - t0_K, 0.0)
    e_sample = mass_kg * cp_J_kg_K * delta_T
    if sample_energy_fraction <= 0:
        raise ValueError("sample_energy_fraction must be > 0")
    return e_sample / sample_energy_fraction


def scale_batch_mass(
    mass_g: float = 20.0,
    config: ReactorConfiguration | None = None,
    electrode_sink_fraction: float = 0.30,
) -> dict[str, Any]:
    """
    Estimate how many identical capacitors are needed for a batch mass.

    electrode_sink_fraction is a HYPOTHESIS (graphite rods). It is not measured.
    """
    cfg = config or ReactorConfiguration.physical_lab_setup()
    e_each = energy_per_capacitor_J(cfg)
    cp = _effective_cp(VULCAN_XC72_THERMAL)
    t0 = cfg.initial_sample_temperature_K
    v = cfg.initial_voltage_V or cfg.capacitor_nominal_voltage_V
    current_n = cfg.capacitor_count
    current_E = current_n * e_each
    sample_fraction = max(1.0 - electrode_sink_fraction, 0.05)

    current_density = current_E / mass_g
    current_dT = (current_E * sample_fraction) / ((mass_g / 1000.0) * cp)
    current_T = t0 + current_dT

    cases = []

    def _case(name, desc, energy, graphene, notes, fraction=1.0):
        n = capacitors_for_energy_J(energy, e_each)
        stored = n * e_each
        dT = (stored * fraction) / ((mass_g / 1000.0) * cp)
        cases.append(BatchScaleCase(
            name=name,
            target_description=desc,
            energy_J=energy,
            energy_density_J_per_g=energy / mass_g,
            capacitor_count=n,
            stored_energy_at_count_J=stored,
            adiabatic_peak_temperature_K=t0 + dT,
            graphene_relevant=graphene,
            notes=notes,
        ))

    _case(
        "keep_current_12_caps",
        f"{current_n} caps on {mass_g:g} g (no added capacitors)",
        current_E,
        False,
        f"Only {current_density:.1f} J/g. Temperature rise is tens of kelvin, not flash heating.",
        fraction=sample_fraction,
    )
    _case(
        "match_1g_energy_density",
        "Same J/g as the current 1 g / 12-cap setup (already too cold for graphene)",
        mass_g * (current_E / 1.0),
        False,
        "Matches the 1 g energy density that peaked ~1200-1800 K. Still a weak graphene case.",
        fraction=1.0,
    )
    _case(
        "adiabatic_2000K",
        "Adiabatic 2000 K, all bank energy into the sample, no electrode sink",
        adiabatic_energy_for_temperature_J(mass_g, 2000.0, t0, cp, 1.0),
        False,
        "Lower-bound graphitization-adjacent temperature. Ignores graphite-rod heat sinking.",
        fraction=1.0,
    )
    _case(
        "adiabatic_2500K",
        "Adiabatic 2500 K, all bank energy into the sample, no electrode sink",
        adiabatic_energy_for_temperature_J(mass_g, 2500.0, t0, cp, 1.0),
        True,
        "Closer to flash-graphene temperature literature. Still ignores rod heat sinking.",
        fraction=1.0,
    )
    _case(
        "adiabatic_3000K",
        "Adiabatic 3000 K, all bank energy into the sample, no electrode sink",
        adiabatic_energy_for_temperature_J(mass_g, 3000.0, t0, cp, 1.0),
        True,
        "Upper flash-graphene-like adiabatic target. Ignores radiation and rod sinking.",
        fraction=1.0,
    )
    _case(
        "2500K_with_graphite_sink",
        f"2500 K after hypothesized {electrode_sink_fraction*100:.0f}% electrode heat sink",
        adiabatic_energy_for_temperature_J(mass_g, 2500.0, t0, cp, sample_fraction),
        True,
        "Electrode sink fraction is a HYPOTHESIS until rod dimensions are measured. "
        "This is a virtual energy budget, not a build/fire plan.",
        fraction=sample_fraction,
    )

    recommended = next(c for c in cases if c.name == "2500K_with_graphite_sink")

    return {
        "mass_g": mass_g,
        "capacitor_each_capacitance_uF": cfg.capacitor_each_capacitance_uF,
        "capacitor_nominal_voltage_V": v,
        "energy_per_capacitor_J": e_each,
        "current_bank_count": current_n,
        "current_bank_energy_J": current_E,
        "current_energy_density_J_per_g": current_density,
        "current_estimated_peak_T_K": current_T,
        "specific_heat_J_kg_K": cp,
        "specific_heat_provenance": "LITERATURE_DERIVED_ASSUMPTION",
        "electrode_sink_fraction_hypothesis": electrode_sink_fraction,
        "cases": [c.to_dict() for c in cases],
        "answer_summary": {
            "do_not_use_12_caps_for_20g": True,
            "caps_to_match_already_weak_1g": next(
                c.capacitor_count for c in cases if c.name == "match_1g_energy_density"
            ),
            "caps_for_virtual_2500K_with_rod_sink": recommended.capacitor_count,
            "stored_energy_J_at_that_count": recommended.stored_energy_at_count_J,
        },
        "safety": {
            "hardware_control_enabled": False,
            "this_is_not_a_firing_recommendation": True,
            "note": (
                f"A {recommended.stored_energy_at_count_J/1000:.1f} kJ bank is a large stored-energy "
                "hazard. 4 AWG leads, the current IGBT path, and the brown bleeder are not "
                "automatically valid at that scale. Do not treat this count as a shopping list "
                "to fire."
            ),
        },
    }
