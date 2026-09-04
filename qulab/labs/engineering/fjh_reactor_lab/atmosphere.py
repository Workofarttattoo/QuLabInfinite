"""
Atmosphere model for FJH reactor chamber.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import GasComposition, ReactorConfiguration
from .types import AtmosphereType, UnknownValue, is_unknown


@dataclass
class AtmosphereState:
    """Virtual atmosphere state."""

    atmosphere_type: AtmosphereType
    initial_pressure_Pa: float | UnknownValue
    gas_composition: GasComposition
    temperature_K: float
    residual_oxygen_fraction: float | UnknownValue
    pressure_evolution_modeled: bool = False
    modeled_effects: list[str] = field(default_factory=list)
    placeholder_effects: list[str] = field(default_factory=list)


def create_atmosphere_state(config: ReactorConfiguration) -> AtmosphereState:
    """Build atmosphere state from reactor config."""
    modeled: list[str] = []
    placeholders: list[str] = []

    if config.atmosphere_type == AtmosphereType.VACUUM:
        modeled.append("reduced_convective_heat_transfer")
        placeholders.extend([
            "residual_gas_composition_not_auto_zero",
            "outgassing_not_modeled",
        ])
    elif config.atmosphere_type == AtmosphereType.ARGON:
        modeled.append("inert_atmosphere_assumption")
        placeholders.extend([
            "residual_oxygen_not_auto_zero",
            "pressure_evolution_during_heating_not_modeled",
            "gas_thermal_conductivity_effect_placeholder",
        ])
    else:
        placeholders.append("user_defined_mixture_effects_partial")

    residual = config.gas_composition.residual_oxygen_fraction
    if is_unknown(residual):
        placeholders.append("residual_oxygen_fraction_unknown")

    return AtmosphereState(
        atmosphere_type=config.atmosphere_type,
        initial_pressure_Pa=config.chamber_pressure_Pa,
        gas_composition=config.gas_composition,
        temperature_K=config.ambient_temperature_K,
        residual_oxygen_fraction=residual,
        pressure_evolution_modeled=False,
        modeled_effects=modeled,
        placeholder_effects=placeholders,
    )


def compare_atmospheres(
    config_vacuum: ReactorConfiguration,
    config_argon: ReactorConfiguration,
) -> dict[str, Any]:
    """Compare vacuum vs argon atmosphere configurations."""
    vac = create_atmosphere_state(config_vacuum)
    ar = create_atmosphere_state(config_argon)
    return {
        "vacuum": {
            "type": vac.atmosphere_type.value,
            "modeled_effects": vac.modeled_effects,
            "placeholder_effects": vac.placeholder_effects,
            "residual_oxygen": (
                float(vac.residual_oxygen_fraction)
                if not is_unknown(vac.residual_oxygen_fraction)
                else "UNKNOWN"
            ),
        },
        "argon": {
            "type": ar.atmosphere_type.value,
            "modeled_effects": ar.modeled_effects,
            "placeholder_effects": ar.placeholder_effects,
            "residual_oxygen": (
                float(ar.residual_oxygen_fraction)
                if not is_unknown(ar.residual_oxygen_fraction)
                else "UNKNOWN"
            ),
        },
        "note": (
            "Vacuum vs argon comparison identifies which effects are modeled "
            "vs placeholders. Residual oxygen is NOT auto-zero in either case."
        ),
    }
