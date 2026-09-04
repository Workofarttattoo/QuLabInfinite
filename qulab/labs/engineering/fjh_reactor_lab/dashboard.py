"""
FJH Digital Twin dashboard data generator.
"""

from __future__ import annotations

from typing import Any

from .atmosphere import AtmosphereState, create_atmosphere_state
from .config import ReactorConfiguration
from .types import SimulationResult


def build_dashboard(
    config: ReactorConfiguration,
    result: SimulationResult,
    atmosphere: AtmosphereState | None = None,
    uncertainty: dict[str, Any] | None = None,
    comparison_runs: list[SimulationResult] | None = None,
    pareto_front: list[dict] | None = None,
) -> dict[str, Any]:
    """Build complete dashboard payload."""
    atm = atmosphere or create_atmosphere_state(config)

    dashboard = {
        "CAPACITOR_BANK": {
            "total_capacitance_uF": config.total_capacitance_uF(),
            "initial_voltage_V": config.initial_voltage_V or config.capacitor_nominal_voltage_V,
            "stored_energy_J": config.initial_stored_energy_J(),
            "capacitor_count": config.capacitor_count,
            "connection": config.capacitor_connection.value,
        },
        "ELECTRICAL": {
            "V_cap": {
                "time_s": result.V_cap.time_s[:50],  # subsample for display
                "values": result.V_cap.values[:50],
                "unit": "V",
            },
            "current": {
                "time_s": result.current.time_s[:50],
                "values": result.current.values[:50],
                "unit": "A",
            },
            "P_sample": {
                "time_s": result.P_sample.time_s[:50],
                "values": result.P_sample.values[:50],
                "unit": "W",
            },
            "peak_current_A": max(result.current.values) if result.current.values else 0,
            "model_level": result.model_level.name,
        },
        "THERMAL": _thermal_section(result),
        "ENERGY_ACCOUNTING": {
            "initial_stored_energy_J": result.energy.initial_capacitor_energy_J,
            "sample_energy_J": result.energy.sample_energy_J,
            "switch_losses_J": result.energy.switch_losses_J,
            "bus_losses_J": result.energy.busbar_losses_J,
            "contact_losses_J": result.energy.contact_losses_J,
            "remaining_capacitor_energy_J": result.energy.remaining_capacitor_energy_J,
            "is_conserved": bool(result.energy.is_conserved),
            "balance_error_fraction": result.energy.balance_error_fraction,
        },
        "ATMOSPHERE": {
            "type": atm.atmosphere_type.value,
            "pressure_Pa": (
                float(atm.initial_pressure_Pa)
                if not isinstance(atm.initial_pressure_Pa, type(None))
                and hasattr(atm.initial_pressure_Pa, "__float__")
                else "UNKNOWN"
            ),
            "gas_composition": atm.gas_composition.primary_gas,
            "residual_oxygen_assumption": (
                float(atm.residual_oxygen_fraction)
                if hasattr(atm.residual_oxygen_fraction, "__float__")
                and not str(atm.residual_oxygen_fraction).startswith("UNKNOWN")
                else "UNKNOWN"
            ),
            "modeled_effects": atm.modeled_effects,
            "placeholder_effects": atm.placeholder_effects,
        },
        "MATERIAL_HYPOTHESIS": (
            result.hypothesis_scores.to_dict()
            if result.hypothesis_scores
            else {"label": "HYPOTHESIS SCORES — not run"}
        ),
        "UNCERTAINTY": uncertainty or result.uncertainty or {"status": "not computed"},
        "SANITY": {
            "status": result.sanity_status.value,
            "messages": result.sanity_messages,
        },
        "EXPERIMENT": {
            "experiment_id": result.experiment_id,
            "hardware_control_enabled": config.hardware_control_enabled,
            "simulation_only_phase": True,
        },
    }

    if comparison_runs:
        dashboard["EXPERIMENT_COMPARISON"] = [
            {
                "experiment_id": r.experiment_id,
                "peak_current_A": max(r.current.values) if r.current.values else 0,
                "delivered_energy_J": r.energy.sample_energy_J,
                "sanity_status": r.sanity_status.value,
                "hypothesis_scores": (
                    r.hypothesis_scores.to_dict() if r.hypothesis_scores else None
                ),
            }
            for r in comparison_runs
        ]

    if pareto_front:
        dashboard["PARETO_FRONT"] = pareto_front

    return dashboard


def _thermal_section(result: SimulationResult) -> dict[str, Any]:
    if result.T_sample is None:
        return {"status": "thermal model not run"}
    return {
        "T_sample": {
            "time_s": result.T_sample.time_s[:50],
            "values": result.T_sample.values[:50],
            "unit": "K",
        },
        "peak_temperature_K": max(result.T_sample.values) if result.T_sample.values else 0,
        "provenance": "SIMULATED_RESULT",
    }
