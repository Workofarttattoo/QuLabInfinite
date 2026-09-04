"""
Physics sanity checker for FJH virtual experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import ReactorConfiguration
from .electrical import check_impossible_rectangular_pulse
from .energy import ENERGY_TOLERANCE_FRACTION, EnergyAccounting
from .types import ModelLevel, SanityStatus, is_unknown


@dataclass
class SanityCheckResult:
    """Result of physics sanity checks."""

    status: SanityStatus
    messages: list[str] = field(default_factory=list)
    failed_invariants: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "messages": self.messages,
            "failed_invariants": self.failed_invariants,
        }


def check_energy_conservation(energy: EnergyAccounting) -> tuple[bool, str]:
    if energy.initial_capacitor_energy_J <= 0:
        return False, "Initial stored energy is zero or negative"
    delivered = (
        energy.sample_energy_J
        + energy.busbar_losses_J
        + energy.switch_losses_J
        + energy.contact_losses_J
        + energy.other_losses_J
    )
    if delivered > energy.initial_capacitor_energy_J * 1.01:
        return False, (
            f"Delivered energy ({delivered:.1f} J) exceeds initial stored energy "
            f"({energy.initial_capacitor_energy_J:.1f} J)"
        )
    if not energy.is_conserved:
        return False, (
            f"Energy balance error {energy.balance_error_fraction*100:.1f}% "
            f"exceeds tolerance {ENERGY_TOLERANCE_FRACTION*100:.1f}%"
        )
    return True, "Energy conserved within tolerance"


def check_capacitor_rating(config: ReactorConfiguration) -> tuple[bool, str]:
    V = config.initial_voltage_V or config.capacitor_nominal_voltage_V
    if config.capacitor_nominal_voltage_V * 1.05 < V:
        return False, (
            f"Initial voltage {V} V exceeds nominal rating "
            f"{config.capacitor_nominal_voltage_V} V"
        )
    return True, "Voltage within capacitor rating"


def check_parameter_completeness(
    config: ReactorConfiguration,
    model_level: ModelLevel,
) -> tuple[bool, list[str]]:
    warnings = []
    critical_unknowns = []
    if model_level.value >= 2 and is_unknown(config.sample_resistance_ohm):
        if is_unknown(config.sample_resistance_vs_temperature.reference_resistance_ohm):
            warnings.append("sample_resistance unknown — using placeholder")
    if model_level.value >= 1 and is_unknown(config.measured_ESR_each_ohm):
        warnings.append("ESR unknown — using placeholder")
    return len(critical_unknowns) == 0, warnings


def check_rectangular_pulse_assumption(
    config: ReactorConfiguration,
    V: float,
    I: float,
    t_s: float,
) -> tuple[bool, str]:
    """Reject impossible constant rectangular pulse."""
    impossible, msg = check_impossible_rectangular_pulse(config, V, I, t_s)
    if impossible:
        return False, msg
    return True, "Rectangular pulse within energy budget"


def check_numerical_stability(
    max_current_A: float,
    min_voltage_V: float,
) -> tuple[bool, str]:
    if max_current_A > 1e6:
        return False, f"Peak current {max_current_A:.0f} A exceeds numerical stability bounds"
    if min_voltage_V < -1.0:
        return False, f"Negative capacitor voltage {min_voltage_V:.2f} V detected"
    return True, "Numerical stability OK"


def run_sanity_checks(
    config: ReactorConfiguration,
    energy: EnergyAccounting | None = None,
    model_level: ModelLevel = ModelLevel.LEVEL_1,
    max_current_A: float | None = None,
    min_voltage_V: float | None = None,
    rectangular_pulse: dict[str, float] | None = None,
) -> SanityCheckResult:
    """
    Run all physics sanity checks before accepting virtual experiment.
    """
    messages: list[str] = []
    failed: list[str] = []
    status = SanityStatus.VALID

    ok, msg = check_capacitor_rating(config)
    messages.append(f"capacitor_rating: {msg}")
    if not ok:
        failed.append("capacitor_rating")
        status = SanityStatus.PHYSICALLY_INVALID

    complete, warnings = check_parameter_completeness(config, model_level)
    for w in warnings:
        messages.append(f"parameter_completeness: {w}")
    if warnings and status == SanityStatus.VALID:
        status = SanityStatus.QUESTIONABLE

    if energy is not None:
        ok, msg = check_energy_conservation(energy)
        messages.append(f"energy_conservation: {msg}")
        if not ok:
            failed.append("energy_conservation")
            status = SanityStatus.PHYSICALLY_INVALID

    if rectangular_pulse:
        V = rectangular_pulse.get("V", 450)
        I = rectangular_pulse.get("I", 1000)
        t = rectangular_pulse.get("t_s", 0.005)
        ok, msg = check_rectangular_pulse_assumption(config, V, I, t)
        messages.append(f"rectangular_pulse: {msg}")
        if not ok:
            failed.append("rectangular_pulse_energy")
            status = SanityStatus.PHYSICALLY_INVALID

    if max_current_A is not None and min_voltage_V is not None:
        ok, msg = check_numerical_stability(max_current_A, min_voltage_V)
        messages.append(f"numerical_stability: {msg}")
        if not ok:
            failed.append("numerical_stability")
            status = SanityStatus.PHYSICALLY_INVALID

    if config.hardware_control_enabled:
        failed.append("hardware_control")
        messages.append(
            "hardware_control_enabled=True is forbidden in simulation-only phase"
        )
        status = SanityStatus.PHYSICALLY_INVALID

    unknowns = config.unknown_parameters()
    if len(unknowns) > 10 and status == SanityStatus.VALID:
        status = SanityStatus.INSUFFICIENT_DATA
        messages.append(f"Many unknown parameters ({len(unknowns)}); results are preliminary")

    return SanityCheckResult(status=status, messages=messages, failed_invariants=failed)
