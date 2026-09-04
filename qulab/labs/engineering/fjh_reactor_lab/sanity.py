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


def check_igbt_voltage(config: ReactorConfiguration) -> tuple[bool, str]:
    """450 V bank vs Infineon 600 V IGBT: voltage-legal, current still unknown."""
    V = config.initial_voltage_V or config.capacitor_nominal_voltage_V
    v_igbt = config.igbt.voltage_rating_V
    if V > v_igbt:
        return False, (
            f"Bank voltage {V} V exceeds IGBT rating {v_igbt} V"
        )
    headroom = v_igbt - V
    return True, (
        f"IGBT {config.igbt.manufacturer} {v_igbt:.0f} V rating covers "
        f"{V:.0f} V bank ({headroom:.0f} V label headroom). "
        "Pulse current rating UNKNOWN; inductive spikes not modeled."
    )


def check_igbt_current_unknown(
    config: ReactorConfiguration,
    peak_current_A: float | None = None,
) -> tuple[bool, str]:
    """600 V nameplate is not a current rating. Unknown Ic stays QUESTIONABLE."""
    current_unknown = is_unknown(config.igbt.current_rating_A) and is_unknown(
        config.igbt.pulse_current_rating_A
    )
    if not current_unknown:
        return True, "IGBT current rating provided"
    peak_note = ""
    if peak_current_A is not None:
        peak_note = (
            f" Simulated peak {peak_current_A:.0f} A cannot be compared to an "
            "UNKNOWN Infineon current rating."
        )
    return False, (
        "Infineon IGBT current and pulse ratings are UNKNOWN. "
        "600 V vs a 450 V bank is voltage-legal only."
        f"{peak_note} Do not fire."
    )


def check_nonflash_dump_bank(config: ReactorConfiguration) -> tuple[bool, str]:
    """Refuse operator-stated non-flash electrolytics as the dump bank."""
    if config.uses_nonflash_electrolytic_dump:
        return False, (
            "JCCON CD136 10×4700 µF / 450 V are operator-stated not flash-rated. "
            "They must not be used as the FJH dump bank. Do not fire."
        )
    return True, (
        "Dump path is the 12×900 µF flash bank. "
        "Side JCCON 10×4700 µF inventory is not connected."
    )


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
    peak_temperature_K: float | None = None,
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

    ok, msg = check_igbt_voltage(config)
    messages.append(f"igbt_voltage: {msg}")
    if not ok:
        failed.append("igbt_voltage")
        status = SanityStatus.PHYSICALLY_INVALID

    ok, msg = check_igbt_current_unknown(config, max_current_A)
    messages.append(f"igbt_current: {msg}")
    if not ok and status == SanityStatus.VALID:
        status = SanityStatus.QUESTIONABLE

    ok, msg = check_nonflash_dump_bank(config)
    messages.append(f"nonflash_electrolytic_dump: {msg}")
    if not ok:
        failed.append("nonflash_electrolytic_dump")
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

    if peak_temperature_K is not None and peak_temperature_K > 3500:
        messages.append(
            f"model_domain: peak T {peak_temperature_K:.0f} K exceeds carbon "
            "sublimation-scale (~3900 K) / lumped-model validity. "
            "Radiation and vaporization are not modeled."
        )
        failed.append("thermal_model_domain")
        if status == SanityStatus.VALID:
            status = SanityStatus.QUESTIONABLE

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
