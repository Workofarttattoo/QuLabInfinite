"""
Energy conservation accounting for FJH reactor simulations.
"""

from __future__ import annotations

import numpy as np

from .config import ReactorConfiguration
from .types import EnergyAccounting, ModelLevel, TimeSeries, is_unknown

ENERGY_TOLERANCE_FRACTION = 0.05  # 5% numerical tolerance


def integrate_power(time_s: list[float], power_W: list[float]) -> float:
    """Trapezoidal integration of power over time -> energy in J."""
    if len(time_s) < 2:
        return 0.0
    return float(np.trapezoid(power_W, time_s))


def capacitor_energy_J(C_F: float, V_V: float) -> float:
    return 0.5 * C_F * V_V ** 2


def compute_energy_accounting(
    config: ReactorConfiguration,
    electrical: dict,
    model_level: ModelLevel,
    contact_resistance_ohm: float | None = None,
) -> EnergyAccounting:
    """
    Compute full energy balance from electrical simulation results.

    initial_energy = remaining + sample + bus + switch + contact + other
    """
    C = config.total_capacitance_F()
    V0 = config.initial_voltage_V or config.capacitor_nominal_voltage_V
    E_initial = 0.5 * C * V0 ** 2

    V_cap_ts: TimeSeries = electrical["V_cap"]
    I_ts: TimeSeries = electrical["current"]
    P_sample_ts: TimeSeries = electrical["P_sample"]

    V_final = V_cap_ts.values[-1] if V_cap_ts.values else 0.0
    E_remaining = 0.5 * C * V_final ** 2

    E_sample = integrate_power(P_sample_ts.time_s, P_sample_ts.values)

    # Loss partitioning from I^2*R components
    t = np.array(I_ts.time_s)
    I = np.array(I_ts.values)
    I2 = I ** 2

    def _loss_energy(R_ohm: float) -> float:
        if R_ohm <= 0 or len(t) < 2:
            return 0.0
        P_loss = I2 * R_ohm
        return float(np.trapezoid(P_loss, t))

    E_esr = _loss_energy(config.effective_ESR_ohm())
    E_bus = 0.0
    if not is_unknown(config.busbar_resistance_ohm):
        E_bus = _loss_energy(float(config.busbar_resistance_ohm))
    E_connection = 0.0
    if not is_unknown(config.connection_resistance_ohm):
        E_connection = _loss_energy(float(config.connection_resistance_ohm))

    R_contact = contact_resistance_ohm
    if R_contact is None and not is_unknown(config.electrode_contact_resistance_ohm):
        R_contact = float(config.electrode_contact_resistance_ohm)
    E_contact = _loss_energy(R_contact or 0.0)

    E_switch = 0.0
    if not is_unknown(config.igbt.on_resistance_ohm):
        E_switch = _loss_energy(float(config.igbt.on_resistance_ohm))
    if not is_unknown(config.igbt.switching_energy_J):
        E_switch += float(config.igbt.switching_energy_J)

    E_other = E_connection
    E_accounted = E_remaining + E_sample + E_esr + E_bus + E_switch + E_contact + E_other
    balance_error = E_initial - E_accounted
    balance_fraction = abs(balance_error) / E_initial if E_initial > 0 else 0.0
    is_conserved = balance_fraction <= ENERGY_TOLERANCE_FRACTION

    return EnergyAccounting(
        initial_capacitor_energy_J=E_initial,
        remaining_capacitor_energy_J=E_remaining,
        sample_energy_J=E_sample,
        busbar_losses_J=E_esr + E_bus,
        switch_losses_J=E_switch,
        contact_losses_J=E_contact,
        other_losses_J=E_other,
        balance_error_J=balance_error,
        balance_error_fraction=balance_fraction,
        is_conserved=is_conserved,
        model_level=model_level,
    )
