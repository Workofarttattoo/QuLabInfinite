"""
Transient electrical model for FJH capacitor discharge.

Model levels:
  LEVEL 0: Idealized RC sanity model
  LEVEL 1: RLC lumped circuit
  LEVEL 2: Temperature-dependent resistance
  LEVEL 3: Coupled electrical/thermal
  LEVEL 4: Placeholder for future multiphysics
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .config import ReactorConfiguration
from .types import ModelLevel, TimeSeries, is_unknown


def _total_series_resistance(config: ReactorConfiguration) -> float:
    """Sum resistances in discharge path."""
    r = config.effective_ESR_ohm()
    for attr in (
        "busbar_resistance_ohm",
        "connection_resistance_ohm",
        "electrode_contact_resistance_ohm",
    ):
        val = getattr(config, attr)
        if not is_unknown(val):
            r += float(val)
    if not is_unknown(config.igbt.on_resistance_ohm):
        r += float(config.igbt.on_resistance_ohm)
    return r


def _total_inductance(config: ReactorConfiguration) -> float:
    L = 0.0
    if not is_unknown(config.measured_ESL_H):
        L += float(config.measured_ESL_H)
    if not is_unknown(config.busbar_inductance_H):
        L += float(config.busbar_inductance_H)
    return L


def simulate_level0_rc(
    config: ReactorConfiguration,
    dt: float = 1e-6,
    duration_s: float | None = None,
) -> dict[str, TimeSeries]:
    """
    LEVEL 0: Idealized RC discharge.
    V(t) = V0 * exp(-t / (R*C))
    I(t) = (V0/R) * exp(-t / (R*C))
    """
    C = config.total_capacitance_F()
    V0 = config.initial_voltage_V or config.capacitor_nominal_voltage_V
    R_sample = config.effective_sample_resistance_ohm()
    R_total = _total_series_resistance(config) + R_sample
    tau = R_total * C

    duration = duration_s or config.pulse_duration_s
    n_steps = max(int(duration / dt), 10)
    t = np.linspace(0, duration, n_steps)

    V_cap = V0 * np.exp(-t / tau)
    I = (V0 / R_total) * np.exp(-t / tau)
    V_sample = I * R_sample
    P_sample = V_sample * I

    return {
        "V_cap": TimeSeries(list(t), list(V_cap), "V", "V_cap(t)"),
        "V_sample": TimeSeries(list(t), list(V_sample), "V", "V_sample(t)"),
        "current": TimeSeries(list(t), list(I), "A", "I(t)"),
        "P_sample": TimeSeries(list(t), list(P_sample), "W", "P_sample(t)"),
        "model_level": ModelLevel.LEVEL_0,
        "R_total_ohm": R_total,
        "tau_s": tau,
    }


def simulate_level1_rlc(
    config: ReactorConfiguration,
    dt: float = 1e-7,
    duration_s: float | None = None,
) -> dict[str, TimeSeries]:
    """
    LEVEL 1: RLC lumped circuit via numerical integration.
    L * dI/dt + R * I + V_c = 0
    dV_c/dt = -I/C
    """
    C = config.total_capacitance_F()
    V0 = config.initial_voltage_V or config.capacitor_nominal_voltage_V
    R_sample = config.effective_sample_resistance_ohm()
    R = _total_series_resistance(config) + R_sample
    L = _total_inductance(config)

    duration = duration_s or config.pulse_duration_s
    n_steps = max(int(duration / dt), 100)

    V_cap = np.zeros(n_steps)
    I = np.zeros(n_steps)
    t = np.zeros(n_steps)

    V_cap[0] = V0
    I[0] = 0.0
    t[0] = 0.0

    for k in range(1, n_steps):
        t[k] = k * dt
        if L > 1e-12:
            dI = (V_cap[k - 1] - R * I[k - 1]) / L
            I[k] = I[k - 1] + dI * dt
        else:
            # Degenerate to RC if no inductance
            I[k] = V_cap[k - 1] / R if R > 0 else 0.0
        dV = -I[k] / C
        V_cap[k] = V_cap[k - 1] + dV * dt
        V_cap[k] = max(V_cap[k], 0.0)

    V_sample = I * R_sample
    P_sample = V_sample * I

    return {
        "V_cap": TimeSeries(list(t), list(V_cap), "V", "V_cap(t)"),
        "V_sample": TimeSeries(list(t), list(V_sample), "V", "V_sample(t)"),
        "current": TimeSeries(list(t), list(I), "A", "I(t)"),
        "P_sample": TimeSeries(list(t), list(P_sample), "W", "P_sample(t)"),
        "model_level": ModelLevel.LEVEL_1,
        "R_total_ohm": R,
        "L_total_H": L,
    }


def simulate_level2_temp_dependent(
    config: ReactorConfiguration,
    dt: float = 1e-7,
    duration_s: float | None = None,
    T_sample_func: Callable[[float], float] | None = None,
) -> dict[str, TimeSeries]:
    """
    LEVEL 2: Temperature-dependent sample resistance during discharge.
    R_sample(T) updated each timestep from thermal feedback if provided.
    """
    C = config.total_capacitance_F()
    V0 = config.initial_voltage_V or config.capacitor_nominal_voltage_V
    R_fixed = _total_series_resistance(config)
    L = _total_inductance(config)
    T0 = config.initial_sample_temperature_K

    duration = duration_s or config.pulse_duration_s
    n_steps = max(int(duration / dt), 100)

    V_cap = np.zeros(n_steps)
    I = np.zeros(n_steps)
    R_sample = np.zeros(n_steps)
    t = np.zeros(n_steps)

    V_cap[0] = V0
    I[0] = 0.0
    R_sample[0] = config.effective_sample_resistance_ohm(T0)
    t[0] = 0.0

    for k in range(1, n_steps):
        t[k] = k * dt
        T = T0
        if T_sample_func is not None:
            T = T_sample_func(t[k - 1])
        R_sample[k] = config.effective_sample_resistance_ohm(T)
        R_total = R_fixed + R_sample[k]

        if L > 1e-12:
            dI = (V_cap[k - 1] - R_total * I[k - 1]) / L
            I[k] = I[k - 1] + dI * dt
        else:
            I[k] = V_cap[k - 1] / R_total if R_total > 0 else 0.0
        dV = -I[k] / C
        V_cap[k] = max(V_cap[k - 1] + dV * dt, 0.0)

    V_sample = I * R_sample
    P_sample = V_sample * I

    return {
        "V_cap": TimeSeries(list(t), list(V_cap), "V", "V_cap(t)"),
        "V_sample": TimeSeries(list(t), list(V_sample), "V", "V_sample(t)"),
        "current": TimeSeries(list(t), list(I), "A", "I(t)"),
        "P_sample": TimeSeries(list(t), list(P_sample), "W", "P_sample(t)"),
        "R_sample": TimeSeries(list(t), list(R_sample), "ohm", "R_sample(t)"),
        "model_level": ModelLevel.LEVEL_2,
    }


def simulate_electrical(
    config: ReactorConfiguration,
    model_level: ModelLevel = ModelLevel.LEVEL_1,
    dt: float = 1e-7,
    duration_s: float | None = None,
    T_sample_func: Callable[[float], float] | None = None,
) -> dict:
    """Dispatch to appropriate electrical model level."""
    if model_level == ModelLevel.LEVEL_0:
        return simulate_level0_rc(config, dt=dt, duration_s=duration_s)
    if model_level == ModelLevel.LEVEL_1:
        return simulate_level1_rlc(config, dt=dt, duration_s=duration_s)
    if model_level in (ModelLevel.LEVEL_2, ModelLevel.LEVEL_3):
        return simulate_level2_temp_dependent(
            config, dt=dt, duration_s=duration_s, T_sample_func=T_sample_func
        )
    if model_level == ModelLevel.LEVEL_4:
        return {
            "error": "LEVEL_4 spatial/multiphysics model not yet implemented",
            "model_level": ModelLevel.LEVEL_4,
            "placeholder": True,
        }
    return simulate_level1_rlc(config, dt=dt, duration_s=duration_s)


def rectangular_pulse_energy_J(V: float, I: float, t_s: float) -> float:
    """Energy for hypothetical constant rectangular pulse: E = V * I * t."""
    return V * I * t_s


def check_impossible_rectangular_pulse(
    config: ReactorConfiguration,
    V: float,
    I: float,
    t_s: float,
) -> tuple[bool, str]:
    """
    Check if constant V/I/t pulse exceeds stored capacitor energy.
    Returns (is_impossible, explanation).
    """
    E_pulse = rectangular_pulse_energy_J(V, I, t_s)
    E_bank = config.initial_stored_energy_J()
    if E_pulse > E_bank * 1.01:  # 1% tolerance
        return True, (
            f"Rectangular pulse {V}V × {I}A × {t_s*1000:.1f}ms requires "
            f"{E_pulse:.1f} J but bank stores only {E_bank:.1f} J. "
            f"Physically impossible without external energy source."
        )
    return False, ""
