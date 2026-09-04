"""
Thermal response model driven by P_sample(t).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ReactorConfiguration
from .types import UNKNOWN, TimeSeries, UnknownValue, is_unknown


@dataclass
class MaterialThermalProperties:
    """Parameterized thermal/electrical material properties."""

    specific_heat_J_kg_K: float | UnknownValue = UNKNOWN
    thermal_conductivity_W_m_K: float | UnknownValue = UNKNOWN
    density_kg_m3: float | UnknownValue = UNKNOWN
    electrical_resistivity_ohm_m: float | UnknownValue = UNKNOWN
    temperature_coefficient_resistivity: float | UnknownValue = UNKNOWN
    uncertainty_fraction: float = 0.2  # literature uncertainty range


@dataclass
class ThermalResult:
    """Thermal simulation output."""

    T_sample: TimeSeries
    heating_rate_K_s: TimeSeries
    cooling_rate_K_s: TimeSeries
    peak_temperature_K: float
    time_above_thresholds: dict[str, float]
    energy_absorbed_sample_J: float
    energy_lost_electrodes_J: float
    energy_lost_chamber_J: float
    thermal_gradient_estimate_K: float | UnknownValue = UNKNOWN


# Literature-order-of-magnitude defaults for Vulcan XC-72 carbon (HYPOTHESIS)
VULCAN_XC72_THERMAL = MaterialThermalProperties(
    specific_heat_J_kg_K=710.0,
    thermal_conductivity_W_m_K=0.15,
    density_kg_m3=180.0,
    electrical_resistivity_ohm_m=0.01,
    temperature_coefficient_resistivity=0.002,
    uncertainty_fraction=0.3,
)


def _effective_cp(props: MaterialThermalProperties) -> float:
    if not is_unknown(props.specific_heat_J_kg_K):
        return float(props.specific_heat_J_kg_K)
    return 710.0  # placeholder with note


def _effective_mass_kg(config: ReactorConfiguration) -> float:
    if not is_unknown(config.sample_mass_g):
        return float(config.sample_mass_g) / 1000.0
    return 0.001  # 1 mg placeholder


def simulate_thermal_lumped(
    config: ReactorConfiguration,
    P_sample: TimeSeries,
    material: MaterialThermalProperties | None = None,
    electrode_coupling_fraction: float = 0.15,
    chamber_coupling_fraction: float = 0.05,
    cooling_tau_s: float = 0.5,
) -> ThermalResult:
    """
    Lumped thermal model: m*cp*dT/dt = P_in - P_out
    P_out includes electrode and chamber heat sinks.
    """
    props = material or VULCAN_XC72_THERMAL
    cp = _effective_cp(props)
    mass = _effective_mass_kg(config)
    T0 = config.initial_sample_temperature_K

    t = np.array(P_sample.time_s)
    P = np.array(P_sample.values)
    n = len(t)
    if n < 2:
        T = np.full(n, T0)
        return ThermalResult(
            T_sample=TimeSeries(list(t), list(T), "K", "T_sample(t)"),
            heating_rate_K_s=TimeSeries(list(t), [0.0] * n, "K/s", "dT/dt"),
            cooling_rate_K_s=TimeSeries(list(t), [0.0] * n, "K/s", "cooling_rate"),
            peak_temperature_K=T0,
            time_above_thresholds={},
            energy_absorbed_sample_J=0.0,
            energy_lost_electrodes_J=0.0,
            energy_lost_chamber_J=0.0,
        )

    dt = t[1] - t[0]
    T = np.zeros(n)
    T[0] = T0
    dTdt = np.zeros(n)

    P_to_sample = P * (1.0 - electrode_coupling_fraction - chamber_coupling_fraction)
    P_electrodes = P * electrode_coupling_fraction
    P_chamber = P * chamber_coupling_fraction

    for k in range(1, n):
        P_net = P_to_sample[k - 1] - (T[k - 1] - config.ambient_temperature_K) / cooling_tau_s * cp * mass
        dTdt[k] = P_net / (mass * cp) if mass * cp > 0 else 0.0
        T[k] = T[k - 1] + dTdt[k] * dt

    heating = np.maximum(dTdt, 0.0)
    cooling = np.maximum(-dTdt, 0.0)

    thresholds = {"500K": 500, "1000K": 1000, "2000K": 2000, "3000K": 3000}
    time_above = {}
    for label, T_thresh in thresholds.items():
        above = T_thresh <= T
        time_above[label] = float(np.sum(above) * dt) if np.any(above) else 0.0

    E_sample = float(np.trapezoid(P_to_sample, t))
    E_electrodes = float(np.trapezoid(P_electrodes, t))
    E_chamber = float(np.trapezoid(P_chamber, t))

    grad = UNKNOWN
    if not is_unknown(props.thermal_conductivity_W_m_K):
        k_th = float(props.thermal_conductivity_W_m_K)
        peak_dT = float(np.max(T) - T0)
        # crude gradient estimate: delta_T over ~1mm characteristic length
        grad = peak_dT / 0.001 * 0.5  # scaled estimate

    return ThermalResult(
        T_sample=TimeSeries(list(t), list(T), "K", "T_sample(t)"),
        heating_rate_K_s=TimeSeries(list(t), list(heating), "K/s", "heating_rate"),
        cooling_rate_K_s=TimeSeries(list(t), list(cooling), "K/s", "cooling_rate"),
        peak_temperature_K=float(np.max(T)),
        time_above_thresholds=time_above,
        energy_absorbed_sample_J=E_sample,
        energy_lost_electrodes_J=E_electrodes,
        energy_lost_chamber_J=E_chamber,
        thermal_gradient_estimate_K=grad,
    )
