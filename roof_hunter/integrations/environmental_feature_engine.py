from __future__ import annotations

import math
from typing import Any, Dict

from environmental_sim import EnvironmentalSimulator

from roof_hunter.models import ForecastState


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_environmental_hail_features(
    state: ForecastState,
    analysis: Dict[str, Any],
    cape_j_kg: float,
    moisture_factor: float,
) -> Dict[str, float]:
    """Compute normalized physics-inspired hail features.

    The EnvironmentalSimulator acts as a coherent state normalizer (temperature/pressure/
    wind/moisture context), while hail-specific feature transforms remain conservative and
    bounded to [0, 1] for stable online Bayesian updates.
    """
    sim = EnvironmentalSimulator(update_rate=10.0)
    sim.controller.temperature.set_temperature(state.surface_temp_c, unit="C")
    sim.controller.pressure.set_pressure(state.surface_pressure_hpa, unit="hPa")
    sim.controller.fluid.set_wind((state.wind_speed_m_s, 0.0, 0.0), unit="m/s")
    sim.controller.atmosphere.set_standard_atmosphere("air")
    sim.controller.atmosphere.set_humidity(_clamp01(state.relative_humidity) * 100.0)

    env = sim.get_conditions(position=(0.0, 0.0, 0.0))
    temp_c = float(env.get("temperature_C", state.surface_temp_c))
    pressure_hpa = float(env.get("pressure_hPa", state.surface_pressure_hpa))
    wind_m_s = float(env.get("wind_speed_m_s", state.wind_speed_m_s))
    rh_01 = _clamp01(float(env.get("relative_humidity", state.relative_humidity)))

    dewpoint_c = state.surface_dewpoint_c
    if dewpoint_c is None:
        dewpoint_c = float(analysis.get("moisture_analysis", {}).get("dewpoint_c", temp_c - 6.0))
    spread_c = max(0.0, temp_c - dewpoint_c)

    # Approximate mixed-phase hail-growth layer favorability.
    supercooled_layer_depth_km = _clamp01((18.0 - spread_c) / 18.0)

    # Updraft survival proxy: instability + moisture + organized low-level flow.
    cape_norm = _clamp01(1.0 - math.exp(-max(0.0, cape_j_kg) / 1600.0))
    shear_proxy = _clamp01((wind_m_s - 5.0) / 20.0)
    updraft_survival_index = _clamp01(0.55 * cape_norm + 0.25 * moisture_factor + 0.20 * shear_proxy)

    # Hail growth potential: supercooled depth + liquid water signal.
    pwat = float(state.precipitable_water_mm or 0.0)
    lwp_proxy = _clamp01(pwat / 50.0)
    hail_growth_potential_0_1 = _clamp01(
        0.45 * supercooled_layer_depth_km + 0.35 * lwp_proxy + 0.20 * updraft_survival_index
    )

    # Pressure-fall + moisture loading proxy for gust front / downdraft behavior.
    pressure_trend = float(state.surface_pressure_trend_hpa_per_hour or 0.0)
    pressure_fall_signal = _clamp01(max(0.0, -pressure_trend) / 1.6)
    downdraft_cooling_index = _clamp01(0.5 * pressure_fall_signal + 0.3 * rh_01 + 0.2 * _clamp01(pwat / 40.0))

    # Organization proxy: low-level flow + pressure support.
    pressure_support = _clamp01((1015.0 - pressure_hpa) / 20.0)
    storm_organization_proxy = _clamp01(0.65 * shear_proxy + 0.35 * pressure_support)

    return {
        "supercooled_layer_depth_km": round(supercooled_layer_depth_km, 6),
        "updraft_survival_index": round(updraft_survival_index, 6),
        "hail_growth_potential_0_1": round(hail_growth_potential_0_1, 6),
        "downdraft_cooling_index": round(downdraft_cooling_index, 6),
        "storm_organization_proxy": round(storm_organization_proxy, 6),
    }

