"""QuLab proprietary mesoscale surface refinement (CorrDiff-class behavior, no diffusion checkpoint).

Implements a deterministic **pseudo downscale** to emulate 1–3 km cell corrections from a coarser
parent column using:
- mass-consistent dewpoint retuning after temperature nudges (constant mixing ratio below lifting),
- low-pass / high-pass split on a synthetic coarse residual using wind-aligned micro-advection,
- optional terrain slope proxy to nudge temperature with a dry-adiabatic lapse fraction.

This is not a generative super-resolution model; it is fast, auditable, and stable for lead scoring.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from roof_hunter.models import ForecastState


def _saturation_vapor_pressure_hpa(t_c: float) -> float:
    return 6.112 * math.exp((17.67 * t_c) / (t_c + 243.5))


def _dewpoint_from_rh(t_c: float, rh_0_1: float) -> float:
    rh = max(0.05, min(0.999, rh_0_1))
    es = _saturation_vapor_pressure_hpa(t_c)
    e = rh * es
    b = math.log(e / 6.112)
    return float((243.5 * b) / (17.67 - b))


def _rh_from_td(t_c: float, td_c: float) -> float:
    e = _saturation_vapor_pressure_hpa(td_c)
    es = _saturation_vapor_pressure_hpa(t_c)
    return float(max(0.05, min(0.999, e / max(1e-6, es))))


def refine_surface_dict(
    coarse: Dict[str, Any],
    *,
    coarse_cell_m: float = 3000.0,
    target_cell_m: float = 2000.0,
    dt_minutes: float = 10.0,
    terrain_slope_proxy_0_1: float = 0.0,
    residual_t_c: float = 0.0,
) -> Dict[str, Any]:
    """Return refined surface fields (dict keys compatible with ``ForecastState``).

    ``residual_t_c`` is the sub-grid temperature anomaly inferred upstream (e.g. from radar cold
    pool); if zero, a small wind-driven residual is synthesized from horizontal heterogeneity proxy.
    """
    t = float(coarse.get("surface_temp_c", 20.0))
    rh = float(coarse.get("relative_humidity", 0.65))
    td = coarse.get("surface_dewpoint_c")
    if td is None:
        td = _dewpoint_from_rh(t, rh)
    else:
        td = float(td)
    wind = float(coarse.get("wind_speed_m_s", 0.0))
    wind_dir = float(coarse.get("wind_direction_deg", 0.0))
    p = float(coarse.get("surface_pressure_hpa", 1013.25))

    scale = max(0.5, min(2.5, coarse_cell_m / max(500.0, target_cell_m)))
    # Sub-grid sharpening fraction (smaller target cell → slightly stronger mesoscale adjustment)
    alpha = 0.08 + 0.06 * (scale - 1.0)
    alpha = max(0.04, min(0.22, alpha))

    rad = math.radians(wind_dir)
    u = -wind * math.sin(rad)
    v = -wind * math.cos(rad)
    # Synthetic along-wind gradient proxy (K) across parent cell
    grad_k = min(1.8, 0.11 * math.hypot(u, v) * (coarse_cell_m / 3000.0))
    syn_residual = residual_t_c if abs(residual_t_c) > 1e-6 else grad_k * math.copysign(1.0, u + 0.1 * v)

    dt_h = max(1e-3, dt_minutes / 60.0)
    advect_k = 0.045 * math.hypot(u, v) * dt_h / (coarse_cell_m / 3000.0)
    delta_t = alpha * syn_residual - advect_k * math.tanh(syn_residual)
    # Terrain: upslope cooling proxy
    delta_t -= 1.1 * max(0.0, min(1.0, terrain_slope_proxy_0_1))

    t2 = t + delta_t
    # Hold mixing ratio approximately fixed when retuning dewpoint after T nudge
    es_old = _saturation_vapor_pressure_hpa(t)
    denom = max(1e-3, p - rh * es_old)
    w = 0.622 * (rh * es_old) / denom
    e_new = w * p / (0.622 + w)
    td2 = float((243.5 * math.log(e_new / 6.112)) / (17.67 - math.log(e_new / 6.112)))
    rh2 = _rh_from_td(t2, td2)

    out = dict(coarse)
    out["surface_temp_c"] = round(t2, 3)
    out["surface_dewpoint_c"] = round(td2, 3)
    out["relative_humidity"] = round(rh2, 4)
    out["mesoscale_refined"] = True
    out["corrdiff_refined"] = True
    return out


def apply_mesoscale_patch_to_state(
    state: ForecastState,
    patch: Dict[str, Any],
    *,
    only_if_flag: bool = True,
) -> ForecastState:
    """Apply refined surface fields from ``refine_surface_dict`` or explicit overrides."""
    if only_if_flag and not patch.get("apply"):
        return state
    data = state.to_dict()
    if patch.get("coarse_context"):
        merged = refine_surface_dict({**data, **patch["coarse_context"]}, **patch.get("refine_kw", {}))
        for k in ("surface_temp_c", "relative_humidity", "surface_pressure_hpa", "surface_dewpoint_c"):
            if k in merged:
                data[k] = merged[k]
        data["mesoscale_refined"] = True
        data["corrdiff_refined"] = True
    for key in (
        "surface_temp_c",
        "relative_humidity",
        "surface_pressure_hpa",
        "surface_dewpoint_c",
        "wind_speed_m_s",
        "wind_direction_deg",
        "precip_mm",
    ):
        if key in patch and patch[key] is not None:
            data[key] = patch[key]
    if patch.get("mesoscale_refined") is not None:
        data["mesoscale_refined"] = bool(patch["mesoscale_refined"])
    if patch.get("corrdiff_refined") is not None:
        data["corrdiff_refined"] = bool(patch["corrdiff_refined"])
    return ForecastState.from_dict(data)


def apply_corrdiff_patch_to_state(
    state: ForecastState,
    patch: Dict[str, Any],
    *,
    only_if_flag: bool = True,
) -> ForecastState:
    """Backward-compatible alias for ``apply_mesoscale_patch_to_state``."""
    return apply_mesoscale_patch_to_state(state, patch, only_if_flag=only_if_flag)
