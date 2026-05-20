"""QuLab proprietary global hail-environment outlook (replaces external global NWP bridges).

Design goals vs. a black-box global neural model:
- Transparent thermodynamic pipeline: mixed-layer CAPE scaling, heuristic CIN from low-level
  thermodynamic geometry, precipitable-water depth, and bulk shear proxy.
- Uncertainty bracket via spread–moisture consistency (``qu_outlook_uncertainty_0_1``).
- No external model checkpoints; uses the same parcel physics as ``AtmosphericScienceLab`` where
  available, with conservative MLCAPE-style uplift for severe-hail regimes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from atmospheric_science_lab.atmospheric_science_lab import AtmosphericScienceLab


def _heuristic_cin_j_kg(
    t_c: float,
    td_c: float,
    rh: float,
    p_hpa: float,
    precip_mm: float,
) -> float:
    """Synthetic CIN scale (J/kg) from near-surface stability / moisture decoupling proxies.

    This is not a lifted-parcel integral; it ranks *relative* cap strength for outlook sorting
    alongside CAPE (weak cap → low CIN magnitude in this scale).
    """
    spread = max(0.0, t_c - td_c)
    # Warm, moist, low-spread → small cap
    base = 12.0 + 4.2 * spread + 35.0 * max(0.0, rh - 0.72)
    # Dry slot aloft proxy: big spread + lower RH increases cap
    dry_slot = max(0.0, spread - 10.0) * (1.0 - rh) * 18.0
    # Ongoing precip reduces inferred cap (mesoscale lifting already eroding inhibition)
    lift = min(55.0, precip_mm * 6.5)
    cin = max(5.0, min(320.0, base + dry_slot - lift))
    # Surface pressure anomaly proxy: lower pressure → slightly less inferred lid
    p_norm = (p_hpa - 1013.25) / 18.0
    cin = cin * (1.0 - 0.06 * math.tanh(p_norm))
    return float(cin)


def _outlook_uncertainty(spread: float, pw_mm: Optional[float], wind_ms: float) -> float:
    """0..1: higher when thermodynamic inputs disagree with typical severe profiles."""
    u = 0.33 * min(1.0, abs(spread - 11.0) / 14.0)
    if pw_mm is not None:
        u += 0.28 * (1.0 - min(1.0, pw_mm / 55.0))  # very dry column for warm sector
    u += 0.22 * max(0.0, 1.0 - wind_ms / 18.0)  # weak flow vs hail climatology
    return float(min(1.0, u))


def _hail_environment_index(cape: float, cin: float, shear_factor: float, pw_mm: Optional[float]) -> float:
    """0..1 outlook index: CAPE dominance with CIN lid and shear organization."""
    cape_term = min(1.0, cape / 2400.0) ** 0.92
    cin_term = min(1.0, max(0.0, 1.0 - cin / 140.0))
    pw_term = min(1.0, (pw_mm or 32.0) / 52.0) ** 0.55 if pw_mm is not None else 0.62
    org = 0.38 + 0.62 * shear_factor
    raw = 0.48 * cape_term + 0.30 * cin_term + 0.14 * pw_term + 0.08 * org
    return float(min(1.0, max(0.0, round(raw, 4))))


def estimate_outlook_for_row(row: Dict[str, Any], lab: Optional[AtmosphericScienceLab] = None) -> Dict[str, float]:
    """Compute QuLab global outlook scalars from a single forecast row dict."""
    lab = lab or AtmosphericScienceLab()
    t = float(row.get("surface_temp_c", 20.0))
    rh = float(row.get("relative_humidity", 0.65))
    p = float(row.get("surface_pressure_hpa", 1013.25))
    td = row.get("surface_dewpoint_c")
    if td is None:
        td = float(lab.run_weather_forecast_analysis(0, t, rh, p)["moisture_analysis"]["dewpoint_c"])
    else:
        td = float(td)
    pw = row.get("precipitable_water_mm")
    pw_f = float(pw) if pw is not None else None
    wind = float(row.get("wind_speed_m_s", 0.0))
    precip = float(row.get("precip_mm", 0.0))

    raw_cape = float(
        lab.weather.predict_convective_available_potential_energy(t, td, p)["cape_j_kg"]
    )
    spread = max(0.0, t - td)
    ml_scale = 1.18 + 0.48 * min(1.0, spread / 16.0)
    pw_boost = 1.0 + 0.28 * min(1.0, (pw_f or 28.0) / 48.0)
    shear_boost = 1.0 + 0.11 * min(1.0, max(0.0, wind - 6.0) / 24.0)
    ml_cape = min(5800.0, raw_cape * ml_scale * pw_boost * shear_boost)

    cin = _heuristic_cin_j_kg(t, td, rh, p, precip)
    shear_f = min(1.0, max(0.0, wind - 3.0) / 26.0)
    idx = _hail_environment_index(ml_cape, cin, shear_f, pw_f)
    unc = _outlook_uncertainty(spread, pw_f, wind)
    return {
        "qu_global_cape_j_kg": round(ml_cape, 1),
        "qu_global_cin_j_kg": round(cin, 1),
        "qu_outlook_hail_environment_0_1": idx,
        "qu_outlook_uncertainty_0_1": round(unc, 4),
    }


def enrich_forecast_outlook(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach QuLab outlook fields; never overwrites user-provided ``qu_*`` keys."""
    lab = AtmosphericScienceLab()
    out: List[Dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        needs = any(
            r.get(k) is None
            for k in (
                "qu_global_cape_j_kg",
                "qu_global_cin_j_kg",
                "qu_outlook_hail_environment_0_1",
            )
        )
        if needs:
            est = estimate_outlook_for_row(r, lab)
            for k, v in est.items():
                if r.get(k) is None:
                    r[k] = v
        if r.get("graphcast_hail_potential_0_1") is None and r.get("qu_outlook_hail_environment_0_1") is not None:
            r["graphcast_hail_potential_0_1"] = float(r["qu_outlook_hail_environment_0_1"])
        if r.get("graphcast_cape_j_kg") is None and r.get("qu_global_cape_j_kg") is not None:
            r["graphcast_cape_j_kg"] = float(r["qu_global_cape_j_kg"])
        if r.get("graphcast_cin_j_kg") is None and r.get("qu_global_cin_j_kg") is not None:
            r["graphcast_cin_j_kg"] = float(r["qu_global_cin_j_kg"])
        out.append(r)
    return out


# --- Backward-compatible names (thin wrappers) ---
def hail_potential_from_cape_cin(
    cape_j_kg: float,
    cin_j_kg: Optional[float] = None,
    *,
    cape_high: float = 2000.0,
    cin_low: float = 50.0,
) -> float:
    """Legacy CAPE/CIN merge curve; prefer ``qu_outlook_hail_environment_0_1`` from rows."""
    cape = max(0.0, float(cape_j_kg))
    cape_term = min(1.0, cape / max(1.0, cape_high))
    if cin_j_kg is None:
        return float(round(cape_term, 4))
    cin = abs(float(cin_j_kg))
    cin_term = min(1.0, max(0.0, 1.0 - cin / max(1.0, cin_low)))
    return float(round(0.55 * cape_term + 0.45 * cin_term, 4))


def merge_graphcast_into_forecast_dicts(
    forecast_rows: List[Dict[str, Any]],
    *,
    cape_key: str = "graphcast_cape_j_kg",
    cin_key: str = "graphcast_cin_j_kg",
) -> List[Dict[str, Any]]:
    """Apply explicit CAPE/CIN hail index first, then QuLab outlook fill for remaining gaps."""
    staged: List[Dict[str, Any]] = []
    for row in forecast_rows:
        r = dict(row)
        cape = r.get(cape_key)
        if cape is not None and r.get("graphcast_hail_potential_0_1") is None:
            cin = r.get(cin_key)
            r["graphcast_hail_potential_0_1"] = hail_potential_from_cape_cin(
                float(cape), float(cin) if cin is not None else None
            )
        staged.append(r)
    return enrich_forecast_outlook(staged)
