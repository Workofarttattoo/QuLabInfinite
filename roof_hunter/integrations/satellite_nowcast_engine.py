"""QuLab proprietary GOES-inspired hail nowcast (15–40 min horizon) without external model weights.

Combines:
- Cloud-top cooling trend from IR window brightness temperature (Ch13-class proxy, °C).
- Overshooting-top proxy from very cold IR tops.
- Glaciation / vigorous updraft proxy from visible–IR coupling when visible norm is supplied.
- Electrification coupling from GLM-style flash rates when present.

Outputs ``qu_satellite_hail_nowcast_0_1`` and can populate ``lightning_severe_hail_prob_0_1`` when absent.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def satellite_hail_nowcast_from_row(row: Dict[str, Any]) -> Optional[float]:
    """Return 0..1 hail nowcast probability from satellite / lightning proxies, or None if no signal."""
    # IR cloud-top temperature (°C): cold = deep convection
    bt = row.get("goes_ch13_bt_c")
    bt_prev = row.get("goes_ch13_bt_prior_c")
    dt_min = row.get("goes_pair_dt_minutes")
    vis = row.get("goes_ch02_reflectance_norm_0_1")  # 0..1 normalized visible
    flashes = row.get("lightning_flashes_per_hour")

    terms: List[float] = []
    weights: List[float] = []

    if bt is not None:
        t_now = float(bt)
        # Overshoot / depth: colder than -42 °C ramps strongly; cap beyond -62 °C
        cold = _sigmoid(-(t_now + 48.0) / 5.5)
        terms.append(min(1.0, cold))
        weights.append(0.34)

        if bt_prev is not None and dt_min is not None and float(dt_min) > 0.5:
            cooling = (float(bt_prev) - t_now) / float(dt_min)  # °C per minute, positive = deepening
            growth = _sigmoid((cooling - 0.35) / 0.22)
            terms.append(min(1.0, growth))
            weights.append(0.36)

    if vis is not None and bt is not None:
        v = max(0.0, min(1.0, float(vis)))
        t_now = float(bt)
        glac = v * _sigmoid(-(t_now + 40.0) / 6.0)
        terms.append(min(1.0, glac))
        weights.append(0.18)

    if flashes is not None and float(flashes) > 0:
        f = float(flashes)
        elec = min(1.0, math.log1p(f) / math.log1p(900.0))
        terms.append(elec)
        weights.append(0.22)

    if not terms:
        return None

    wsum = sum(weights)
    p = sum(t * w for t, w in zip(terms, weights)) / wsum
    # Conservative squashing: satellite-only rarely exceeds 0.82 without lightning
    if flashes is None or float(flashes or 0) <= 0:
        p = 0.55 + 0.45 * p
        p = min(0.82, p)
    else:
        p = 0.5 + 0.5 * p
    return float(round(max(0.0, min(1.0, p)), 4))


def satellite_hail_nowcast_boost(prob_0_1: float, *, cap: float = 0.26) -> float:
    """Nonlinear boost for the twin: sqrt compression avoids over-weighting marginal satellite hits."""
    p = max(0.0, min(1.0, float(prob_0_1)))
    shaped = math.sqrt(p)
    return float(min(cap, 0.62 * shaped))


def enrich_forecast_satellite(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fill ``qu_satellite_hail_nowcast_0_1`` / ``lightning_severe_hail_prob_0_1`` when missing."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        if r.get("qu_satellite_hail_nowcast_0_1") is None:
            p = satellite_hail_nowcast_from_row(r)
            if p is not None:
                r["qu_satellite_hail_nowcast_0_1"] = p
        if r.get("lightning_severe_hail_prob_0_1") is None and r.get("qu_satellite_hail_nowcast_0_1") is not None:
            r["lightning_severe_hail_prob_0_1"] = float(r["qu_satellite_hail_nowcast_0_1"])
        out.append(r)
    return out


def lightning_severe_hail_boost(prob_0_1: float, *, cap: float = 0.26) -> float:
    """Backward-compatible name; delegates to ``satellite_hail_nowcast_boost``."""
    return satellite_hail_nowcast_boost(prob_0_1, cap=cap)


def merge_lightning_severe_into_forecast_dicts(
    forecast_rows: List[Dict[str, Any]],
    *,
    prob_key: str = "lightning_severe_hail_prob_0_1",
) -> List[Dict[str, Any]]:
    return enrich_forecast_satellite(list(forecast_rows))
