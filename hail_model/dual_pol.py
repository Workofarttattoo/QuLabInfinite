"""
Dual-Polarisation Hail Size Estimation.

Implements NOAA's operational algorithms for hail detection and size
estimation using dual-pol radar variables:
  - Hydrometeor Classification Algorithm (HCA)
  - Maximum Expected Size of Hail (MESH)
  - Probability of Severe Hail (POSH)
  - Hail Size Discrimination Algorithm (HSDA)

References:
  Park et al. (2009) — HCA for WSR-88D
  Witt et al. (1998) — MESH / POSH
  Ortega et al. (2016) — HSDA
  Ryzhkov et al. (2013) — Dual-pol hail signatures

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class HydrometeorType(Enum):
    """HCA classification categories (Park et al. 2009)."""
    BIOLOGICAL = "BI"
    GROUND_CLUTTER = "GC"
    ICE_CRYSTALS = "IC"
    DRY_SNOW = "DS"
    WET_SNOW = "WS"
    LIGHT_RAIN = "LR"
    MODERATE_RAIN = "MR"
    HEAVY_RAIN = "HR"
    BIG_DROPS = "BD"
    GRAUPEL = "GR"
    HAIL_RAIN = "HA"      # hail mixed with rain
    LARGE_HAIL = "LH"     # dominant large hail
    GIANT_HAIL = "GH"     # baseball+ sized hail
    UNKNOWN = "UK"


@dataclass
class DualPolObservation:
    """Dual-pol radar moments for a single range gate."""
    reflectivity_h: float    # ZH — horizontal reflectivity (dBZ)
    differential_reflectivity: float  # ZDR (dB)
    correlation_coefficient: float    # ρHV (unitless, 0–1)
    specific_differential_phase: float  # KDP (°/km)
    temperature_c: float = -10.0     # environment temperature at beam height


@dataclass
class HailEstimate:
    """Output from hail size estimation."""
    hydrometeor_class: HydrometeorType
    mesh_inches: float
    posh_percent: float
    estimated_diameter_inches: float
    confidence: float
    hail_detected: bool


# =====================================================================
# Hydrometeor Classification Algorithm (HCA)
# =====================================================================

def classify_hydrometeor(obs: DualPolObservation) -> HydrometeorType:
    """Classify the dominant hydrometeor using dual-pol moments.

    Simplified version of the NSSL HCA (Park et al. 2009) using
    membership-function logic on ZH, ZDR, ρHV, KDP, and T.
    """
    zh = obs.reflectivity_h
    zdr = obs.differential_reflectivity
    rho = obs.correlation_coefficient
    kdp = obs.specific_differential_phase
    t = obs.temperature_c

    # Large / giant hail signature
    if zh >= 60 and zdr < 1.0 and rho < 0.92:
        return HydrometeorType.GIANT_HAIL if zh >= 70 else HydrometeorType.LARGE_HAIL

    # Hail mixed with rain
    if zh >= 50 and zdr < 2.0 and rho < 0.95 and t < 0:
        return HydrometeorType.HAIL_RAIN

    # Graupel
    if 40 <= zh < 55 and 0.0 <= zdr <= 2.0 and rho >= 0.95 and t < -5:
        return HydrometeorType.GRAUPEL

    # Heavy rain
    if zh >= 50 and zdr >= 2.0 and rho >= 0.95 and kdp >= 1.5:
        return HydrometeorType.HEAVY_RAIN

    # Big drops
    if 35 <= zh < 50 and zdr >= 2.5 and rho >= 0.97:
        return HydrometeorType.BIG_DROPS

    # Moderate rain
    if 30 <= zh < 50 and 0.5 <= zdr < 2.5 and rho >= 0.97:
        return HydrometeorType.MODERATE_RAIN

    # Dry snow (check before light rain — temperature is the discriminator)
    if t < -2 and zh < 35 and zdr < 0.5 and rho >= 0.98:
        return HydrometeorType.DRY_SNOW

    # Light rain
    if 15 <= zh < 35 and zdr >= 0.0 and rho >= 0.98:
        return HydrometeorType.LIGHT_RAIN

    # Wet snow (near 0 °C)
    if -2 <= t <= 3 and rho < 0.95 and 20 <= zh <= 45:
        return HydrometeorType.WET_SNOW

    # Ice crystals
    if t < -10 and zh < 20 and zdr > 1.0 and rho >= 0.99:
        return HydrometeorType.ICE_CRYSTALS

    # Biological / ground clutter (low rho, low reflectivity)
    if zh < 20 and rho < 0.90:
        return HydrometeorType.GROUND_CLUTTER if t > 5 else HydrometeorType.BIOLOGICAL

    return HydrometeorType.UNKNOWN


# =====================================================================
# MESH — Maximum Expected Size of Hail  (Witt et al. 1998)
# =====================================================================

def compute_mesh(
    reflectivity_profile_dbz: list[float],
    heights_km: list[float],
    freezing_level_km: float = 3.5,
) -> float:
    """Compute MESH (inches) from a vertical reflectivity profile.

    MESH uses the Severe Hail Index (SHI) which integrates weighted
    reflectivity above the freezing level.

    Parameters
    ----------
    reflectivity_profile_dbz : dBZ values at successive height levels
    heights_km : corresponding heights (km AGL)
    freezing_level_km : 0 °C isotherm height

    Returns
    -------
    MESH in inches (0 if no hail signal)
    """
    if len(reflectivity_profile_dbz) != len(heights_km):
        raise ValueError("Profile and height arrays must match")

    # Compute SHI
    shi = 0.0
    for i in range(len(heights_km) - 1):
        z = reflectivity_profile_dbz[i]
        h = heights_km[i]
        dh = heights_km[i + 1] - h

        if h <= freezing_level_km or z < 40.0:
            continue

        # Hail kinetic energy flux (Witt 1998 eq.)
        wt = _temperature_weight(h, freezing_level_km)
        hke = 5e-6 * 10 ** (0.084 * z) * wt
        shi += hke * dh

    if shi <= 0:
        return 0.0

    # MESH empirical formula: MESH = 2.54 * SHI^0.5
    mesh_mm = 2.54 * shi ** 0.5
    mesh_inches = mesh_mm / 25.4
    return round(max(0.0, mesh_inches), 2)


def _temperature_weight(height_km: float, freezing_km: float) -> float:
    """Temperature weighting function for SHI (Witt 1998)."""
    if height_km <= freezing_km:
        return 0.0
    ratio = (height_km - freezing_km) / max(freezing_km, 1.0)
    return min(1.0, ratio)


# =====================================================================
# POSH — Probability of Severe Hail  (Witt et al. 1998)
# =====================================================================

def compute_posh(
    reflectivity_max_dbz: float,
    echo_top_km: float,
    freezing_level_km: float = 3.5,
) -> float:
    """Probability of Severe Hail (≥ 1.0 inch) as a percentage.

    Uses the simplified POSH formula from Witt et al. (1998):
        WT = (echo_top - freezing_level) / freezing_level
        SHI = f(Z, WT)
        POSH = 29 * ln(SHI) + 50   (clamped 0–100)
    """
    if echo_top_km <= freezing_level_km or reflectivity_max_dbz < 40:
        return 0.0

    wt = min(1.0, (echo_top_km - freezing_level_km) / max(freezing_level_km, 1.0))
    hke = 5e-6 * 10 ** (0.084 * reflectivity_max_dbz) * wt
    shi = hke * (echo_top_km - freezing_level_km)

    if shi <= 0:
        return 0.0

    posh = 29.0 * math.log(shi) + 50.0
    return round(max(0.0, min(100.0, posh)), 1)


# =====================================================================
# HSDA — Hail Size Discrimination Algorithm  (Ortega et al. 2016)
# =====================================================================

def estimate_hail_size(obs: DualPolObservation) -> HailEstimate:
    """Full hail estimation: HCA class + MESH + POSH + diameter.

    Combines dual-pol classification with empirical size estimates.
    """
    hclass = classify_hydrometeor(obs)

    # Determine if hail is present
    hail_types = {
        HydrometeorType.HAIL_RAIN,
        HydrometeorType.LARGE_HAIL,
        HydrometeorType.GIANT_HAIL,
        HydrometeorType.GRAUPEL,
    }
    hail_detected = hclass in hail_types

    # Estimate POSH from single-level data (proxy)
    echo_top_proxy = max(5.0, obs.reflectivity_h / 5.5)
    freezing_proxy = max(2.0, 4.0 + obs.temperature_c / 5.0)
    posh = compute_posh(obs.reflectivity_h, echo_top_proxy, freezing_proxy)

    # Reflectivity-based hail diameter (Ortega et al. 2016 regression)
    if obs.reflectivity_h >= 40 and hail_detected:
        diameter_mm = _zdr_adjusted_diameter(obs.reflectivity_h, obs.differential_reflectivity)
    else:
        diameter_mm = 0.0

    diameter_inches = round(diameter_mm / 25.4, 2)

    # MESH from a synthetic single-level profile
    profile_z = [obs.reflectivity_h] * 5
    profile_h = [freezing_proxy + i * 1.0 for i in range(5)]
    mesh = compute_mesh(profile_z, profile_h, freezing_proxy)

    # Confidence score
    confidence = _hail_confidence(obs, hclass)

    return HailEstimate(
        hydrometeor_class=hclass,
        mesh_inches=mesh,
        posh_percent=posh,
        estimated_diameter_inches=max(diameter_inches, mesh),
        confidence=confidence,
        hail_detected=hail_detected,
    )


def _zdr_adjusted_diameter(zh: float, zdr: float) -> float:
    """Estimate hail diameter (mm) from ZH and ZDR.

    Lower ZDR with high ZH → larger, more spherical hailstones.
    Based on Ryzhkov et al. (2013) and Ortega (2016).
    """
    base_diameter = 10 ** ((zh - 44.0) / 21.0)  # rough D(Z) relation

    # ZDR adjustment: near-zero ZDR indicates tumbling (large) hailstones
    if zdr < 0.5:
        base_diameter *= 1.4  # boost for tumbling hail
    elif zdr < 1.0:
        base_diameter *= 1.2
    elif zdr > 3.0:
        base_diameter *= 0.6  # high ZDR → rain-dominant, reduce

    return max(0.0, min(200.0, base_diameter))


def _hail_confidence(obs: DualPolObservation, hclass: HydrometeorType) -> float:
    """Assign a 0–1 confidence to the hail classification."""
    if hclass == HydrometeorType.GIANT_HAIL:
        return 0.95
    if hclass == HydrometeorType.LARGE_HAIL:
        return 0.90
    if hclass == HydrometeorType.HAIL_RAIN:
        return 0.75
    if hclass == HydrometeorType.GRAUPEL:
        return 0.60

    # Non-hail types: confidence that there is *no* hail
    if obs.reflectivity_h >= 55 and obs.correlation_coefficient < 0.93:
        return 0.50  # ambiguous
    return 0.15
