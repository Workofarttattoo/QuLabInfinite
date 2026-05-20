"""Monte Carlo hail physics engine (Lagrangian particle ensemble through a 1D column).

This module is the **physics-forward** simulator: it draws stochastic hail embryos, lifts them
with a updraft distribution (optionally tied to lightning as a proxy for electrified updraft
strength), then falls them through a simplified warm layer with melting. Outputs are empirical
PDFs (percentiles, P(D > threshold)), intended as a **ground-truth-style** check or prior for
Bayesian / ML layers—not a replacement for full cloud microphysics or radar hydrometeor ID.

Research touchpoint: Lagrangian stochastic parcel / particle methods in cloudy boundary layers
and convection (see e.g. arXiv:2301.05656 for LSM context). Implementation here is deliberately
compact and auditable for product integration.

Patent posture (engineering, not legal advice): the public API emphasizes **lightning-derived
updraft statistics** (``LightningUpdraftMapper``) + **explicit Monte Carlo hail trajectories**,
distinct from broad “single-score weather risk” claims or full NEXRAD hydrometeor classification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# --- Physical constants (SI) ---
RHO_AIR = 1.15  # kg/m^3 near surface warm layer
RHO_ICE = 917.0  # kg/m^3
G = 9.80665
L_F = 3.34e5  # latent heat of fusion J/kg


@dataclass
class ColumnProfile:
    """Simplified vertical column for 1D hail trajectories."""

    #: Height of cloud base above ground level (m).
    cloud_base_agl_m: float = 3500.0
    #: Depth below cloud base through which melting is applied (m), e.g. warm layer under CB.
    melting_slab_depth_m: float = 3000.0
    #: Environmental lapse rate dT/dz (K/m), negative = temperature decreases with height.
    lapse_k_per_m: float = -6.5e-3
    #: Surface temperature (°C) — sets wet-bulb deficit for melting.
    surface_temp_c: float = 25.0
    #: Surface dewpoint (°C); if None, uses rh with surface_temp_c.
    surface_dewpoint_c: Optional[float] = None
    relative_humidity: float = 0.65


def _saturation_vapor_pressure_hpa(t_c: float) -> float:
    return 6.112 * math.exp((17.67 * t_c) / (t_c + 243.5))


def _wet_bulb_stull_2011(t_c: float, rh_percent: float) -> float:
    """Stull (2011) wet-bulb from T (°C) and RH (0–100)."""
    rh = min(100.0, max(1.0, rh_percent))
    t = t_c
    tw = (
        t * math.atan(0.151977 * (rh + 8.313659) ** 0.5)
        + math.atan(t + rh)
        - 1.676331
        + 0.00391838 * rh**1.5 * math.atan(0.023101 * rh * t - 0.107312)
    )
    return float(tw)


class LightningUpdraftMapper:
    """Map lightning activity + CAPE to updraft speed mean/std (m/s) for Monte Carlo draws.

    This is intentionally **diagnostic**: GLM or network flash rates modulate updraft distribution
    while CAPE sets the thermodynamic ceiling. It is not NEXRAD-derived vertical velocity.
    """

    def __init__(
        self,
        *,
        cape_scale: float = 2200.0,
        flash_ref: float = 400.0,
        w_base: float = 6.0,
        w_per_sqrt_cape: float = 0.22,
        w_per_log_flash: float = 3.2,
        std_floor: float = 2.8,
        std_slope: float = 0.12,
    ) -> None:
        self.cape_scale = cape_scale
        self.flash_ref = flash_ref
        self.w_base = w_base
        self.w_per_sqrt_cape = w_per_sqrt_cape
        self.w_per_log_flash = w_per_log_flash
        self.std_floor = std_floor
        self.std_slope = std_slope

    def updraft_m_s(self, cape_j_kg: float, flashes_per_hour: Optional[float]) -> Tuple[float, float]:
        cape = max(0.0, float(cape_j_kg))
        flash_term = 0.0
        if flashes_per_hour is not None and flashes_per_hour > 0:
            flash_term = self.w_per_log_flash * (math.log1p(flashes_per_hour) / math.log1p(self.flash_ref))
        w_mean = self.w_base + self.w_per_sqrt_cape * math.sqrt(cape / self.cape_scale) * 22.0 + flash_term
        w_mean = float(min(55.0, max(4.0, w_mean)))
        w_std = self.std_floor + self.std_slope * w_mean
        return w_mean, w_std


class HailPhysicsEngine:
    """Monte Carlo ensemble of hail stones through a 1D melting slab."""

    def __init__(self, iterations: int = 1000, seed: Optional[int] = None) -> None:
        self.iterations = int(iterations)
        self._rng = np.random.default_rng(seed)
        self.lightning_mapper = LightningUpdraftMapper()

    def run_simulation(
        self,
        cloud_base_agl_m: float,
        lapse_k_per_m: float,
        updraft_speed_mean: float,
        *,
        updraft_speed_std: float = 5.0,
        melting_slab_depth_m: float = 3000.0,
        surface_temp_c: float = 25.0,
        surface_dewpoint_c: Optional[float] = None,
        relative_humidity: float = 0.65,
    ) -> Dict[str, float]:
        """Return summary statistics of ground-reach diameters (meters)."""
        profile = ColumnProfile(
            cloud_base_agl_m=cloud_base_agl_m,
            melting_slab_depth_m=melting_slab_depth_m,
            lapse_k_per_m=lapse_k_per_m,
            surface_temp_c=surface_temp_c,
            surface_dewpoint_c=surface_dewpoint_c,
            relative_humidity=relative_humidity,
        )
        sizes: List[float] = []
        for _ in range(self.iterations):
            d0 = float(self._rng.triangular(0.005, 0.012, 0.045))  # 0.5–4.5 cm-ish, m
            w = float(self._rng.normal(updraft_speed_mean, updraft_speed_std))
            w = max(1.0, w)
            d_ground = self._integrate_one_stone(d0, w, profile)
            sizes.append(d_ground)
        return self._analyze(np.array(sizes, dtype=np.float64))

    def run_from_forecast_row(
        self,
        row: Dict[str, Any],
        *,
        cloud_base_agl_m: float = 3200.0,
        melting_slab_depth_m: float = 2800.0,
        lapse_k_per_m: Optional[float] = None,
    ) -> Dict[str, float]:
        """Build column inputs from a Roof Hunter-style JSON row and run the ensemble.

        Uses ``qu_global_cape_j_kg`` or ``graphcast_cape_j_kg`` or CAPE implied by surface fields
        via a coarse proxy when missing: ``max(400, 12 * (T-Td) * sqrt(RH))`` (J/kg scale, not MLCAPE).
        """
        cape = row.get("qu_global_cape_j_kg")
        if cape is None:
            cape = row.get("graphcast_cape_j_kg")
        if cape is None:
            t = float(row.get("surface_temp_c", 20.0))
            td = row.get("surface_dewpoint_c")
            if td is None:
                td = t - 8.0
            else:
                td = float(td)
            rh = float(row.get("relative_humidity", 0.65))
            spread = max(0.0, t - td)
            cape = max(200.0, min(4500.0, 320.0 + 38.0 * spread * math.sqrt(max(0.15, rh))))

        flashes = row.get("lightning_flashes_per_hour")
        fph = float(flashes) if flashes is not None else None
        w_mean, w_std = self.lightning_mapper.updraft_m_s(float(cape), fph)

        if lapse_k_per_m is None:
            lapse_k_per_m = -6.5e-3

        return self.run_simulation(
            cloud_base_agl_m,
            lapse_k_per_m,
            w_mean,
            updraft_speed_std=w_std,
            melting_slab_depth_m=melting_slab_depth_m,
            surface_temp_c=float(row.get("surface_temp_c", 22.0)),
            surface_dewpoint_c=row.get("surface_dewpoint_c"),
            relative_humidity=float(row.get("relative_humidity", 0.65)),
        )

    def run_from_hrrr_column_features(
        self,
        features: Dict[str, Any],
        *,
        cloud_base_agl_m: float = 3200.0,
        melting_slab_depth_m: float = 2800.0,
    ) -> Dict[str, float]:
        """Run from HRRR column feature dict produced by hrrr_column_ingest."""
        cape = float(features.get("cape_proxy_j_kg", 1200.0))
        flashes = features.get("lightning_flashes_per_hour")
        fph = None if flashes is None else float(flashes)
        w_mean, w_std = self.lightning_mapper.updraft_m_s(cape, fph)

        # Blend in HRRR-derived vertical velocity if present.
        col_w = features.get("updraft_speed_ms")
        if col_w is not None:
            col = max(1.0, float(col_w))
            w_mean = 0.55 * w_mean + 0.45 * col
            w_std = max(2.0, 0.8 * w_std + 0.12 * col)

        lapse_k_per_m = float(features.get("lapse_700_500_k_per_km", 6.5)) / -1000.0
        return self.run_simulation(
            cloud_base_agl_m=cloud_base_agl_m,
            lapse_k_per_m=lapse_k_per_m,
            updraft_speed_mean=w_mean,
            updraft_speed_std=w_std,
            melting_slab_depth_m=melting_slab_depth_m,
            surface_temp_c=float(features.get("surface_temp_c", 25.0)),
            surface_dewpoint_c=features.get("surface_dewpoint_c"),
            relative_humidity=float(features.get("relative_humidity", 0.65)),
        )

    def _terminal_velocity_m_s(self, d_m: float) -> float:
        """Approximate hail terminal velocity (m/s); scales ~ sqrt(d) for large spheres in drag."""
        d_cm = max(0.05, d_m * 100.0)
        # Empirical blend: small stones drag-limited, large stones approach ~sqrt(d) growth
        return float(min(55.0, 4.2 + 6.8 * math.sqrt(d_cm)))

    def _wet_bulb_surface(self, profile: ColumnProfile) -> float:
        t = profile.surface_temp_c
        if profile.surface_dewpoint_c is not None:
            es = _saturation_vapor_pressure_hpa(t)
            e = _saturation_vapor_pressure_hpa(profile.surface_dewpoint_c)
            rh_pct = min(100.0, max(1.0, 100.0 * e / max(1e-6, es)))
        else:
            rh_pct = min(100.0, max(1.0, 100.0 * profile.relative_humidity))
        return _wet_bulb_stull_2011(t, rh_pct)

    def _ambient_temp_c(self, z_m: float, profile: ColumnProfile) -> float:
        """Linear profile: T(z) = T_sfc + lapse * z for z AGL in melting slab."""
        return profile.surface_temp_c + profile.lapse_k_per_m * z_m

    def _integrate_one_stone(self, d0_m: float, updraft_m_s: float, profile: ColumnProfile) -> float:
        """Lift then fall through melting slab; return residual diameter at ground (m)."""
        tw_sfc = self._wet_bulb_surface(profile)
        # Short residence in updraft: larger stones recycle longer — stochastic multiplier
        tau_up = self._rng.uniform(0.25, 1.1) * (d0_m / 0.02) ** 0.35
        growth_m = 1.0e-6 * updraft_m_s * tau_up * (max(0.0, tw_sfc - 5.0)) ** 0.5
        d = d0_m + growth_m
        d = min(0.12, d)

        z = 0.0
        dz = 50.0
        depth = max(200.0, profile.melting_slab_depth_m)
        while z < depth:
            v_fall = self._terminal_velocity_m_s(d)
            dt = dz / max(0.5, v_fall)
            t_env = self._ambient_temp_c(z, profile)
            # Warm-layer melting: excess above 0 °C, damped with height (most melting near cloud base).
            thermal_excess = max(0.0, t_env)
            drive = min(thermal_excess, max(0.0, tw_sfc)) * math.exp(-z / 1400.0)
            # Ventilation ~ d^1.85; small coefficient so cm-class stones survive realistic 2–4 km paths.
            melt_m = 7.0e-7 * drive * (d * 100.0) ** 1.85 * dt
            d = max(0.0, d - melt_m)
            if d <= 1e-5:
                break
            z += dz
        return float(d)

    def _analyze(self, sizes_m: np.ndarray) -> Dict[str, float]:
        inch = 0.0254
        p1 = float(np.mean(sizes_m > inch))
        p2 = float(np.mean(sizes_m > 2 * inch))
        return {
            "iterations": float(self.iterations),
            "max_diameter_mm": float(np.max(sizes_m) * 1000.0),
            "p95_diameter_mm": float(np.quantile(sizes_m, 0.95) * 1000.0),
            "p50_diameter_mm": float(np.quantile(sizes_m, 0.50) * 1000.0),
            "mean_diameter_mm": float(np.mean(sizes_m) * 1000.0),
            "damage_probability_gt_1in": p1,
            "damage_probability_gt_2in": p2,
            "std_diameter_mm": float(np.std(sizes_m) * 1000.0),
        }


def run_demo() -> Dict[str, float]:
    eng = HailPhysicsEngine(iterations=2000, seed=42)
    return eng.run_simulation(
        cloud_base_agl_m=3200.0,
        lapse_k_per_m=-6.5e-3,
        updraft_speed_mean=22.0,
        updraft_speed_std=4.5,
        melting_slab_depth_m=2800.0,
        surface_temp_c=28.0,
        surface_dewpoint_c=21.0,
    )


if __name__ == "__main__":
    out = run_demo()
    for k, v in out.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
