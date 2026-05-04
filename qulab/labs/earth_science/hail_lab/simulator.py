# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Refined Hail Simulator - 3D Lagrangian trajectory model.
Implements mass growth budget (Kumjian & Lombardo 2020) and Lightning Jump Algorithm.
"""

import numpy as np
import logging
from typing import Any, Dict, List, Tuple, Optional
from qulab.core.base_lab import BaseLab, register_lab

logger = logging.getLogger(__name__)

def refined_hail_trajectory(lat, lon, u_wind, v_wind, fall_velocity, time_to_ground):
    """
    Calculates the landing coordinate based on horizontal wind displacement.
    u_wind: West-to-East wind speed (m/s)
    v_wind: South-to-North wind speed (m/s)
    """
    # Earth's radius in meters for coordinate offset
    R = 6378137

    # Horizontal drift in meters
    delta_x = u_wind * time_to_ground
    delta_y = v_wind * time_to_ground

    # Convert meters to Lat/Lon offsets
    new_lat = lat + (delta_y / R) * (180 / np.pi)
    new_lon = lon + (delta_x / R) * (180 / np.pi) / np.cos(lat * np.pi / 180)

    return new_lat, new_lon

@register_lab(
    name="hail_simulator",
    category="earth_science",
    description="Refined 3D Hail Trajectory and Growth Simulator",
    version="2.0.0",
    tags=("hail", "simulation", "3D", "lightning-jump")
)
class RefinedHailSimulator(BaseLab):
    """
    Refined Hail Simulator implementing Lagrangian trajectory and mass growth budget.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.iterations = self.config.get("iterations", 1000)
        self.rho_air = 1.225  # kg/m^3 at sea level
        self.g = 9.80665      # Gravity m/s^2

    def calculate_terminal_velocity(self, diameter: float, mass: float) -> float:
        """Calculate terminal velocity with drag coefficient Cd=0.55."""
        Cd = 0.55
        area = np.pi * (diameter / 2)**2
        # Avoid division by zero
        if area == 0 or Cd == 0:
            return 0.0
        return np.sqrt((2 * mass * self.g) / (self.rho_air * area * Cd))

    def calculate_mass_growth(self, D_h: float, W_liq: float, W_ice: float, V_rel: float, temp_c: float) -> float:
        """
        Calculate dm_h/dt using mass growth budget (Kumjian & Lombardo 2020).
        dt is assumed to be 1 second for rate calculation.
        """
        # Collection efficiencies (varying by temperature as placeholder logic)
        # E_cw (liquid), E_ci (ice)
        if temp_c > 0:
            E_cw = 1.0
            E_ci = 0.0
        elif temp_c > -20:
            E_cw = 0.8
            E_ci = 0.1
        else:
            E_cw = 0.5
            E_ci = 0.2

        # dm_h/dt = (pi * D_h^2 / 4) * (E_cw * W_liq + E_ci * W_ice) * V_rel
        dm_dt = (np.pi * D_h**2 / 4) * (E_cw * W_liq + E_ci * W_ice) * V_rel
        return dm_dt

    def run_experiment(self, experiment_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Monte Carlo hail simulation.
        """
        self._track_experiment()

        start_lat = experiment_spec.get("lat", 0.0)
        start_lon = experiment_spec.get("lon", 0.0)
        storm_params = experiment_spec.get("storm_params", {})

        # Lightning Jump Algorithm: Override updraft vertical velocity (w)
        flash_rate = storm_params.get("flash_rate", 0.0)
        flash_rate_sigma = storm_params.get("flash_rate_sigma", 0.0)
        flash_rate_mean = storm_params.get("flash_rate_mean", 0.0)

        updraft = storm_params.get("updraft", 10.0)

        # 2-sigma spike detection
        if flash_rate > (flash_rate_mean + 2 * flash_rate_sigma) and flash_rate_sigma > 0:
            logger.info("Lightning jump detected (>= 2 sigma). Intensifying updraft.")
            # Refinement: w approx sqrt(FlashRate)
            updraft = np.sqrt(flash_rate)

        v_shear = storm_params.get("v_shear", (0.0, 0.0)) # (u, v)
        height = storm_params.get("height", 5000.0) # meters

        # Herbie data placeholders
        W_liq = storm_params.get("W_liq", 0.001) # kg/m^3
        W_ice = storm_params.get("W_ice", 0.0005) # kg/m^3
        temp_c = storm_params.get("temp_c", -10.0)

        strikes = []
        sizes = []

        for _ in range(self.iterations):
            # Monte Carlo variance in initial size (mean 2.5cm/1 inch)
            size = np.random.normal(0.025, 0.005)
            if size < 0.005: size = 0.005 # minimum size

            mass = (4/3) * np.pi * (size/2)**3 * 900 # Ice density ~900kg/m^3

            # Refined Growth integration (simplified 1-step for demo)
            # In a real model this would be integrated over the trajectory
            V_rel = updraft # Relative velocity approx updraft speed
            dm_dt = self.calculate_mass_growth(size, W_liq, W_ice, V_rel, temp_c)

            # Assume 100s growth phase in updraft
            mass += dm_dt * 100
            # Recalculate size from new mass
            size = 2 * ( (3 * mass) / (4 * np.pi * 900) )**(1/3)

            v_term = self.calculate_terminal_velocity(size, mass)
            if v_term <= 0: v_term = 0.1

            fall_time = height / v_term

            # 2-Layer Wind Shear: average storm steering flow and surface wind
            # user suggested (v_shear * 0.7) * fall_time
            drift_u = (v_shear[0] * 0.7)
            drift_v = (v_shear[1] * 0.7)

            new_lat, new_lon = refined_hail_trajectory(start_lat, start_lon, drift_u, drift_v, v_term, fall_time)

            strikes.append((new_lat, new_lon))
            sizes.append(size)

        return {
            "status": "success",
            "strikes": strikes,
            "sizes_meters": sizes,
            "storm_parameters": {
                "updraft_used": updraft,
                "height": height,
                "v_shear": v_shear
            }
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "lab": "RefinedHailSimulator",
            "experiment_count": self._experiment_count,
            "uptime": self.uptime_seconds
        }

if __name__ == "__main__":
    # Quick test
    sim = RefinedHailSimulator()
    spec = {
        "lat": 35.0,
        "lon": -97.0,
        "storm_params": {
            "updraft": 20.0,
            "v_shear": (15.0, 5.0),
            "height": 6000.0,
            "flash_rate": 50,
            "flash_rate_mean": 10,
            "flash_rate_sigma": 5
        }
    }
    results = sim.run_experiment(spec)
    print(f"Generated {len(results['strikes'])} strikes.")
    print(f"First strike: {results['strikes'][0]}")
    print(f"Updraft used: {results['storm_parameters']['updraft_used']}")
