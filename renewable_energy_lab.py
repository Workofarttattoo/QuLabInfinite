"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

RENEWABLE ENERGY LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.constants import pi, h, c, k, e

@dataclass
class SolarPanel:
    efficiency: float = 0.20  # Typical solar panel efficiency
    area_m2: float = 1.5      # Area of the solar panel in square meters
    max_power_w: float = 300  # Maximum power output in Watts

@dataclass
class WindTurbine:
    rotor_radius_m: float = 70.0  # Radius of the turbine's rotor
    air_density_kgpm3: float = 1.225  # Air density at sea level (kg/m^3)
    efficiency: float = 0.4  # Typical wind turbine efficiency

    def power_output(self, wind_speed_mps: np.float64) -> np.float64:
        return 0.5 * self.air_density_kgpm3 * pi * self.rotor_radius_m**2 * wind_speed_mps**3 * self.efficiency

@dataclass
class HydroelectricPlant:
    water_flow_rate_m3s: float = 100.0   # Flow rate in cubic meters per second (m^3/s)
    head_height_m: float = 50.0           # Head height or drop distance of water, measured vertically in meters (m)

    def power_output(self) -> np.float64:
        return self.water_flow_rate_m3s * self.head_height_m * 9.81

class RenewableEnergyLab:

    def __init__(self):
        self.solar_panel = SolarPanel()
        self.wind_turbine = WindTurbine(rotor_radius_m=70.0, air_density_kgpm3=1.225)
        self.hydroelectric_plant = HydroelectricPlant()

    def simulate_solar_power(self, irradiance_wm2: np.float64) -> np.float64:
        return self.solar_panel.efficiency * irradiance_wm2 * self.solar_panel.area_m2

    def simulate_wind_energy(self, wind_speeds_kmph: np.ndarray) -> np.ndarray:
        wind_speeds_mps = wind_speeds_kmph / 3.6
        power_output_arr = np.array([self.wind_turbine.power_output(ws) for ws in wind_speeds_mps], dtype=np.float64)
        return power_output_arr

    def simulate_hydroelectric_energy(self, water_flow_rates_m3s: np.ndarray, head_heights_m: np.ndarray) -> np.ndarray:
        power_output_arr = np.array([self.hydroelectric_plant.power_output() for _ in range(len(water_flow_rates_m3s))], dtype=np.float64)
        return power_output_arr

def run_demo():
    lab = RenewableEnergyLab()

    # Simulate solar power output given the irradiance of 1000 W/m^2
    print(f"Simulated Solar Power Output (W): {lab.simulate_solar_power(irradiance_wm2=1000.0)}")

    # Simulate wind power output for various wind speeds in km/h
    wind_speeds_kmph = np.array([5, 10, 15, 20, 25], dtype=np.float64)
    print(f"Simulated Wind Power Output (W): {lab.simulate_wind_energy(wind_speeds_kmph)}")

    # Simulate hydroelectric power output for a given flow rate and head height
    water_flow_rates_m3s = np.array([100, 200], dtype=np.float64)
    head_heights_m = np.array([50, 75], dtype=np.float64)
    print(f"Simulated Hydroelectric Power Output (W): {lab.simulate_hydroelectric_energy(water_flow_rates_m3s, head_heights_m)}")

if __name__ == '__main__':
    run_demo()