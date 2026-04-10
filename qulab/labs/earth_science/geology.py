"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

GEOLOGY LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass
from scipy.constants import pi, g
from typing import List

# Constants and configuration
GRAVITY = g
EARTH_RADIUS = 6371e3  # in meters


@dataclass
class Rock:
    name: str
    density: float
    porosity: float
    compressibility: float
    thermal_conductivity: float
    specific_heat_capacity: float

    def __post_init__(self):
        self.bulk_density = (1 - self.porosity) * self.density


class GeologicalSite:
    def __init__(self, latitude: float, rocks: List[Rock]):
        self.latitude = np.radians(latitude)
        self.rocks = rocks
        self.surface_area = 4 * pi * EARTH_RADIUS ** 2 * (1 + np.cos(self.latitude)) / 2

    def calculate_gravity_effect(self):
        # Calculate the effective gravity at the site based on latitude and rock density
        return GRAVITY * np.cos(np.degrees(self.latitude))

    def simulate_weathering_process(self, years: int):
        for year in range(years):
            for rock in self.rocks:
                erosion_rate = 0.1 + (rock.bulk_density / 1000) ** 2
                rock.porosity += erosion_rate * np.sin(np.degrees(self.latitude)) * year

    def analyze_thermal_properties(self, temperature_range: List[float]):
        temperature_array = np.arange(temperature_range[0], temperature_range[-1] + 1, dtype=np.float64)
        heat_flux = np.zeros_like(temperature_array)

        for rock in self.rocks:
            thermal_diffusivity = rock.thermal_conductivity / (rock.bulk_density * rock.specific_heat_capacity)
            heat_flux += np.exp(-temperature_array / thermal_diffusivity)

        return temperature_array, heat_flux

    def simulate_groundwater_flow(self, hydraulic_conductivity: float, area: float):
        time_steps = 100
        dt = 365 * 24 * 3600  # one year in seconds
        n = int(np.sqrt(area / self.surface_area))

        h = np.zeros((n, n), dtype=np.float64)
        q = np.full_like(h, hydraulic_conductivity)

        for t in range(time_steps):
            h_new = np.copy(h)
            for i in range(n - 1):
                for j in range(n - 1):
                    h_new[i + 1, j] += dt * (q[i, j] - q[i + 1, j]) / self.surface_area
                    h_new[i, j + 1] += dt * (q[i, j] - q[i, j + 1]) / self.surface_area

            h = np.copy(h_new)

        return h


def run_demo():
    quartzite = Rock(name="Quartzite", density=2700.0, porosity=0.05, compressibility=4e-10,
                     thermal_conductivity=7.6, specific_heat_capacity=843)
    schist = Rock(name="Schist", density=2900.0, porosity=0.1, compressibility=5e-10,
                  thermal_conductivity=2.5, specific_heat_capacity=920)

    site = GeologicalSite(latitude=-34.6087, rocks=[quartzite, schist])

    print("Effective Gravity:", site.calculate_gravity_effect())
    print("\nWeathering Process Simulation for 10 Years:")
    site.simulate_weathering_process(10)
    for rock in site.rocks:
        print(f"{rock.name} porosity after 10 years: {rock.porosity:.4f}")

    temperatures, heat_flux = site.analyze_thermal_properties([273.15, 373.15])
    print("\nThermal Analysis:")
    for temp, flux in zip(temperatures, heat_flux):
        print(f"Temperature {temp}K: Heat Flux {flux:.4f}")

    hydraulic_conductivity = 1e-6
    area = 100 * EARTH_RADIUS ** 2 / np.cos(site.latitude)
    groundwater_flow = site.simulate_groundwater_flow(hydraulic_conductivity, area)

    print("\nGroundwater Flow Simulation (height [m]):")
    for row in groundwater_flow:
        print(" ".join(f"{val:.3f}" for val in row[:5]))  # showing first 5 values per row


if __name__ == '__main__':
    run_demo()