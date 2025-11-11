"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

OCEANOGRAPHY LAB
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.constants import k, Avogadro, g, c, h, e, pi, physical_constants

@dataclass
class OceanographyLab:
    """
    A class representing an oceanographic laboratory with various functionalities for marine research and analysis.
    """

    # Constants
    sea_water_density: float = 1025.0  # kg/m^3
    salinity: float = 35.0  # PSU
    temperature_range: tuple = (-2, 30)  # Celsius

    def __init__(self):
        self.pressure_levels: np.ndarray = np.linspace(0, 1000, 200, dtype=np.float64)
        self.temperature_profiles: dict[str, np.ndarray] = {}
        self.salinity_profiles: dict[str, np.ndarray] = {}

    def calculate_density(self, temperature: float) -> np.ndarray:
        """
        Calculate the density of seawater using polynomial approximation.
        """
        a1 = 999.842594
        a2 = 6.79385e-2
        a3 = -9.09529e-3
        a4 = 1.001685e-4
        a5 = -1.17387e-6
        a6 = 1.120083e-8

        b0 = 0.999947
        b1 = -6.122e-2
        b2 = +1.018e-4
        b3 = -1.558e-6
        b4 = +1.856e-8

        c1 = 4.8314e-9
        c2 = +2.361e-7
        c3 = -2.075e-6

        d1 = -1.1954e-6
        d2 = +1.7122e-5
        d3 = -8.576e-6

        t = np.linspace(self.temperature_range[0], self.temperature_range[1], 30, dtype=np.float64)

        temp_part_a = (a1 + a2 * temperature + a3 * temperature ** 2) / (1 + b1 * temperature + b2 * temperature ** 2)
        temp_part_b = c1 * temperature - t
        density_temp = temp_part_a * (1 + d1 * np.cosh(t) * temp_part_b)

        return density_temp

    def calculate_salinity_effects(self, salinity: float):
        """
        Calculate the effect of salinity on water density.
        """
        rho_0 = 999.842594
        beta = (1 - e ** (-salinity / 36)) * 8.24e-2 + 7.74e-3
        alpha = beta * k * 298.15 / (rho_0 * salinity) + 2.10e-4

        return rho_0 + alpha * salinity

    def calculate_pressure_at_depth(self):
        """
        Calculate pressure at different depths in the ocean.
        """
        return self.pressure_levels * g * sea_water_density

    def run_simulation(self, temperature: float = 25.0, salinity: float = 34.0):
        density_profile = self.calculate_density(temperature)
        salinity_effect = self.calculate_salinity_effects(salinity)

        pressure_at_depth = self.calculate_pressure_at_depth()

        return density_profile, salinity_effect, pressure_at_depth

    def run_demo(self) -> None:
        """
        Run a demonstration of the OceanographyLab simulation.
        """
        demo_temperature = 20.0
        demo_salinity = 35.0

        density_profile, salinity_effect, pressure_at_depth = self.run_simulation(demo_temperature, demo_salinity)

        print("Density Profile (kg/m^3):", density_profile)
        print(f"Salinity Effect on Density at {demo_salinity} PSU:", salinity_effect)
        print("Pressure at Depth (Pa):", pressure_at_depth)


if __name__ == '__main__':
    lab = OceanographyLab()
    lab.run_demo()