"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ECOLOGY LAB
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
import scipy
import scipy.constants
from scipy.constants import k, Avogadro, g, c, h, e, pi
from typing import Any

@dataclass
class EcologyData:
    temperature: float = 25.0  # Celsius
    pressure: float = 101325.0  # Pa
    volume: float = 1.0  # m^3
    n_moles: int = 1
    species_count: dict[str, int] = field(default_factory=dict)

@dataclass
class EcologyLab:
    config: EcologyData

    def __init__(self, config):
        self.config = config

    def kelvin_to_celsius(self, temperature_k):
        return np.float64(temperature_k - 273.15)

    def celsius_to_kelvin(self, temperature_c):
        return np.float64(temperature_c + 273.15)

    def ideal_gas_law(self) -> float:
        """Calculate the volume using the Ideal Gas Law"""
        r = scipy.constants.R
        t_k = self.celsius_to_kelvin(self.config.temperature)
        p_pa = self.config.pressure

        return np.float64(r * t_k / (p_pa * Avogadro))

    def standard_molar_volume(self) -> float:
        """Calculate the standard molar volume"""
        r_t = scipy.constants.R
        t0_k = 273.15
        p0_pa = 101325

        return np.float64(r_t * t0_k / (p0_pa * Avogadro))

    def gas_constant(self) -> float:
        """Calculate the universal gas constant"""
        r_gas = scipy.constants.R

        return np.float64(r_gas)

    def species_distribution(self, species_counts: dict[str, int]) -> np.ndarray[np.float64]:
        """Return a numpy array of species distribution"""
        self.config.species_count = species_counts
        total_species = sum(species_counts.values())

        return np.array([count / total_species for count in species_counts.values()], dtype=np.float64)

    def run_demo(self):
        config = EcologyData(temperature=25.0, pressure=101325.0, volume=1.0, n_moles=1)
        lab = EcologyLab(config)

        print(f"Temperature in Celsius: {config.temperature}")
        print(f"Pressure (Pa): {config.pressure}")
        print(f"Ideal Gas Volume (m^3/mol): {lab.ideal_gas_law():.5f}")
        print(f"Standard Molar Volume (m^3/mol): {lab.standard_molar_volume():.5f}")
        print(f"Gas Constant: {lab.gas_constant()}")

        species_counts = {"species1": 20, "species2": 15, "species3": 5}
        dist = lab.species_distribution(species_counts)
        print(f"Species Distribution: {dist}")

if __name__ == '__main__':
    EcologyLab.run_demo(EcologyLab(None))