"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ANALYTICAL CHEMISTRY LAB
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.constants import k, Avogadro, g, c, h, e, pi, physical_constants

@dataclass
class AnalyticalChemistryLab:
    _temperature: float = 298.15  # K
    _density_water: float = 0.997  # kg/m^3 at 20°C
    _molality_kcl: np.ndarray = field(default_factory=lambda: np.array([0, 0], dtype=np.float64))
    _molality_naoh: np.ndarray = field(default_factory=lambda: np.array([0, 0], dtype=np.float64))

    def __post_init__(self):
        self._molar_mass_kcl = physical_constants["chlorine-krypton bond length in KCl"][1]
        self._molar_mass_naoh = physical_constants["oxygen-hydrogen bond length in NaOH"][1]

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        if value < 0 or value > 500:
            raise ValueError("Temperature must be between 0 and 500 K")
        self._temperature = value

    def calculate_density_kcl_solution(self) -> np.ndarray:
        return (self._density_water * (1 - (np.array([self._molality_kcl[0], self._molality_kcl[1]], dtype=np.float64) / 1000)) + (self._molar_mass_kcl / Avogadro))

    def calculate_density_naoh_solution(self) -> np.ndarray:
        return (self._density_water * (1 - (np.array([self._molality_naoh[0], self._molality_naoh[1]], dtype=np.float64) / 1000)) + (self._molar_mass_naoh / Avogadro))

    def calculate_chemical_potential(self, molality: np.ndarray) -> np.ndarray:
        return k * self.temperature * np.log(molality)

    def calculate_activity_coefficient(self, molality: np.ndarray) -> np.ndarray:
        return 1 / (1 + ((np.array([self._molality_kcl[0], self._molality_naoh[1]], dtype=np.float64)) ** 2))

def run_demo():
    lab = AnalyticalChemistryLab()
    print(f"Density of KCl solution: {lab.calculate_density_kcl_solution()}")
    print(f"Density of NaOH solution: {lab.calculate_density_naoh_solution()}")
    print(f"Chemical potential for KCl: {lab.calculate_chemical_potential(lab._molality_kcl)}")
    print(f"Activity coefficient for NaOH: {lab.calculate_activity_coefficient(lab._molality_naoh)}")

if __name__ == '__main__':
    run_demo()