"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

CELL BIOLOGY LAB
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.constants import k, Avogadro, g, c, h, e, pi
from typing import Tuple

@dataclass
class CellBiologyLab:
    temperature: float = 310.15  # Default temperature in Kelvin
    volume: float = field(default=1e-6)  # Volume of cell culture in m^3 (default is 1 cubic millimeter)
    pressure: float = field(default=101325)  # Atmospheric pressure in Pa

    def __post_init__(self):
        self.constants = {
            'k': k,
            'Avogadro': Avogadro,
            'g': g,
            'c': c,
            'h': h,
            'e': e,
            'pi': pi
        }

    def ideal_gas_law(self, n: float) -> np.ndarray:
        """Calculates pressure for an ideal gas given number of moles."""
        p = self.pressure * (self.constants['n'] / n)
        return np.array([p], dtype=np.float64)

    def calculate_concentration(self, volume: float, mass: float, molar_mass: float) -> np.ndarray:
        """Calculates concentration of a solute in mol/m^3."""
        concentration = (mass / molar_mass) / volume
        return np.array([concentration], dtype=np.float64)

    def osmotic_pressure(self, i: float, c: np.ndarray, temperature: float) -> np.ndarray:
        """Calculates osmotic pressure of a solution given molality."""
        osmotic = self.constants['i'] * c[0] * k * temperature
        return np.array([osmotic], dtype=np.float64)

    def diffusion_coefficient(self, r: float, t: float) -> np.ndarray:
        """Calculates diffusion coefficient D from Stokes-Einstein equation."""
        d = (k * self.temperature) / (6 * pi * self.constants['e'] * r)
        return np.array([d], dtype=np.float64)

    def hydrostatic_pressure(self, depth: float) -> np.ndarray:
        """Calculates hydrostatic pressure at given depth in Pa."""
        p_hydro = self.constants['g'] * self.temperature * depth
        return np.array([p_hydro], dtype=np.float64)

    def electrical_potential_difference(self, i1: float, i2: float) -> np.ndarray:
        """Calculates potential difference between two concentrations using Nernst equation."""
        v_nernst = (self.constants['e'] * self.temperature / (self.constants['k']) *
                    np.log(i1 / i2))
        return np.array([v_nernst], dtype=np.float64)

    def ion_molarity(self, concentration: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates molarity of ions given initial concentration."""
        c_neg = (self.constants['e'] * self.temperature * concentration / 18.0)
        c_pos = (-1.0 * self.constants['e'] * self.temperature * concentration / 27.0)

        return np.array([c_pos], dtype=np.float64), np.array([c_neg], dtype=np.float64)

    def permeability_coefficient(self, d: float) -> np.ndarray:
        """Calculates permeability coefficient for a given diffusion constant."""
        p = (d * self.volume)
        return np.array([p], dtype=np.float64)


def run_demo():
    cell_bio_lab = CellBiologyLab()

    # Example of ideal gas law
    n_moles = 1e-5
    print("Ideal Gas Law Calculation")
    print(f"Pressure at {n_moles} moles: {cell_bio_lab.ideal_gas_law(n_moles)} Pa")

    # Calculate concentration example
    volume_ml = 0.5
    mass_grams = 0.01
    molar_mass_gmol = 18.0
    print(f"Concentration Calculation")
    print(f"Molar Concentration: {cell_bio_lab.calculate_concentration(volume_ml * 1e-6, mass_grams, molar_mass_gmol)} mol/m^3")

    # Osmotic pressure example
    concentration_mol = cell_bio_lab.calculate_concentration(volume_ml * 1e-6, mass_grams, molar_mass_gmol)
    print(f"Osmotic Pressure Calculation")
    print(f"Osmotic Pressure: {cell_bio_lab.osmotic_pressure(2.0, concentration_mol, cell_bio_lab.temperature)} Pa")

    # Diffusion coefficient example
    radius_nm = 1e-9 * 50
    print(f"Diffusion Coefficient Calculation")
    print(f"Diffusion Constant (D): {cell_bio_lab.diffusion_coefficient(radius_nm, cell_bio_lab.temperature)} m^2/s")

    # Hydrostatic pressure at a depth of 2 meters
    depth_meters = 2.0
    print(f"Hydrostatic Pressure at Depth")
    print(f"Hydrostatic Pressure: {cell_bio_lab.hydrostatic_pressure(depth_meters)} Pa")

    # Electrical potential difference example (Nernst equation)
    i1, i2 = 1e-3, 5e-4
    print(f"Electrical Potential Difference Calculation")
    print(f"Nernst Potential: {cell_bio_lab.electrical_potential_difference(i1, i2)} V")

    # Permeability coefficient example
    d_coefficient = cell_bio_lab.diffusion_coefficient(radius_nm, cell_bio_lab.temperature)
    print(f"Permeability Coefficient Calculation")
    print(f"Permeability Coefficient: {cell_bio_lab.permeability_coefficient(d_coefficient[0])} m^3/s")

if __name__ == '__main__':
    run_demo()