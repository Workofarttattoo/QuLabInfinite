"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

MECHANICAL ENGINEERING LAB
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.constants import k, Avogadro, g, c, h, e, pi, physical_constants

@dataclass
class MechanicalEngineeringLab:
    Youngs_modulus: float = 200e9     # Pa (Steel)
    Poissons_ratio: float = 0.3       # Dimensionless
    density: float = 7850             # kg/m^3 (Density of steel)
    temperature: float = 293          # Kelvin

    def __post_init__(self):
        self.k_value = k
        self.Avogadro_number = Avogadro
        self.gravity = g
        self.light_speed = c
        self.Planck_constant = h
        self.elementary_charge = e
        self.pi_value = pi

    @property
    def linear_stress_strain(self, strain):
        return np.array([self.Youngs_modulus * i for i in strain], dtype=np.float64)

    @property
    def poisson_effective_modulus(self):
        E = self.Youngs_modulus
        v = self.Poissons_ratio
        effective_E = E / (1 - v**2)
        return np.array([effective_E])

    @property
    def material_mass_density(self, volume=np.array([1.0], dtype=np.float64)):
        mass = self.density * volume
        return mass

    @staticmethod
    def thermal_expansion_coefficient(alpha_thermal):
        expansion = alpha_thermal * g
        return expansion

    def bulk_modulus_virial_theory(self, pressure_array: np.ndarray) -> np.ndarray:
        K = (2 / 3) * self.Youngs_modulus * (1 - self.Poissons_ratio)
        virial = K + (self.density * g)**2 / (K + (4/3)*self.density*g)
        return virial

    def thermal_conductivity(self, thermal_cond=np.array([50], dtype=np.float64)):
        conductivity = thermal_cond
        return conductivity

    def heat_capacity_virial_theory(self):
        C_V = 3 * self.density / (self.Avogadro_number) * k * pi**2
        C_P = C_V + R_gas() * self.Poissons_ratio
        return np.array([C_V, C_P])

def R_gas():
    r"""Ideal gas constant in J/(mol·K)."""
    return physical_constants['Avogadro constant'][0] / 1.66053904e-27

def run_demo():
    lab = MechanicalEngineeringLab()
    
    strain_array = np.linspace(0, 1, 10)
    stress_array = lab.linear_stress_strain(strain=strain_array)

    print(f"Linear Stress: {stress_array}")

    effective_modulus = lab.poisson_effective_modulus
    print(f"Effective Modulus: {effective_modulus[0]} Pa")

    density_mass = lab.material_mass_density(volume=np.array([1, 2, 3], dtype=np.float64))
    print(f"Material Mass (1m³): {density_mass}")

    thermal_expansion_coefficient_array = np.linspace(0.01e-5, 0.1e-5, 10)
    expansion_array = lab.thermal_expansion_coefficient(alpha_thermal=thermal_expansion_coefficient_array)

    print(f"Thermal Expansion: {expansion_array}")

    pressure_array = np.array([1e3, 2e3, 3e3], dtype=np.float64)   # Pa
    virial_array = lab.bulk_modulus_virial_theory(pressure_array=pressure_array)
    
    print(f"Virial Theory: {virial_array}")

    thermal_conductivity_value = np.array([50.0, 100.0], dtype=np.float64)   # W/(m·K)
    conductivity_values = lab.thermal_conductivity(thermal_cond=thermal_conductivity_value)

    print(f"Thermal Conductivity: {conductivity_values}")

    heat_capacity_values = lab.heat_capacity_virial_theory()
    
    print(f"Heat Capacity (C_V, C_P): {heat_capacity_values}")

if __name__ == '__main__':
    run_demo()
