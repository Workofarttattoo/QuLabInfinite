"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

MATERIALS SCIENCE LAB
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.constants import k, Avogadro, g, c, h, e, pi
from typing import TypeVar

T = TypeVar('T')

@dataclass
class Material:
    name: str
    lattice_parameters: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    elastic_constants: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float64))
    density: float = 0.0
    thermal_conductivity: float = 0.0

class MaterialsLab:
    def __init__(self) -> None:
        self.materials: list[Material] = []
    
    def add_material(self, material: Material) -> None:
        self.materials.append(material)
    
    def calculate_bulk_modulus(self, elastic_constants: np.ndarray) -> float:
        C11, C12, C44 = elastic_constants
        bulk_modulus = (C11 + 2 * C12 / 3) * pi / (6 * k)
        return bulk_modulus
    
    def calculate_shear_modulus(self, elastic_constants: np.ndarray) -> float:
        G11 = self.calculate_bulk_modulus(elastic_constants) - pi * e**2 / (k * Avogadro)
        return G11
    
    def calculate_yield_strength(self, material: Material) -> float:
        yield_strength = 0.5 * self.calculate_shear_modulus(material.elastic_constants) * g
        return yield_strength

def run_demo() -> None:
    lab = MaterialsLab()
    
    # Example material parameters
    lattice_parameters = np.array([4.69, 4.69, 5.49], dtype=np.float64)
    elastic_constants = np.array([[182.37e9, -0.0, -0.0],
                                  [-0.0, 182.37e9, -0.0],
                                  [-0.0, -0.0, 124.58e9]], dtype=np.float64)
    density = 2070.0
    thermal_conductivity = 230.0

    # Create material instance
    copper = Material(name='Copper', lattice_parameters=lattice_parameters,
                      elastic_constants=elastic_constants, density=density,
                      thermal_conductivity=thermal_conductivity)

    # Add material to lab
    lab.add_material(copper)
    
    # Calculate properties
    bulk_modulus = lab.calculate_bulk_modulus(elastic_constants)
    shear_modulus = lab.calculate_shear_modulus(elastic_constants)
    yield_strength = lab.calculate_yield_strength(copper)

    print(f"Material: {copper.name}")
    print(f"Lattice Parameters (a, b, c): {lattice_parameters[0]:.2f} Å, {lattice_parameters[1]:.2f} Å, {lattice_parameters[2]:.2f} Å")
    print(f"Density: {density:.2f} kg/m³")
    print(f"Thermal Conductivity: {thermal_conductivity:.2f} W/(m·K)")
    print(f"Elastic Constants (C11, C12, C44): \n{elastic_constants}")
    print(f"Bulk Modulus: {bulk_modulus:.2e} Pa")
    print(f"Shear Modulus: {shear_modulus:.2e} Pa")
    print(f"Yield Strength: {yield_strength:.2f} Pa")

if __name__ == '__main__':
    run_demo()