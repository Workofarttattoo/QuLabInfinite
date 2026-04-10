"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ELECTROMAGNETISM LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.constants import k, Avogadro, g, c, h, e, pi
from typing import TypeVar, Callable

T = TypeVar('T')


@dataclass(frozen=True)
class ElectromagneticConstants:
    k: float = k
    Avogadro: int = Avogadro
    g: float = g
    c: float = c
    h: float = h
    e: float = e
    pi: float = pi


@dataclass(frozen=True)
class ElectromagneticField:
    electric_field_strength: np.ndarray = field(default=np.zeros((3, 1), dtype=np.float64))
    magnetic_flux_density: np.ndarray = field(default=np.zeros((3, 1), dtype=np.float64))

    def apply_force_on_charge(self, charge_vector: np.ndarray) -> np.ndarray:
        return self.electric_field_strength * charge_vector

    def calculate_magnetic_force(self, current_vector: np.ndarray) -> np.ndarray:
        magnetic_force = np.cross(self.magnetic_flux_density, current_vector)
        return magnetic_force


@dataclass(frozen=True)
class ElectricPotential:
    potential_gradient: np.ndarray = field(default=np.zeros((3, 1), dtype=np.float64))
    charge_distribution: Callable[[np.ndarray], np.ndarray] = lambda x: np.ones_like(x)

    def electric_field_from_potential(self) -> ElectromagneticField:
        electric_field_strength = -self.potential_gradient
        return ElectromagneticField(electric_field_strength=electric_field_strength, magnetic_flux_density=np.zeros((3, 1), dtype=np.float64))

    @classmethod
    def from_charge_distribution(cls, charge_distribution: Callable[[np.ndarray], np.ndarray]) -> 'ElectricPotential':
        return cls(charge_distribution=charge_distribution)


@dataclass(frozen=True)
class ConductingMaterial:
    conductivity: float = 0.0
    permittivity: float = 8.85418782e-12

    def __post_init__(self):
        if self.conductivity <= 0 or self.permittivity <= 0:
            raise ValueError("Conductivity and permittivity must be positive.")

    def electric_field_strength_at_boundary(self, voltage_diff: float, distance_between_boundaries: float) -> float:
        return voltage_diff / distance_between_boundaries

    def magnetic_flux_density_for_current_sheet(self, current_per_unit_length: float, distance_from_sheet: float) -> np.ndarray:
        permeability_of_free_space = 1.25663706e-6
        magnetic_flux_density = (permeability_of_free_space * self.conductivity * current_per_unit_length / (2 * pi * distance_from_sheet))
        return np.array([[magnetic_flux_density], [0], [0]], dtype=np.float64)


@dataclass(frozen=True)
class ElectromagnetismLab:
    constants: ElectromagneticConstants = field(default_factory=ElectromagneticConstants)

    def setup_conductor(self, conductivity_value: float) -> ConductingMaterial:
        return ConductingMaterial(conductivity=conductivity_value, permittivity=self.constants.e / self.constants.pi * np.sqrt(self.constants.c ** 2 - (self.constants.h / self.constants.e) ** 2))

    def calculate_magnetic_flux_density_for_current_sheet(self, current_per_unit_length: float, distance_from_sheet: float, conductor: ConductingMaterial = None) -> np.ndarray:
        if conductor is None:
            conductor = self.setup_conductor(5.96e7)
        return conductor.magnetic_flux_density_for_current_sheet(current_per_unit_length=current_per_unit_length, distance_from_sheet=distance_from_sheet)

    def calculate_electric_field_strength_at_boundary(self, voltage_diff: float, distance_between_boundaries: float, conductor: ConductingMaterial = None) -> float:
        if conductor is None:
            conductor = self.setup_conductor(5.96e7)
        return conductor.electric_field_strength_at_boundary(voltage_diff=voltage_diff, distance_between_boundaries=distance_between_boundaries)


def run_demo():
    lab_setup = ElectromagnetismLab()
    
    # Setup a conducting material with specific conductivity
    copper_conductor = lab_setup.setup_conductor(5.96e7)
    
    # Calculate magnetic flux density for a current sheet at a distance
    print(lab_setup.calculate_magnetic_flux_density_for_current_sheet(current_per_unit_length=20, distance_from_sheet=10, conductor=copper_conductor))
    
    # Calculate electric field strength at boundary given voltage and distance
    print(lab_setup.calculate_electric_field_strength_at_boundary(voltage_diff=5, distance_between_boundaries=1e-3))


if __name__ == '__main__':
    run_demo()