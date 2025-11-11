from dataclasses import dataclass, field
import numpy as np
from scipy.constants import k, Avogadro, g, c, h, e, pi
from typing import *

# Constants and Configuration
DEFAULT_PEPTIDE_MASS = 50.0  # in Daltons (amu)
DEFAULT_PROTON_MASS = 1.00728  # in amu

class ProteomicsLab:
    def __init__(self):
        self.peptide_mass: np.ndarray = DEFAULT_PEPTIDE_MASS * np.ones(1, dtype=np.float64)
        self.proton_mass: np.ndarray = DEFAULT_PROTON_MASS * np.ones(1, dtype=np.float64)

    @staticmethod
    def calculate_ion_mass(peptide_mass: np.ndarray, charge: int) -> np.ndarray:
        """
        Calculate the mass of an ion given peptide mass and charge.
        """
        proton_mass = ProteomicsLab().proton_mass  # Default proton mass
        return peptide_mass + (proton_mass * charge)

    @staticmethod
    def calculate_kinetic_energy(mass: float, velocity: float) -> np.ndarray:
        """
        Calculate kinetic energy given mass and velocity.
        """
        kinetic_energy = 0.5 * mass * velocity**2
        return kinetic_energy

    @classmethod
    def get_physical_constant(cls, name):
        return physical_constants[name]

@dataclass
class Ion:
    peptide_mass: np.ndarray
    charge: int
    ion_mass: np.ndarray = field(init=False)
    kinetic_energy: np.ndarray = field(init=False)

    def __post_init__(self):
        self.ion_mass = ProteomicsLab.calculate_ion_mass(self.peptide_mass, self.charge)
        # Assuming velocity to be 1 for demonstration purposes.
        self.kinetic_energy = ProteomicsLab.calculate_kinetic_energy(self.ion_mass[0], 1)

def run_demo():
    peptide_mass = np.array([250.0], dtype=np.float64)  # Example peptide mass
    charge = -1  # Example ion charge

    ion = Ion(peptide_mass, charge)
    print(f"Ion Mass: {ion.ion_mass}")
    print(f"Kinetic Energy: {ion.kinetic_energy}")

if __name__ == '__main__':
    run_demo()