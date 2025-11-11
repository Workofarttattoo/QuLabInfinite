"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

QUANTUM MECHANICS LAB
Free gift to the scientific community from QuLabInfinite.
"""

from typing import Union, TypeVar, List
import numpy as np
from dataclasses import dataclass, field
from scipy import constants

# Constants and configuration
kB = 1.380649e-23  # Boltzmann constant (already in scipy.constants)
hbar = 1.0545718e-34  # Reduced Planck's constant (Planck constant / (2 * pi))
c = constants.c  # Speed of light
elementary_charge = constants.e
Avogadro_number = constants.Avogadro
g_gravity = 9.80665

# Type alias for clarity
NDArray = np.ndarray


@dataclass
class QuantumState:
    name: str
    energy_levels: NDArray
    occupation_probability: NDArray = field(init=False)

    def __post_init__(self):
        self.occupation_probability = self.calculate_occupation()

    def calculate_occupation(self) -> NDArray:
        return np.exp(-self.energy_levels / (kB * 300)) / sum(np.exp(-self.energy_levels / (kB * 300)))

    
@dataclass
class QuantumSystem:
    states: List[QuantumState]
    hamiltonian_matrix: NDArray = field(init=False)

    def __post_init__(self):
        self.hamiltonian_matrix = np.diag([state.energy_levels for state in self.states])

    def energy_spectrum(self) -> NDArray:
        return np.linalg.eigvals(self.hamiltonian_matrix)

    
def run_demo():
    # Define a simple quantum system
    e0, e1 = 0.0e-20, 5.0e-20
    
    ground_state = QuantumState(
        name="Ground State",
        energy_levels=np.array([e0], dtype=np.float64)
    )
    
    excited_state = QuantumState(
        name="Excited State",
        energy_levels=np.array([e1], dtype=np.float64)
    )
    
    system = QuantumSystem(states=[ground_state, excited_state])

    print("Occupation probability for ground state:", ground_state.occupation_probability)
    print("Energy spectrum of the system:", system.energy_spectrum())

if __name__ == "__main__":
    run_demo()