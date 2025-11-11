"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

POLYMER CHEMISTRY LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.constants import epsilon_0, Boltzmann

@dataclass
class PolymerChain:
    """
    A class representing a polymer chain with properties and methods for simulation.
    """
    monomer_mass: float = 150.0       # Average mass of monomers in g/mol
    n_monomers: int = 100             # Number of monomers in the polymer chain
    temperature: float = 300          # Temperature in Kelvin
    solvent_permittivity: float = 78   # Permittivity of solvent

    def __post_init__(self):
        self.molecular_weight = self.n_monomers * self.monomer_mass
        self.charge_density = np.zeros(self.n_monomers, dtype=np.float64)  # Charge density on each monomer
        self.dipole_strength = np.array([0.1] * self.n_monomers, dtype=np.float64)  # Dipole moment strength per monomer

    def calculate_end_to_end_distance(self):
        """
        Calculates the end-to-end distance of a polymer chain in a theta solvent.
        Uses R_g = (Mw/3N)^0.5 where Mw is molecular weight and N is number of monomers.
        """
        r_squared_mean = (self.molecular_weight / self.n_monomers) ** 2
        return np.sqrt(r_squared_mean)

    def calculate_dielectric_screening(self):
        """
        Calculates the dielectric screening factor based on solvent permittivity.
        """
        alpha = np.exp(-np.pi * epsilon_0 * self.dipole_strength / (self.solvent_permittivity * Boltzmann * self.temperature))
        return alpha

    def thermal_fluctuations(self, time_steps: int):
        """
        Simulates thermal fluctuations of the polymer chain over given time steps.
        """
        delta_q = np.random.normal(0, 1, size=(time_steps, self.n_monomers)).astype(np.float64)
        q_t = np.cumsum(delta_q, axis=0) * (self.temperature / self.molecular_weight)**0.5
        return q_t

@dataclass
class PolymerNetwork:
    """
    A class representing a polymer network made of many polymer chains.
    """
    n_chains: int = 10
    polymers: list[PolymerChain] = field(default_factory=list)
    
    def __post_init__(self):
        self.polymers = [PolymerChain() for _ in range(self.n_chains)]

def run_demo():
    polymer_chain = PolymerChain()
    print("End-to-end distance:", polymer_chain.calculate_end_to_end_distance())
    print("Dielectric screening factor:", polymer_chain.calculate_dielectric_screening())

    time_steps = 10
    q_t = polymer_chain.thermal_fluctuations(time_steps)
    print(f"Thermal fluctuations over {time_steps} steps:\n{q_t}")

if __name__ == '__main__':
    run_demo()