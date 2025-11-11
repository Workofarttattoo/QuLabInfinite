"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

INORGANIC CHEMISTRY LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.constants import k, Avogadro, g, c, h, e, pi, physical_constants
from typing import Dict
import scipy.constants

# Constants and configuration
PHYSICAL_CONSTANTS = {
    'planck_mass': physical_constants['Planck mass'][0],
    'rydberg_energy': physical_constants['Rydberg constant times hc in eV'][0],
}

@dataclass
class InorganicChemistryLab:
    temperature: float = field(default=298.15, metadata={"help": "Temperature in Kelvin"})
    pressure: float = field(default=101325, metadata={"help": "Pressure in Pascal"})
    gas_constant: float = k * Avogadro

    def calculate_molar_volume(self) -> np.ndarray:
        """Calculate molar volume at given temperature and pressure."""
        return (self.gas_constant * self.temperature) / self.pressure

    def lattice_energy_calculator(self, q1: int, q2: int, r0: float, epsilon: float = 9.0, n: float = 8.5):
        """Calculate Born-Haber cycle lattice energy."""
        return (epsilon * h * c) / (n * r0)

    def band_gap_calculator(self, e1: float, e2: float, k_value: int = 8.617333262e-5):
        """Calculate band gap from energy levels and Boltzmann constant."""
        return -(k_value * self.temperature) * np.log(e1 / e2)

    def crystal_field_splitting_energy(self, d_orbital_population: int, octahedral_field_strength: float = 0.429,
                                      tetrahedral_field_strength: float = 1.735):
        """Calculate crystal field splitting energy for transition metals."""
        return octahedral_field_strength if d_orbital_population > 6 else tetrahedral_field_strength

    def redox_potential(self, standard_electrode_potentials: Dict[str, float], anode: str = None,
                        cathode: str = None):
        """Calculate redox potential for electrochemical cells."""
        # If anode/cathode not provided, use first two entries from dict
        keys = list(standard_electrode_potentials.keys())
        if cathode is None and len(keys) > 0:
            cathode = keys[0]
        if anode is None and len(keys) > 1:
            anode = keys[1]
        if cathode not in standard_electrode_potentials or anode not in standard_electrode_potentials:
            return 0.0
        return (standard_electrode_potentials[cathode] - standard_electrode_potentials[anode]) / e

    def activation_energy(self, Ea: float, T1: float = 300.0, T2: float = 400.0):
        """Calculate activation energy using Arrhenius equation."""
        A = np.exp(-Ea / (self.gas_constant * T1))
        B = np.exp(-Ea / (self.gas_constant * T2))
        return -(self.gas_constant) * ((np.log(A) - np.log(B)) / (1/T1 - 1/T2))

    def gibbs_free_energy(self, delta_h: float, temperature: float):
        """Calculate Gibbs free energy change at given enthalpy and temperature."""
        return delta_h - (self.gas_constant * temperature)

def run_demo():
    lab = InorganicChemistryLab()
    
    print("Molar Volume:", lab.calculate_molar_volume())
    print("Lattice Energy (for NaCl, q1=+1, q2=-1, r0=2.36 Angstrom):",
          lab.lattice_energy_calculator(q1=+1, q2=-1, r0=2.36e-10))
    print("Band Gap for a semiconductor with E1 = 5 eV and E2 = 4.8 eV:",
          lab.band_gap_calculator(e1=5, e2=4.8))
    print("Crystal Field Splitting Energy for d^7 configuration in octahedral field:",
          lab.crystal_field_splitting_energy(d_orbital_population=7))
    standard_electrode_potentials = {"Fe3+/Fe2+": 0.771, "O2/OH-": -0.401}
    print("Redox Potential for Fe3+|Fe2+ -> O2|OH- electrode pair:",
          lab.redox_potential(standard_electrode_potentials))
    print("Activation Energy (Ea=50 kJ/mol) at T1 = 300 K and T2 = 400 K:",
          lab.activation_energy(Ea=50e3))

if __name__ == '__main__':
    run_demo()