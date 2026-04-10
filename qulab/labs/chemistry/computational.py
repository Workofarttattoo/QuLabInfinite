"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

COMPUTATIONAL CHEMISTRY LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.constants import physical_constants

# Constants and configuration
HARTREE_TO_EV = physical_constants['Hartree energy'][0] / 1.60218e-19  # Conversion factor from Hartree to eV
BOHR_RADIUS = physical_constants['Bohr radius'][0] * 1e10               # Bohr radius in Angstroms

@dataclass
class Atom:
    element: str                  # Chemical symbol of the atom
    position: np.ndarray          # Position vector [x, y, z]
    charge: float = 0.0           # Atomic charge (default is neutral)
    
    def __post_init__(self):
        self.position = np.array(self.position, dtype=np.float64)

@dataclass
class Molecule:
    atoms: list[Atom]             # List of atoms in the molecule
    
    def calculate_dipole_moment(self) -> np.ndarray:
        total_charge_vector = np.zeros(3, dtype=np.float64)
        
        for atom in self.atoms:
            charge_vector = atom.position * atom.charge
            total_charge_vector += charge_vector
            
        return total_charge_vector

@dataclass
class System:
    molecules: list[Molecule]     # List of molecules in the system
    
    def calculate_total_dipole_moment(self) -> np.ndarray:
        total_system_dipole = np.zeros(3, dtype=np.float64)
        
        for molecule in self.molecules:
            dipole_moment = molecule.calculate_dipole_moment()
            total_system_dipole += dipole_moment
            
        return total_system_dipole

def run_demo():
    # Define a simple molecule: H2O
    h1_position = np.array([0.75, 0.0, 0.0], dtype=np.float64) * BOHR_RADIUS
    o_position = np.zeros(3, dtype=np.float64)
    
    atoms = [
        Atom('H', h1_position),
        Atom('O', o_position),
        Atom('H', -h1_position),
    ]
    
    water_molecule = Molecule(atoms=atoms)

    # Define a system with multiple molecules
    system = System(molecules=[water_molecule])

    total_dipole_moment = system.calculate_total_dipole_moment()
    print(f"Total dipole moment of the system: {total_dipole_moment}")

if __name__ == '__main__':
    run_demo()
