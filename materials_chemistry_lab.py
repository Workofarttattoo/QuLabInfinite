"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

MATERIALS CHEMISTRY LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.constants import pi, Avogadro

@dataclass
class Material:
    name: str
    lattice_constant: float = 3.615  # Silicon example in Å
    basis_atoms: list = field(default_factory=lambda: ['Si'])
    num_cells_x: int = 2
    num_cells_y: int = 2
    num_cells_z: int = 2

class MaterialsChemistryLab:
    def __init__(self, materials):
        self.materials = materials
    
    def create_crystal_structure(self, material):
        lattice_constant = np.array([material.lattice_constant], dtype=np.float64)
        basis_atoms = np.array(material.basis_atoms, dtype=object)
        
        # Construct the primitive cell matrix
        a1 = lattice_constant * np.array([1.0, 0.0, 0.0])
        a2 = lattice_constant * np.array([0.5, np.sqrt(3)/2, 0.0])
        a3 = lattice_constant * np.array([0.0, 0.0, np.cbrt(material.lattice_constant**3)])
        
        # Create the full crystal structure
        crystal_vectors = []
        for i in range(-material.num_cells_x//2 + 1, material.num_cells_x//2 + 1):
            for j in range(-material.num_cells_y//2 + 1, material.num_cells_y//2 + 1):
                for k in range(-material.num_cells_z//2 + 1, material.num_cells_z//2 + 1):
                    r = i*a1 + j*a2 + k*a3
                    crystal_vectors.append(r)
        crystal_vectors = np.array(crystal_vectors, dtype=np.float64)

        return crystal_vectors, basis_atoms
    
    def calculate_unit_volume(self, lattice_constant):
        return (lattice_constant ** 3) * pi / 3

def run_demo():
    material1 = Material(name="Silicon Diamond Cubic", lattice_constant=5.431)
    lab = MaterialsChemistryLab(materials=[material1])

    crystal_vectors, basis_atoms = lab.create_crystal_structure(material1)

    print(f"Crystal structure for {material1.name}:")
    print("Lattice Vectors:")
    print(crystal_vectors[:8])  # Print first few vectors as example
    print("\nBasis Atoms:", basis_atoms)

if __name__ == '__main__':
    run_demo()