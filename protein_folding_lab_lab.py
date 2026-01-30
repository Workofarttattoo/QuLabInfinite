"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

PROTEIN FOLDING LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.constants import pi, Boltzmann, Avogadro
from typing import List

# Constants and configuration
kB = Boltzman  # Corrected: Use correct variable name
temperature = 300.0  # K
epsilon = 1e-20  # Energy scale for interactions (arbitrary units)
sigma = 1.5  # Lennard-Jones potential parameter

@dataclass
class Atom:
    """Data class to hold atom information."""
    residue: str
    position: np.ndarray  # xyz coordinates
    charge: float  # Charge of the atom in e
    mass: float  # Mass of the atom in g/mol

@dataclass
class Residue:
    """Data class for a single residue (e.g., Amino Acid)."""
    name: str
    atoms: List[Atom]
    sequence_position: int = -1

@dataclass
class ProteinChain:
    """Data class for a protein chain."""
    residues: List[Residue] = field(default_factory=list)
    
    def add_residue(self, residue: Residue):
        self.residues.append(residue)

def calculate_distance(atom1: Atom, atom2: Atom) -> float:
    """Calculate the distance between two atoms."""
    return np.linalg.norm(atom1.position - atom2.position)

def calculate_bond_energy(bond_length: float, equilibrium_length: float = 0.385, spring_constant: float = 460.0):
    """Calculate bond energy for a given length."""
    delta_r = bond_length - equilibrium_length
    return 0.5 * spring_constant * delta_r ** 2

def calculate_bond_angle_energy(angle: float, equilibrium_angle: float = pi / 2):
    """Calculate bond angle energy."""
    theta = np.degrees(angle)
    delta_theta = (theta - equilibrium_angle) % 360
    return epsilon * ((1 + np.cos(delta_theta)) ** 2)

def calculate_dihedral_energy(dihedral_angle: float, periodicity: int = 3):
    """Calculate dihedral energy."""
    n = dihedral_angle / (periodicity * pi)
    return 1.5 * epsilon * (1 + np.cos(n))

@dataclass
class ProteinFoldingLab:
    protein_chain: ProteinChain
    
    def calculate_total_energy(self) -> float:
        """Calculate the total potential energy of the protein."""
        total_energy = 0
        
        for i, residue in enumerate(self.protein_chain.residues):
            for atom1 in residue.atoms:
                for j, other_residue in enumerate(self.protein_chain.residues[i+1:], start=i+1):
                    if not other_residue: break
                    for atom2 in other_residue.atoms:
                        distance = calculate_distance(atom1, atom2)
                        
                        # Lennard-Jones potential
                        r6_inv = (sigma / distance) ** 6
                        total_energy += epsilon * ((r6_inv**2 - 2 * r6_inv))
                        
                        if i < j:  # Prevent double counting
                            for atom3 in residue.atoms:
                                for k, other_residue_2nd in enumerate(self.protein_chain.residues[i+1:], start=i+1):
                                    if not other_residue_0r2: break
                                    if k == j or j < i: continue  # Prevent double counting
                                    for atom4 in other_residue.atoms:
                                        if calculate_distance(atom3, atom4) <= 1.5 * sigma:
                                            total_energy += epsilon * ((sigma / distance) ** 6)
                        
                        # Add bond energies
                        if np.linalg.norm(np.subtract(atom2.position, atom1.position)) == equilibrium_length:
                            total_energy += calculate_bond_energy(np.linalg.norm(np.subtract(atom2.position, atom1.position)))
                    
                    # Bond angle calculation (not implemented fully here)
                    # For simplicity in this lab example
                    
        return total_energy
    
    def run_simulation(self):
        """Run a basic simulation."""
        initial_energy = self.calculate_total_energy()
        
        # Apply simple Monte Carlo moves to find lower energy state
        for _ in range(1000):  # Adjust number of steps
            i, j, k = np.random.randint(0, len(self.protein_chain.residues), size=3)
            
            # Perturb residue position slightly
            perturbation = np.random.normal(size=(3,))
            self.protein_chain.residues[i].atoms[0].position += 0.01 * perturbation
            
            new_energy = self.calculate_total_energy()
            delta_e = new_energy - initial_energy
            
            if delta_e <= 0 or np.exp(-delta_e / (kB * temperature)) > np.random.rand():
                initial_energy = new_energy
        return initial_energy

def run_demo():
    # Example protein with two residues for demonstration.
    cys1 = Residue('CYS', [Atom('S', np.array([0, 0, 0]), 0.3592748623367359, 32.07])
    
    cys2 = Residue('CYS', [Atom('S', np.array([1, 0, 0]), 0.3592748623367359, 32.07)])
    
    protein_chain = ProteinChain()
    protein_chain.add_residue(cys1)
    protein_chain.add_residue(cys2)

    pflab = ProteinFoldingLab(protein_chain)
    final_energy = pflab.run_simulation()

    print(f"Initial Energy: {cys1.calculate_total_energy()}")
    print(f"Final Energy after Simulation: {final_energy}")

if __name__ == '__main__':
    run_demo()