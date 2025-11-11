"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

PROTEIN FOLDING LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
from scipy.constants import pi

@dataclass
class Atom:
    name: str
    position: np.ndarray  # (x, y, z)
    mass: float = 0.0

@dataclass
class Residue:
    name: str
    atoms: List[Atom]
    charge: int = 0

@dataclass
class AminoAcidChain:
    sequence: str
    residues: Dict[str, Residue] = field(default_factory=dict)
    
    def build_chain(self):
        for i, aa in enumerate(self.sequence):
            self.residues[f"R{i}"] = Residue(name=aa, atoms=[], charge=-2 if aa == "G" else 0) # Simplified example

@dataclass
class Protein:
    name: str
    amino_acids: AminoAcidChain
    
    def build_protein(self):
        self.amino_acids.build_chain()
    
    def calculate_bonds(self, distance_threshold=1.5):
        for i in range(len(self.amino_acids.residues) - 1):
            r1 = list(self.amino_acids.residues.values())[i]
            r2 = list(self.amino_acids.residues.values())[i + 1]
            
            for a1 in r1.atoms:
                for a2 in r2.atoms:
                    if np.linalg.norm(a1.position - a2.position) < distance_threshold:
                        yield (a1, a2)
    
    def calculate_angles(self):
        angles = []
        for i in range(len(self.amino_acids.residues) - 2):
            r1 = list(self.amino_acids.residues.values())[i]
            r2 = list(self.amino_acids.residues.values())[i + 1]
            r3 = list(self.amino_acids.residues.values())[i + 2]
            
            for a1 in r1.atoms:
                for a2 in r2.atoms:
                    for a3 in r3.atoms:
                        v1 = a2.position - a1.position
                        v2 = a3.position - a2.position
                        
                        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        
                        angles.append(((a1, a2, a3), np.degrees(np.arccos(cos_theta))))
        
        return angles

@dataclass
class ProteinLab:
    proteins: List[Protein] = field(default_factory=list)
    
    def add_protein(self, protein):
        self.proteins.append(protein)
    
    def run_simulations(self):
        for p in self.proproteins:
            p.build_protein()
            
def run_demo():
    sequence = "GLY-ALA-GLY"
    p = Protein(name="TestProtein", amino_acids=AminoAcidChain(sequence=sequence))
    
    lab = ProteinLab()
    lab.add_protein(p)
    lab.run_simulations()
    
    for prot in lab.proteins:
        bonds = list(prot.calculate_bonds())
        print(f"Number of bonds: {len(bonds)}")
        
        angles = prot.calculate_angles()
        print(f"Number of angles: {len(angles)}")

if __name__ == '__main__':
    run_demo()
