"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

MOLECULAR BIOLOGY LAB
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.constants import k, Avogadro, g, c, h, e, pi
from typing import List

@dataclass
class Nucleotide:
    name: str
    base_pairs: List[str] = field(default_factory=lambda: ['A', 'T', 'C', 'G'])

    def probability_distribution(self) -> np.ndarray:
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)

@dataclass
class DNASequence:
    sequence: str
    nucleotides: List[Nucleotide] = field(init=False)
    
    def __post_init__(self):
        self.nucleotides = [Nucleotide(base) for base in list(self.sequence)]
        
    def gc_content(self) -> float:
        return np.mean([nucleotide.name in ['G', 'C'] for nucleotide in self.nucleotides])
    
@dataclass
class RNASequence(DNASequence):
    def __post_init__(self):
        super().__post_init__()
        self.sequence = self.sequence.replace('T', 'U')
        
    def base_pairing(self) -> str:
        return ''.join([n.name if n.name == 'G' else ('C' if n.name == 'A' else ('G' if n.name == 'C' else 'A')) for n in self.nucleotides])

@dataclass
class ProteinSequence:
    sequence: str
    
    def molecular_weight(self) -> float:
        mw = {
            "A": 71.08,
            "R": 156.20,
            "N": 114.10,
            "D": 115.09,
            "C": 103.11,
            "Q": 128.13,
            "E": 129.11,
            "G": 57.05,
            "H": 137.14,
            "I": 113.16,
            "L": 113.16,
            "K": 128.17,
            "M": 131.19,
            "F": 147.18,
            "P": 97.12,
            "S": 87.08,
            "T": 101.11,
            "W": 186.21,
            "Y": 163.18,
            "V": 99.13
        }
        return sum(mw[aa] for aa in self.sequence)
    
    def isoelectric_point(self) -> float:
        pI = {
            "A": 6.05,
            "R": 10.76,
            "N": 8.92,
            "D": 3.45,
            "C": 5.07,
            "Q": 5.92,
            "E": 3.75,
            "G": 5.97,
            "H": 7.19,
            "I": 6.08,
            "L": 6.05,
            "K": 9.74,
            "M": 5.98,
            "F": 5.48,
            "P": 6.30,
            "S": 5.68,
            "T": 5.61,
            "W": 5.89,
            "Y": 5.75,
            "V": 6.05
        }
        return np.mean([pI[aa] for aa in self.sequence])
    
@dataclass
class MolecularBiologist:
    name: str
    
    def run_experiments(self, dna_seq: DNASequence, rna_seq: RNASequence, protein_seq: ProteinSequence) -> None:
        print(f"DNA sequence: {dna_seq.sequence}")
        print(f"GC content of DNA: {dna_seq.gc_content() * 100:.2f}%")
        
        print(f"RNA sequence from transcription: {rna_seq.base_pairing()}")
        print(f"Base pairing in RNA: {rna_seq.sequence}")
                
        print(f"Amino acid chain: {protein_seq.sequence}")
        print(f"Molecular weight of protein: {protein_seq.molecular_weight():.2f} Da")
        print(f"Isoelectric point of protein: {protein_seq.isoelectric_point()}")

def run_demo():
    dna_sequence = DNASequence("ATCGTAGCCTACGTA")
    rna_sequence = RNASequence(dna_sequence.sequence)
    protein_sequence = ProteinSequence("MQFVKL")
    
    mol_bio = MolecularBiologist(name="Dr. Quantum")
    mol_bio.run_experiments(dna_seq=dna_sequence, rna_seq=rna_sequence, protein_seq=protein_sequence)

if __name__ == '__main__':
    run_demo()