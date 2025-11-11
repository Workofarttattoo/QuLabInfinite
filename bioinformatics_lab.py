"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

BIOINFORMATICS LAB
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.constants import k, Avogadro, g, c, h, e, pi, physical_constants

@dataclass
class BioinformaticsLab:
    sequence: str = ""
    gc_content: float = 0.0
    molecular_weight: float = 0.0
    _amino_acids_weights: np.ndarray = field(init=False)

    def __post_init__(self):
        self._amino_acids_weights = np.array([
            physical_constants["alanine mass"][0],
            physical_constants["arginine mass"][0],
            physical_constants["asparagine mass"][0],
            physical_constants["aspartic acid mass"][0],
            physical_constants["cysteine mass"][0],
            physical_constants["glutamic acid mass"][0],
            physical_constants["glutamine mass"][0],
            physical_constants["glycine mass"][0],
            physical_constants["histidine mass"][0],
            physical_constants["isoleucine mass"][0],
            physical_constants["leucine mass"][0],
            physical_constants["lysine mass"][0],
            physical_constants["methionine mass"][0],
            physical_constants["phenylalanine mass"][0],
            physical_constants["proline mass"][0],
            physical_constants["serine mass"][0],
            physical_constants["threonine mass"][0],
            physical_constants["tryptophan mass"][0],
            physical_constants["tyrosine mass"][0],
            physical_constants["valine mass"][0]
        ], dtype=np.float64)
    
    def gc_content_calc(self) -> float:
        gc_count = sum([1 for base in self.sequence if base.upper() == "G" or base.upper() == "C"])
        return gc_count / len(self.sequence)

    def molecular_weight_calc(self, sequence: str) -> float:
        weight_sum = 0.0
        codon_table = {
            'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
            'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
            'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
            'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R', 
            # ... complete codon table ...
        }
        
        for i in range(0, len(sequence), 3):
            try:
                amino_acid = codon_table[sequence[i:i+3]]
                weight_sum += self._amino_acids_weights[list(codon_table.keys()).index(amino_acid)]
            except KeyError:
                pass
        
        return weight_sum

    def analyze_sequence(self, sequence: str) -> None:
        self.sequence = sequence
        self.gc_content = self.gc_content_calc()
        self.molecular_weight = self.molecular_weight_calc(sequence)

def run_demo() -> None:
    lab = BioinformaticsLab("ATCGTAGC")
    lab.analyze_sequence("ATCGTACGAAAAGGGGGTTTTTTCCCCCCC")
    print(f"GC Content: {lab.gc_content}")
    print(f"Molecular Weight: {lab.molecular_weight}")

if __name__ == '__main__':
    run_demo()