"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

GENOMICS LAB
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.constants import k, Avogadro, g, c, h, e, pi, physical_constants

@dataclass
class GenomicsLab:
    sequences: list[str] = field(default_factory=list)
    lengths: np.ndarray = field(init=False)

    def __post_init__(self):
        self.lengths = np.array([len(seq) for seq in self.sequences], dtype=np.float64)

    @staticmethod
    def gc_content(sequence: str) -> float:
        return (sequence.count("G") + sequence.count("C")) / len(sequence)

    @property
    def avg_gc_content(self) -> float:
        return np.mean([self.gc_content(seq) for seq in self.sequences])

    @staticmethod
    def melting_temp(sequence: str, kmersize: int = 4) -> float:
        delta_h = 0.0
        delta_s = 0.0

        for i in range(0, len(sequence) - (kmersize - 1)):
            kmer = sequence[i:i+kmersize]
            if kmer.count("A") == kmersize or kmer.count("T") == kmersize:
                delta_h += physical_constants['adenine-guanine double bond entropy'][0] * len(kmer)
            else:
                delta_h -= sum(physical_constants[f'{base1}{base2} double bond enthalpy'][0]
                               for base1 in kmer[::2] for base2 in kmer[1::2])

        for i in range(len(sequence) - (kmersize - 1)):
            kmer = sequence[i:i+kmersize]
            if kmer.count("A") == kmersize or kmer.count("T") == kmersize:
                delta_s += physical_constants['adenine-guanine double bond entropy'][0] * len(kmer)
            else:
                for base1 in kmer[::2]:
                    for base2 in kmer[1::2]:
                        if base1 != base2 and (base1 == "A" or base1 == "T"):
                            delta_s += physical_constants[f'{base1}{base2} double bond entropy'][0]
                        elif base1 != base2:
                            delta_s -= physical_constants[f'{base1}{base2} double bond entropy'][0]

        return (-delta_h / (k * np.log(2))) + 46.5

    @property
    def avg_melting_temp(self) -> float:
        return np.mean([self.melting_temp(seq) for seq in self.sequences])

def run_demo():
    lab = GenomicsLab(["AGTCGACTGA", "GGCCCGGTAA", "ATATTA"])
    print(f"Sequences: {lab.sequences}")
    print(f"Avg. GC Content: {lab.avg_gc_content:.4f}")
    print(f"Avg. Melting Temp: {lab.avg_melting_temp:.2f}")

if __name__ == '__main__':
    run_demo()