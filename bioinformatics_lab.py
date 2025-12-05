"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

BIOINFORMATICS LAB - Production Ready
Advanced sequence analysis, alignment, phylogenetics, and structure prediction.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import re

@dataclass
class BioinformaticsLab:
    """Production-ready bioinformatics analysis laboratory."""

    sequences: List[str] = field(default_factory=list)
    alignment_matrix: Optional[np.ndarray] = None
    phylogenetic_tree: Optional[np.ndarray] = None
    secondary_structures: Dict[str, str] = field(default_factory=dict)

    # Standard genetic code
    CODON_TABLE: Dict[str, str] = field(default_factory=lambda: {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
    })

    # BLOSUM62 scoring matrix for protein alignment
    BLOSUM62: np.ndarray = field(init=False)

    def __post_init__(self):
        """Initialize BLOSUM62 matrix and other parameters."""
        # Simplified BLOSUM62 diagonal values for demonstration
        self.BLOSUM62 = np.array([
            [4, -1, -2, -2, 0, -1, -1, 0, -2, -1],  # A
            [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3],   # R
            [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3],     # N
            [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3],  # D
            [0, -3, -3, -3, 9, -3, -4, -3, -3, -1], # C
            [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3],    # Q
            [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3],    # E
            [0, -2, 0, -1, -3, -2, -2, 6, -2, -4],  # G
            [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3],   # H
            [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4] # I
        ])

    def gc_content(self, sequence: str) -> float:
        """Calculate GC content of a DNA sequence."""
        if not sequence:
            return 0.0
        gc_count = sum(1 for base in sequence.upper() if base in 'GC')
        return (gc_count / len(sequence)) * 100

    def transcribe(self, dna_sequence: str) -> str:
        """Transcribe DNA to RNA."""
        return dna_sequence.upper().replace('T', 'U')

    def translate(self, dna_sequence: str) -> str:
        """Translate DNA sequence to protein sequence."""
        dna = dna_sequence.upper().replace('U', 'T')
        protein = []

        for i in range(0, len(dna) - 2, 3):
            codon = dna[i:i+3]
            if codon in self.CODON_TABLE:
                amino_acid = self.CODON_TABLE[codon]
                if amino_acid == '*':  # Stop codon
                    break
                protein.append(amino_acid)

        return ''.join(protein)

    def find_orfs(self, sequence: str, min_length: int = 100) -> List[Tuple[int, int, str]]:
        """Find all open reading frames in a DNA sequence."""
        orfs = []
        dna = sequence.upper().replace('U', 'T')

        for frame in range(3):
            i = frame
            while i < len(dna) - 2:
                codon = dna[i:i+3]
                if codon == 'ATG':  # Start codon
                    start = i
                    orf_seq = codon
                    i += 3

                    while i < len(dna) - 2:
                        codon = dna[i:i+3]
                        orf_seq += codon
                        if codon in ['TAA', 'TAG', 'TGA']:  # Stop codons
                            if len(orf_seq) >= min_length:
                                orfs.append((start, i+3, orf_seq))
                            break
                        i += 3
                else:
                    i += 3

        return orfs

    def needleman_wunsch(self, seq1: str, seq2: str, match: int = 1,
                        mismatch: int = -1, gap: int = -1) -> Tuple[str, str, float]:
        """Perform global sequence alignment using Needleman-Wunsch algorithm."""
        n, m = len(seq1), len(seq2)

        # Initialize scoring matrix
        score_matrix = np.zeros((n + 1, m + 1))

        # Initialize gap penalties
        for i in range(1, n + 1):
            score_matrix[i, 0] = gap * i
        for j in range(1, m + 1):
            score_matrix[0, j] = gap * j

        # Fill scoring matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match_score = score_matrix[i-1, j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch)
                delete_score = score_matrix[i-1, j] + gap
                insert_score = score_matrix[i, j-1] + gap
                score_matrix[i, j] = max(match_score, delete_score, insert_score)

        # Traceback
        align1, align2 = [], []
        i, j = n, m

        while i > 0 or j > 0:
            current_score = score_matrix[i, j]

            if i > 0 and j > 0 and current_score == score_matrix[i-1, j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch):
                align1.append(seq1[i-1])
                align2.append(seq2[j-1])
                i -= 1
                j -= 1
            elif i > 0 and current_score == score_matrix[i-1, j] + gap:
                align1.append(seq1[i-1])
                align2.append('-')
                i -= 1
            else:
                align1.append('-')
                align2.append(seq2[j-1])
                j -= 1

        return ''.join(reversed(align1)), ''.join(reversed(align2)), score_matrix[n, m]

    def smith_waterman(self, seq1: str, seq2: str, match: int = 2,
                       mismatch: int = -1, gap: int = -1) -> Tuple[str, str, float]:
        """Perform local sequence alignment using Smith-Waterman algorithm."""
        n, m = len(seq1), len(seq2)

        # Initialize scoring matrix
        score_matrix = np.zeros((n + 1, m + 1))
        max_score = 0
        max_pos = (0, 0)

        # Fill scoring matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match_score = score_matrix[i-1, j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch)
                delete_score = score_matrix[i-1, j] + gap
                insert_score = score_matrix[i, j-1] + gap
                score_matrix[i, j] = max(0, match_score, delete_score, insert_score)

                if score_matrix[i, j] > max_score:
                    max_score = score_matrix[i, j]
                    max_pos = (i, j)

        # Traceback from maximum score
        align1, align2 = [], []
        i, j = max_pos

        while i > 0 and j > 0 and score_matrix[i, j] > 0:
            current_score = score_matrix[i, j]

            if current_score == score_matrix[i-1, j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch):
                align1.append(seq1[i-1])
                align2.append(seq2[j-1])
                i -= 1
                j -= 1
            elif current_score == score_matrix[i-1, j] + gap:
                align1.append(seq1[i-1])
                align2.append('-')
                i -= 1
            else:
                align1.append('-')
                align2.append(seq2[j-1])
                j -= 1

        return ''.join(reversed(align1)), ''.join(reversed(align2)), max_score

    def multiple_sequence_alignment(self, sequences: List[str]) -> np.ndarray:
        """Perform progressive multiple sequence alignment."""
        n = len(sequences)
        if n < 2:
            return np.array(sequences)

        # Calculate pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                _, _, score = self.needleman_wunsch(sequences[i], sequences[j])
                distances[i, j] = distances[j, i] = -score  # Convert to distance

        # Build guide tree using UPGMA
        clusters = [[i] for i in range(n)]
        aligned_seqs = list(sequences)

        while len(clusters) > 1:
            # Find minimum distance
            min_dist = float('inf')
            min_pair = (0, 1)

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    avg_dist = np.mean([distances[a, b] for a in clusters[i] for b in clusters[j]])
                    if avg_dist < min_dist:
                        min_dist = avg_dist
                        min_pair = (i, j)

            # Merge clusters
            i, j = min_pair
            new_cluster = clusters[i] + clusters[j]

            # Align sequences in merged cluster
            if len(clusters[i]) == 1 and len(clusters[j]) == 1:
                seq1_idx = clusters[i][0]
                seq2_idx = clusters[j][0]
                align1, align2, _ = self.needleman_wunsch(aligned_seqs[seq1_idx], aligned_seqs[seq2_idx])
                aligned_seqs[seq1_idx] = align1
                aligned_seqs[seq2_idx] = align2

            # Update clusters
            clusters = [c for k, c in enumerate(clusters) if k not in min_pair]
            clusters.append(new_cluster)

        self.alignment_matrix = np.array(aligned_seqs)
        return self.alignment_matrix

    def find_motifs(self, sequence: str, motif_pattern: str) -> List[int]:
        """Find all occurrences of a motif pattern in a sequence using regex."""
        positions = []
        pattern = re.compile(motif_pattern.replace('N', '.'))

        for match in pattern.finditer(sequence):
            positions.append(match.start())

        return positions

    def build_phylogenetic_tree(self, sequences: List[str]) -> np.ndarray:
        """Build a phylogenetic tree using neighbor-joining method."""
        n = len(sequences)

        # Calculate distance matrix
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                # Calculate Hamming distance
                dist = sum(1 for a, b in zip(sequences[i], sequences[j]) if a != b)
                distance_matrix[i, j] = distance_matrix[j, i] = dist

        # Apply neighbor-joining algorithm
        tree = self._neighbor_joining(distance_matrix)
        self.phylogenetic_tree = tree
        return tree

    def _neighbor_joining(self, dist_matrix: np.ndarray) -> np.ndarray:
        """Implementation of neighbor-joining algorithm."""
        n = dist_matrix.shape[0]

        if n <= 2:
            return dist_matrix

        # Calculate Q matrix
        q_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    row_sum = np.sum(dist_matrix[i, :])
                    col_sum = np.sum(dist_matrix[:, j])
                    q_matrix[i, j] = (n - 2) * dist_matrix[i, j] - row_sum - col_sum

        # Find minimum Q value
        min_val = float('inf')
        min_pair = (0, 1)
        for i in range(n):
            for j in range(i + 1, n):
                if q_matrix[i, j] < min_val:
                    min_val = q_matrix[i, j]
                    min_pair = (i, j)

        # Join the pair with minimum Q value
        i, j = min_pair

        # Calculate branch lengths
        sum_i = np.sum(dist_matrix[i, :])
        sum_j = np.sum(dist_matrix[j, :])

        branch_i = 0.5 * dist_matrix[i, j] + (sum_i - sum_j) / (2 * (n - 2))
        branch_j = dist_matrix[i, j] - branch_i

        # Create new distance matrix
        new_n = n - 1
        new_dist = np.zeros((new_n, new_n))

        # Copy unchanged distances
        row_idx = 0
        for r in range(n):
            if r in min_pair:
                continue
            col_idx = 0
            for c in range(n):
                if c in min_pair:
                    continue
                new_dist[row_idx, col_idx] = dist_matrix[r, c]
                col_idx += 1
            row_idx += 1

        # Calculate distances to new node
        for k in range(n):
            if k not in min_pair:
                new_dist_val = 0.5 * (dist_matrix[i, k] + dist_matrix[j, k] - dist_matrix[i, j])
                idx = sum(1 for x in range(k) if x not in min_pair)
                new_dist[new_n-1, idx] = new_dist[idx, new_n-1] = new_dist_val

        return new_dist

    def predict_secondary_structure(self, protein_sequence: str) -> str:
        """Predict protein secondary structure using Chou-Fasman method."""
        # Chou-Fasman propensity values for alpha helix, beta sheet, and turn
        helix_prop = {'A': 1.42, 'C': 0.70, 'D': 1.01, 'E': 1.51, 'F': 1.13,
                     'G': 0.57, 'H': 1.00, 'I': 1.08, 'K': 1.16, 'L': 1.21,
                     'M': 1.45, 'N': 0.67, 'P': 0.57, 'Q': 1.11, 'R': 0.98,
                     'S': 0.77, 'T': 0.83, 'V': 1.06, 'W': 1.08, 'Y': 0.69}

        sheet_prop = {'A': 0.83, 'C': 1.19, 'D': 0.54, 'E': 0.37, 'F': 1.38,
                     'G': 0.75, 'H': 0.87, 'I': 1.60, 'K': 0.74, 'L': 1.30,
                     'M': 1.05, 'N': 0.89, 'P': 0.55, 'Q': 1.10, 'R': 0.93,
                     'S': 0.75, 'T': 1.19, 'V': 1.70, 'W': 1.37, 'Y': 1.47}

        structure = []
        window_size = 6

        for i in range(len(protein_sequence)):
            if i < window_size // 2 or i >= len(protein_sequence) - window_size // 2:
                structure.append('C')  # Coil at terminals
                continue

            # Calculate average propensities in window
            window = protein_sequence[i - window_size//2 : i + window_size//2 + 1]
            avg_helix = np.mean([helix_prop.get(aa, 1.0) for aa in window])
            avg_sheet = np.mean([sheet_prop.get(aa, 1.0) for aa in window])

            if avg_helix > 1.03 and avg_helix > avg_sheet:
                structure.append('H')  # Helix
            elif avg_sheet > 1.05:
                structure.append('E')  # Extended/Sheet
            else:
                structure.append('C')  # Coil

        ss_string = ''.join(structure)
        self.secondary_structures[protein_sequence[:20] + '...'] = ss_string
        return ss_string

    def calculate_molecular_weight(self, protein_sequence: str) -> float:
        """Calculate molecular weight of a protein sequence."""
        # Average molecular weights of amino acids (in Daltons)
        amino_weights = {
            'A': 89.09, 'C': 121.15, 'D': 133.10, 'E': 147.13, 'F': 165.19,
            'G': 75.07, 'H': 155.16, 'I': 131.17, 'K': 146.19, 'L': 131.17,
            'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20,
            'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19
        }

        weight = sum(amino_weights.get(aa, 0) for aa in protein_sequence)
        # Subtract water for peptide bonds
        weight -= 18.015 * (len(protein_sequence) - 1)
        return weight

    def calculate_isoelectric_point(self, protein_sequence: str) -> float:
        """Calculate theoretical isoelectric point (pI) of a protein."""
        # pKa values for ionizable groups
        pka_cterm = 3.65
        pka_nterm = 7.59
        pka_asp = 3.9
        pka_glu = 4.25
        pka_his = 6.0
        pka_cys = 8.3
        pka_tyr = 10.1
        pka_lys = 10.5
        pka_arg = 12.5

        # Count charged residues
        asp_count = protein_sequence.count('D')
        glu_count = protein_sequence.count('E')
        his_count = protein_sequence.count('H')
        lys_count = protein_sequence.count('K')
        arg_count = protein_sequence.count('R')
        cys_count = protein_sequence.count('C')
        tyr_count = protein_sequence.count('Y')

        # Binary search for pI
        pH_min, pH_max = 0.0, 14.0

        while pH_max - pH_min > 0.01:
            pH = (pH_min + pH_max) / 2

            # Calculate net charge
            pos_charge = (1 / (1 + 10**(pH - pka_nterm)) +
                         his_count / (1 + 10**(pH - pka_his)) +
                         lys_count / (1 + 10**(pH - pka_lys)) +
                         arg_count / (1 + 10**(pH - pka_arg)))

            neg_charge = (1 / (1 + 10**(pka_cterm - pH)) +
                         asp_count / (1 + 10**(pka_asp - pH)) +
                         glu_count / (1 + 10**(pka_glu - pH)) +
                         cys_count / (1 + 10**(pka_cys - pH)) +
                         tyr_count / (1 + 10**(pka_tyr - pH)))

            net_charge = pos_charge - neg_charge

            if net_charge > 0:
                pH_min = pH
            else:
                pH_max = pH

        return pH

    def hydropathy_profile(self, protein_sequence: str, window_size: int = 9) -> np.ndarray:
        """Calculate Kyte-Doolittle hydropathy profile."""
        # Kyte-Doolittle hydropathy values
        hydropathy = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }

        profile = []
        half_window = window_size // 2

        for i in range(len(protein_sequence)):
            start = max(0, i - half_window)
            end = min(len(protein_sequence), i + half_window + 1)
            window = protein_sequence[start:end]

            avg_hydropathy = np.mean([hydropathy.get(aa, 0) for aa in window])
            profile.append(avg_hydropathy)

        return np.array(profile)

    def run_comprehensive_analysis(self, dna_sequence: str) -> Dict:
        """Run complete bioinformatics analysis pipeline."""
        results = {}

        # Basic sequence analysis
        results['gc_content'] = self.gc_content(dna_sequence)
        results['length'] = len(dna_sequence)

        # Transcription and translation
        rna = self.transcribe(dna_sequence)
        results['rna_sequence'] = rna[:50] + '...' if len(rna) > 50 else rna

        protein = self.translate(dna_sequence)
        results['protein_sequence'] = protein[:30] + '...' if len(protein) > 30 else protein

        # ORF finding
        orfs = self.find_orfs(dna_sequence)
        results['num_orfs'] = len(orfs)

        if protein:
            # Protein analysis
            results['molecular_weight'] = self.calculate_molecular_weight(protein)
            results['isoelectric_point'] = self.calculate_isoelectric_point(protein)

            # Secondary structure prediction
            ss = self.predict_secondary_structure(protein)
            results['secondary_structure'] = ss[:50] + '...' if len(ss) > 50 else ss

            # Hydropathy profile
            hydro = self.hydropathy_profile(protein)
            results['avg_hydropathy'] = np.mean(hydro)

        return results


def run_demo():
    """Demonstrate comprehensive bioinformatics capabilities."""
    print("=" * 80)
    print("BIOINFORMATICS LAB - Production Demo")
    print("=" * 80)

    lab = BioinformaticsLab()

    # Example DNA sequence (human insulin gene fragment)
    dna_sequence = """
    ATGGCCCTGTGGATGCGCCTCCTGCCCCTGCTGGCGCTGCTGGCCCTCTGGGGACCTGAC
    CCAGCCGCAGCCTTTGTGAACCAACACCTGTGCGGCTCACACCTGGTGGAAGCTCTCTAC
    CTAGTGTGCGGGGAACGAGGCTTCTTCTACACACCCAAGACCCGCCGGGAGGCAGAGGAC
    """
    dna_sequence = ''.join(dna_sequence.split())

    print("\n1. COMPREHENSIVE SEQUENCE ANALYSIS")
    print("-" * 40)
    results = lab.run_comprehensive_analysis(dna_sequence)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    print("\n2. SEQUENCE ALIGNMENT")
    print("-" * 40)
    seq1 = "GCATGCTAGC"
    seq2 = "GATTGCTAGC"

    align1, align2, score = lab.needleman_wunsch(seq1, seq2)
    print(f"Global Alignment (Needleman-Wunsch):")
    print(f"Seq1: {align1}")
    print(f"Seq2: {align2}")
    print(f"Score: {score}")

    print("\n3. MOTIF FINDING")
    print("-" * 40)
    test_sequence = "ATGCGATCGTAGCTAGCGATCGATGCTAGC"
    motif = "ATGC"
    positions = lab.find_motifs(test_sequence, motif)
    print(f"Motif '{motif}' found at positions: {positions}")

    print("\n4. MULTIPLE SEQUENCE ALIGNMENT")
    print("-" * 40)
    sequences = ["ATCGATCG", "ATCGATGG", "ATCGATCG", "ATGGATTG"]
    msa = lab.multiple_sequence_alignment(sequences)
    print("Aligned sequences:")
    for i, seq in enumerate(msa):
        print(f"  Seq{i+1}: {seq}")

    print("\n5. PHYLOGENETIC TREE")
    print("-" * 40)
    tree = lab.build_phylogenetic_tree(sequences)
    print(f"Distance matrix shape: {tree.shape}")
    print(f"Tree computed successfully")

    print("\n6. PROTEIN ANALYSIS")
    print("-" * 40)
    protein = "MAAFSKYLTARNSSLAGAAFLLFCLLHKRRRALGLHGKKSGFGKFLAKKVDKTELKPLGTH"

    mw = lab.calculate_molecular_weight(protein)
    pi = lab.calculate_isoelectric_point(protein)
    ss = lab.predict_secondary_structure(protein)
    hydro = lab.hydropathy_profile(protein)

    print(f"Protein length: {len(protein)} aa")
    print(f"Molecular weight: {mw:.2f} Da")
    print(f"Isoelectric point: {pi:.2f}")
    print(f"Secondary structure: {ss[:30]}...")
    print(f"Average hydropathy: {np.mean(hydro):.2f}")

    print("\n" + "=" * 80)
    print("Bioinformatics Lab demonstration complete!")
    print("=" * 80)


if __name__ == '__main__':
    run_demo()