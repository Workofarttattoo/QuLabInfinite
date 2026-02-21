"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

MOLECULAR BIOLOGY LAB
Production-ready molecular biology algorithms for DNA/RNA analysis, PCR simulation,
CRISPR design, and gene expression modeling.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import math
from scipy import optimize
from scipy.special import binom

# Genetic code table for translation
GENETIC_CODE = {
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
}

# Restriction enzyme recognition sites (common enzymes)
RESTRICTION_ENZYMES = {
    'EcoRI': 'GAATTC',
    'BamHI': 'GGATCC',
    'HindIII': 'AAGCTT',
    'PstI': 'CTGCAG',
    'SmaI': 'CCCGGG',
    'XbaI': 'TCTAGA',
    'SacI': 'GAGCTC',
    'KpnI': 'GGTACC',
    'BglII': 'AGATCT',
    'NotI': 'GCGGCCGC'
}

@dataclass
class DNASequence:
    """Comprehensive DNA sequence analysis class"""
    sequence: str
    name: str = "DNA"
    circular: bool = False

    def __post_init__(self):
        self.sequence = self.sequence.upper()
        if not all(base in 'ATCGN' for base in self.sequence):
            raise ValueError("Invalid DNA sequence - contains non-DNA characters")

    def gc_content(self) -> float:
        """Calculate GC content percentage"""
        gc_count = self.sequence.count('G') + self.sequence.count('C')
        return (gc_count / len(self.sequence)) * 100 if self.sequence else 0

    def melting_temperature_simple(self, na_concentration: float = 50.0) -> float:
        """Calculate melting temperature using Wallace rule and salt correction

        Args:
            na_concentration: Sodium concentration in mM
        Returns:
            Melting temperature in Celsius
        """
        length = len(self.sequence)
        gc = self.gc_content() / 100.0

        if length < 14:
            # Wallace rule for short oligos
            tm = 2 * (self.sequence.count('A') + self.sequence.count('T')) + \
                 4 * (self.sequence.count('G') + self.sequence.count('C'))
        else:
            # Long sequence formula
            tm = 81.5 + 16.6 * math.log10(na_concentration/1000.0) + 41 * gc - 675/length

        return tm

    def melting_temperature_nn(self, primer_conc: float = 250.0, na_conc: float = 50.0) -> float:
        """Calculate melting temperature using nearest-neighbor thermodynamics

        Args:
            primer_conc: Primer concentration in nM
            na_conc: Sodium concentration in mM
        Returns:
            Melting temperature in Celsius
        """
        # Nearest neighbor parameters (ΔH in kcal/mol, ΔS in cal/mol·K)
        nn_params = {
            'AA': (-7.9, -22.2), 'AT': (-7.2, -20.4), 'AG': (-7.8, -21.0), 'AC': (-8.4, -22.4),
            'TA': (-7.2, -21.3), 'TT': (-7.9, -22.2), 'TG': (-8.5, -22.7), 'TC': (-8.2, -22.2),
            'GA': (-8.2, -22.2), 'GT': (-8.4, -22.4), 'GG': (-8.0, -19.9), 'GC': (-9.8, -24.4),
            'CA': (-8.5, -22.7), 'CT': (-7.8, -21.0), 'CG': (-10.6, -27.2), 'CC': (-8.0, -19.9)
        }

        dh_total = 0.6  # Initiation enthalpy
        ds_total = -9.0  # Initiation entropy

        # Calculate NN contributions
        for i in range(len(self.sequence) - 1):
            dinuc = self.sequence[i:i+2]
            if dinuc in nn_params:
                dh, ds = nn_params[dinuc]
                dh_total += dh
                ds_total += ds

        # Salt correction
        ds_total += 0.368 * (len(self.sequence) - 1) * math.log(na_conc / 1000.0)

        # Calculate Tm
        R = 1.987  # cal/mol·K
        ct = primer_conc * 1e-9  # Convert to M
        tm = (dh_total * 1000) / (ds_total + R * math.log(ct)) - 273.15

        return tm

    def reverse_complement(self) -> str:
        """Generate reverse complement sequence"""
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        return ''.join(complement[base] for base in self.sequence[::-1])

    def transcribe(self) -> 'RNASequence':
        """Transcribe DNA to RNA"""
        rna_seq = self.sequence.replace('T', 'U')
        return RNASequence(rna_seq, f"{self.name}_RNA")

    def find_orfs(self, min_length: int = 100) -> List[Tuple[int, int, str]]:
        """Find open reading frames

        Args:
            min_length: Minimum ORF length in nucleotides
        Returns:
            List of (start, end, frame) tuples
        """
        orfs = []
        start_codons = ['ATG']
        stop_codons = ['TAA', 'TAG', 'TGA']

        for frame in range(3):
            i = frame
            while i < len(self.sequence) - 2:
                codon = self.sequence[i:i+3]
                if codon in start_codons:
                    # Look for stop codon
                    for j in range(i+3, len(self.sequence)-2, 3):
                        stop_codon = self.sequence[j:j+3]
                        if stop_codon in stop_codons:
                            if j - i >= min_length:
                                orfs.append((i, j+3, frame))
                            break
                i += 3

        return orfs

    def restriction_map(self) -> Dict[str, List[int]]:
        """Map restriction enzyme cut sites"""
        sites = {}
        for enzyme, pattern in RESTRICTION_ENZYMES.items():
            positions = []
            for i in range(len(self.sequence) - len(pattern) + 1):
                if self.sequence[i:i+len(pattern)] == pattern:
                    positions.append(i)
            if positions:
                sites[enzyme] = positions
        return sites

    def cpg_islands(self, window: int = 200, min_length: int = 200,
                    gc_threshold: float = 50.0, oe_threshold: float = 0.6) -> List[Tuple[int, int]]:
        """Find CpG islands

        Args:
            window: Sliding window size
            min_length: Minimum island length
            gc_threshold: Minimum GC percentage
            oe_threshold: Minimum observed/expected CpG ratio
        Returns:
            List of (start, end) positions of CpG islands
        """
        islands = []

        for i in range(0, len(self.sequence) - window + 1):
            subseq = self.sequence[i:i+window]
            gc = (subseq.count('G') + subseq.count('C')) / window * 100

            # Calculate observed/expected CpG ratio
            cpg_obs = subseq.count('CG')
            c_count = subseq.count('C')
            g_count = subseq.count('G')
            cpg_exp = (c_count * g_count) / window if window > 0 else 0
            oe_ratio = cpg_obs / cpg_exp if cpg_exp > 0 else 0

            if gc >= gc_threshold and oe_ratio >= oe_threshold:
                # Extend island
                if islands and i <= islands[-1][1]:
                    islands[-1] = (islands[-1][0], i + window)
                else:
                    islands.append((i, i + window))

        # Filter by minimum length
        islands = [(start, end) for start, end in islands if end - start >= min_length]
        return islands


@dataclass
class RNASequence:
    """RNA sequence analysis class"""
    sequence: str
    name: str = "RNA"

    def __post_init__(self):
        self.sequence = self.sequence.upper()
        if not all(base in 'AUCGN' for base in self.sequence):
            raise ValueError("Invalid RNA sequence")

    def translate(self, start_pos: int = 0) -> str:
        """Translate RNA to protein sequence"""
        # Convert RNA codons to DNA for lookup
        dna_seq = self.sequence.replace('U', 'T')
        protein = []

        for i in range(start_pos, len(dna_seq) - 2, 3):
            codon = dna_seq[i:i+3]
            if codon in GENETIC_CODE:
                amino_acid = GENETIC_CODE[codon]
                if amino_acid == '*':
                    break
                protein.append(amino_acid)

        return ''.join(protein)

    def secondary_structure_energy(self) -> float:
        """Calculate minimum free energy for RNA secondary structure
        Using simplified Nussinov algorithm
        """
        n = len(self.sequence)
        if n < 4:
            return 0

        # Simplified base pair energies (kcal/mol)
        bp_energy = {
            ('A', 'U'): -2.0, ('U', 'A'): -2.0,
            ('G', 'C'): -3.0, ('C', 'G'): -3.0,
            ('G', 'U'): -1.0, ('U', 'G'): -1.0
        }

        # Dynamic programming matrix
        dp = [[0.0] * n for _ in range(n)]

        for length in range(4, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1

                # No pairing at j
                dp[i][j] = dp[i][j-1]

                # Try pairing j with k
                for k in range(i, j - 3):
                    pair = (self.sequence[k], self.sequence[j])
                    if pair in bp_energy:
                        energy = bp_energy[pair]
                        if k > i:
                            energy += dp[i][k-1]
                        if k < j - 1:
                            energy += dp[k+1][j-1]
                        dp[i][j] = min(dp[i][j], energy)

        return dp[0][n-1] if n > 0 else 0


@dataclass
class Primer:
    """PCR primer design and analysis"""
    sequence: str
    name: str = "Primer"

    def __post_init__(self):
        self.sequence = self.sequence.upper()
        self.dna = DNASequence(self.sequence, self.name)

    def calculate_properties(self) -> Dict[str, float]:
        """Calculate comprehensive primer properties"""
        props = {
            'length': len(self.sequence),
            'gc_content': self.dna.gc_content(),
            'tm_simple': self.dna.melting_temperature_simple(),
            'tm_nn': self.dna.melting_temperature_nn(),
        }

        # Check for secondary structures
        props['hairpin_tm'] = self._hairpin_tm()
        props['self_dimer_tm'] = self._self_dimer_tm()
        props['gc_clamp'] = self._gc_clamp_score()
        props['complexity'] = self._sequence_complexity()

        return props

    def _hairpin_tm(self) -> float:
        """Calculate hairpin melting temperature"""
        min_stem = 3
        max_tm = 0

        for i in range(len(self.sequence) - 2 * min_stem):
            for j in range(i + min_stem, len(self.sequence) - min_stem + 1):
                stem1 = self.sequence[i:j]
                stem2_rev = self.sequence[j:j+len(stem1)][::-1]

                # Check complementarity
                matches = sum(1 for b1, b2 in zip(stem1, stem2_rev)
                            if (b1 == 'A' and b2 == 'T') or (b1 == 'T' and b2 == 'A') or
                               (b1 == 'G' and b2 == 'C') or (b1 == 'C' and b2 == 'G'))

                if matches >= len(stem1) * 0.7:  # 70% match threshold
                    tm = 4 * matches + 2 * (len(stem1) - matches)
                    max_tm = max(max_tm, tm)

        return max_tm

    def _self_dimer_tm(self) -> float:
        """Calculate self-dimer melting temperature"""
        rev_comp = self.dna.reverse_complement()
        max_tm = 0

        for i in range(1, len(self.sequence)):
            overlap1 = self.sequence[:i]
            overlap2 = rev_comp[-i:]

            matches = sum(1 for b1, b2 in zip(overlap1, overlap2) if b1 == b2)
            if matches >= 3:  # Minimum 3 bp match
                tm = 4 * matches
                max_tm = max(max_tm, tm)

        return max_tm

    def _gc_clamp_score(self) -> int:
        """Calculate GC clamp score (G or C at 3' end)"""
        last_5 = self.sequence[-5:] if len(self.sequence) >= 5 else self.sequence
        return sum(1 for base in last_5 if base in 'GC')

    def _sequence_complexity(self) -> float:
        """Calculate sequence complexity (entropy)"""
        if not self.sequence:
            return 0

        freq = {base: self.sequence.count(base) / len(self.sequence)
                for base in 'ATCG'}
        entropy = -sum(f * math.log2(f) for f in freq.values() if f > 0)
        return entropy / 2.0  # Normalize to 0-1


class PCRSimulator:
    """Simulate PCR amplification kinetics"""

    def __init__(self, template: DNASequence, forward_primer: Primer, reverse_primer: Primer):
        self.template = template
        self.forward = forward_primer
        self.reverse = reverse_primer
        self.amplicon_length = 0
        self._find_amplicon()

    def _find_amplicon(self):
        """Find amplicon region"""
        # Find primer binding sites
        fwd_pos = self.template.sequence.find(self.forward.sequence)

        # Reverse primer binds to complement strand
        rev_comp = self.reverse.dna.reverse_complement()
        rev_pos = self.template.sequence.find(rev_comp)

        if fwd_pos >= 0 and rev_pos >= 0 and rev_pos > fwd_pos:
            self.amplicon_length = rev_pos + len(rev_comp) - fwd_pos

    def simulate_amplification(self, cycles: int = 30, efficiency: float = 0.95,
                             initial_copies: float = 1e6) -> np.ndarray:
        """Simulate PCR amplification curve

        Args:
            cycles: Number of PCR cycles
            efficiency: Amplification efficiency (0-1)
            initial_copies: Initial template copy number
        Returns:
            Copy numbers at each cycle
        """
        if self.amplicon_length == 0:
            return np.zeros(cycles + 1)

        copies = np.zeros(cycles + 1)
        copies[0] = initial_copies

        for cycle in range(1, cycles + 1):
            # Exponential phase with efficiency
            if copies[cycle-1] < 1e12:  # Plateau phase threshold
                copies[cycle] = copies[cycle-1] * (1 + efficiency)
            else:
                # Plateau phase - limited by reagents
                copies[cycle] = copies[cycle-1] + (1e12 - copies[cycle-1]) * 0.1

        return copies

    def calculate_ct(self, threshold: float = 1e8, initial_copies: float = 1e6,
                     efficiency: float = 0.95) -> float:
        """Calculate threshold cycle (Ct) for qPCR

        Args:
            threshold: Fluorescence threshold for Ct
            initial_copies: Initial template copies
            efficiency: PCR efficiency
        Returns:
            Ct value
        """
        if self.amplicon_length == 0:
            return float('inf')

        # Ct = -1/log(1+E) * log(threshold/initial)
        if efficiency > 0 and initial_copies > 0:
            ct = math.log(threshold / initial_copies) / math.log(1 + efficiency)
            return max(0, ct)
        return float('inf')


class CRISPRDesigner:
    """CRISPR-Cas9 guide RNA design and scoring"""

    def __init__(self, target_sequence: str, pam: str = 'NGG'):
        self.target = DNASequence(target_sequence)
        self.pam = pam
        self.guides = []

    def find_guides(self, guide_length: int = 20) -> List[Dict]:
        """Find all possible guide RNAs with PAM sites"""
        guides = []
        pam_pattern = self.pam.replace('N', '.')

        # Search both strands
        for strand, seq in [('sense', self.target.sequence),
                           ('antisense', self.target.reverse_complement())]:
            for i in range(len(seq) - guide_length - len(self.pam) + 1):
                # Check for PAM
                potential_pam = seq[i + guide_length:i + guide_length + len(self.pam)]

                # Simple PAM matching (N = any nucleotide)
                pam_match = all(p == 'N' or p == n for p, n in zip(self.pam, potential_pam))

                if pam_match:
                    guide_seq = seq[i:i + guide_length]
                    guides.append({
                        'sequence': guide_seq,
                        'position': i,
                        'strand': strand,
                        'pam': potential_pam
                    })

        self.guides = guides
        return guides

    def score_guide(self, guide: Dict) -> Dict[str, float]:
        """Score guide RNA for on-target efficiency and off-target risk

        Based on Doench et al. 2016 scoring algorithm (simplified)
        """
        seq = guide['sequence']
        scores = {}

        # On-target score components
        scores['gc_content'] = self._score_gc_content(seq)
        scores['nucleotide_features'] = self._score_nucleotides(seq)
        scores['melting_temp'] = self._score_melting_temp(seq)
        scores['secondary_structure'] = self._score_secondary_structure(seq)

        # Calculate overall on-target score (0-100)
        scores['on_target'] = np.mean([
            scores['gc_content'] * 100,
            scores['nucleotide_features'] * 100,
            scores['melting_temp'] * 100,
            (1 - scores['secondary_structure']) * 100
        ])

        # Off-target risk (simplified - based on sequence complexity)
        scores['off_target_risk'] = self._calculate_off_target_risk(seq)

        # Overall score
        scores['overall'] = scores['on_target'] * (1 - scores['off_target_risk'])

        return scores

    def _score_gc_content(self, seq: str) -> float:
        """Score based on GC content (optimal: 40-60%)"""
        gc = (seq.count('G') + seq.count('C')) / len(seq)
        if 0.4 <= gc <= 0.6:
            return 1.0
        elif gc < 0.4:
            return gc / 0.4
        else:
            return (1 - gc) / 0.4

    def _score_nucleotides(self, seq: str) -> float:
        """Score based on nucleotide preferences at specific positions"""
        score = 0

        # Position-specific nucleotide preferences (simplified)
        if seq[0] == 'G':  # G at position 1
            score += 0.2
        if seq[-1] in 'GC':  # G or C at position 20
            score += 0.2
        if seq[19] == 'G':  # G before PAM
            score += 0.1

        # Avoid poly-T (TTTT) which terminates RNA pol III
        if 'TTTT' not in seq:
            score += 0.3

        # Penalize homopolymers
        for base in 'ATCG':
            if base * 4 in seq:
                score -= 0.1

        return max(0, min(1, score + 0.5))

    def _score_melting_temp(self, seq: str) -> float:
        """Score based on melting temperature distribution"""
        dna = DNASequence(seq)
        tm = dna.melting_temperature_simple()

        # Optimal Tm range: 50-70°C
        if 50 <= tm <= 70:
            return 1.0
        elif tm < 50:
            return tm / 50
        else:
            return max(0, 1 - (tm - 70) / 30)

    def _score_secondary_structure(self, seq: str) -> float:
        """Estimate secondary structure propensity"""
        # Convert to RNA
        rna = seq.replace('T', 'U')
        rna_seq = RNASequence(rna)

        # Calculate folding energy
        energy = abs(rna_seq.secondary_structure_energy())

        # Lower energy = more stable structure = worse for CRISPR
        # Normalize to 0-1 (higher = worse)
        return min(1, energy / 20)

    def _calculate_off_target_risk(self, seq: str) -> float:
        """Estimate off-target risk based on sequence features"""
        risk = 0

        # Penalize low complexity sequences
        unique_kmers = set()
        for k in [3, 4, 5]:
            for i in range(len(seq) - k + 1):
                unique_kmers.add(seq[i:i+k])

        max_kmers = sum(len(seq) - k + 1 for k in [3, 4, 5])
        complexity = len(unique_kmers) / max_kmers

        if complexity < 0.7:
            risk += 0.3

        # Penalize common motifs
        common_motifs = ['GGG', 'CCC', 'AAAA', 'TTTT']
        for motif in common_motifs:
            if motif in seq:
                risk += 0.1

        return min(1, risk)


class GeneExpressionModel:
    """Model gene expression dynamics"""

    def __init__(self, promoter_strength: float = 1.0, rbs_strength: float = 1.0):
        """
        Args:
            promoter_strength: Relative promoter strength (0-10)
            rbs_strength: Ribosome binding site strength (0-10)
        """
        self.promoter = min(10, max(0, promoter_strength))
        self.rbs = min(10, max(0, rbs_strength))

        # Rate constants (per hour)
        self.k_transcription = 0.5 * self.promoter  # mRNA/hour
        self.k_translation = 10.0 * self.rbs  # protein/mRNA/hour
        self.k_mrna_decay = 0.2  # 1/hour
        self.k_protein_decay = 0.05  # 1/hour

    def simulate(self, time_hours: float = 24, dt: float = 0.1) -> Dict[str, np.ndarray]:
        """Simulate gene expression over time

        Args:
            time_hours: Simulation duration
            dt: Time step
        Returns:
            Dictionary with time, mRNA, and protein arrays
        """
        steps = int(time_hours / dt)
        time = np.linspace(0, time_hours, steps)

        mrna = np.zeros(steps)
        protein = np.zeros(steps)

        # Initial conditions
        mrna[0] = 0
        protein[0] = 0

        # ODE simulation using Euler method
        for i in range(1, steps):
            # mRNA dynamics
            dmrna_dt = self.k_transcription - self.k_mrna_decay * mrna[i-1]
            mrna[i] = mrna[i-1] + dmrna_dt * dt

            # Protein dynamics
            dprotein_dt = self.k_translation * mrna[i-1] - self.k_protein_decay * protein[i-1]
            protein[i] = protein[i-1] + dprotein_dt * dt

        return {
            'time': time,
            'mrna': mrna,
            'protein': protein
        }

    def steady_state(self) -> Dict[str, float]:
        """Calculate steady-state expression levels"""
        mrna_ss = self.k_transcription / self.k_mrna_decay
        protein_ss = (self.k_translation * self.k_transcription) / \
                     (self.k_mrna_decay * self.k_protein_decay)

        return {
            'mrna': mrna_ss,
            'protein': protein_ss
        }

    def response_time(self) -> Dict[str, float]:
        """Calculate characteristic response times"""
        return {
            'mrna_halflife': math.log(2) / self.k_mrna_decay,
            'protein_halflife': math.log(2) / self.k_protein_decay,
            'mrna_response': 1 / self.k_mrna_decay,
            'protein_response': 1 / self.k_protein_decay
        }


class MolecularBiologyLab:
    """Main molecular biology laboratory interface"""

    def __init__(self):
        self.experiments = []
        self.results = {}

    def analyze_sequence(self, sequence: str, sequence_type: str = 'DNA') -> Dict:
        """Comprehensive sequence analysis"""
        results = {}

        if sequence_type == 'DNA':
            dna = DNASequence(sequence)
            results['gc_content'] = dna.gc_content()
            results['tm_simple'] = dna.melting_temperature_simple()
            results['tm_nn'] = dna.melting_temperature_nn()
            results['reverse_complement'] = dna.reverse_complement()
            results['orfs'] = dna.find_orfs()
            results['restriction_sites'] = dna.restriction_map()
            results['cpg_islands'] = dna.cpg_islands()

            # Transcription and translation
            rna = dna.transcribe()
            results['rna_sequence'] = rna.sequence
            results['protein'] = rna.translate()

        elif sequence_type == 'RNA':
            rna = RNASequence(sequence)
            results['protein'] = rna.translate()
            results['folding_energy'] = rna.secondary_structure_energy()

        self.results['sequence_analysis'] = results
        return results

    def design_primers(self, template: str, target_tm: float = 60.0,
                      primer_length_range: Tuple[int, int] = (18, 25)) -> Dict:
        """Design optimal PCR primers"""
        dna = DNASequence(template)
        best_primers = {'forward': None, 'reverse': None}
        best_scores = {'forward': 0, 'reverse': 0}

        # Design forward primers
        for length in range(primer_length_range[0], primer_length_range[1] + 1):
            for i in range(len(template) - length + 1):
                primer_seq = template[i:i+length]
                primer = Primer(primer_seq, f"F_{i}_{length}")
                props = primer.calculate_properties()

                # Score based on Tm proximity and other factors
                tm_diff = abs(props['tm_nn'] - target_tm)
                score = 100 - tm_diff * 2  # Tm score
                score -= props['hairpin_tm'] * 0.5  # Penalize hairpins
                score -= props['self_dimer_tm'] * 0.5  # Penalize dimers
                score += props['gc_clamp'] * 5  # Reward GC clamp
                score += props['complexity'] * 20  # Reward complexity

                if score > best_scores['forward']:
                    best_scores['forward'] = score
                    best_primers['forward'] = {
                        'sequence': primer_seq,
                        'position': i,
                        'properties': props,
                        'score': score
                    }

        # Design reverse primers (from the end)
        for length in range(primer_length_range[0], primer_length_range[1] + 1):
            for i in range(len(template) - length, -1, -1):
                primer_seq = dna.reverse_complement()[-(i+length):-i if i > 0 else None]
                if len(primer_seq) >= primer_length_range[0]:
                    primer = Primer(primer_seq, f"R_{i}_{length}")
                    props = primer.calculate_properties()

                    tm_diff = abs(props['tm_nn'] - target_tm)
                    score = 100 - tm_diff * 2
                    score -= props['hairpin_tm'] * 0.5
                    score -= props['self_dimer_tm'] * 0.5
                    score += props['gc_clamp'] * 5
                    score += props['complexity'] * 20

                    if score > best_scores['reverse']:
                        best_scores['reverse'] = score
                        best_primers['reverse'] = {
                            'sequence': primer_seq,
                            'position': len(template) - i - length,
                            'properties': props,
                            'score': score
                        }

        self.results['primer_design'] = best_primers
        return best_primers

    def simulate_pcr(self, template: str, forward_primer: str, reverse_primer: str,
                     cycles: int = 30) -> Dict:
        """Simulate PCR amplification"""
        dna = DNASequence(template)
        fwd = Primer(forward_primer)
        rev = Primer(reverse_primer)

        simulator = PCRSimulator(dna, fwd, rev)
        amplification = simulator.simulate_amplification(cycles)
        ct_value = simulator.calculate_ct()

        results = {
            'amplicon_length': simulator.amplicon_length,
            'amplification_curve': amplification.tolist(),
            'final_yield': amplification[-1],
            'ct_value': ct_value,
            'fold_amplification': amplification[-1] / amplification[0] if amplification[0] > 0 else 0
        }

        self.results['pcr_simulation'] = results
        return results

    def design_crispr_guides(self, target: str, top_n: int = 5) -> List[Dict]:
        """Design and score CRISPR guide RNAs"""
        designer = CRISPRDesigner(target)
        guides = designer.find_guides()

        # Score all guides
        for guide in guides:
            guide['scores'] = designer.score_guide(guide)

        # Sort by overall score
        guides.sort(key=lambda g: g['scores']['overall'], reverse=True)

        # Return top N guides
        top_guides = guides[:top_n]
        self.results['crispr_guides'] = top_guides
        return top_guides

    def model_gene_expression(self, promoter_strength: float = 5.0,
                            rbs_strength: float = 5.0,
                            duration_hours: float = 24) -> Dict:
        """Model gene expression dynamics"""
        model = GeneExpressionModel(promoter_strength, rbs_strength)

        # Run simulation
        dynamics = model.simulate(duration_hours)
        steady_state = model.steady_state()
        response = model.response_time()

        results = {
            'dynamics': dynamics,
            'steady_state': steady_state,
            'response_times': response,
            'parameters': {
                'promoter_strength': promoter_strength,
                'rbs_strength': rbs_strength,
                'k_transcription': model.k_transcription,
                'k_translation': model.k_translation
            }
        }

        self.results['gene_expression'] = results
        return results

    def enzyme_kinetics(self, substrate_conc: np.ndarray, vmax: float = 100.0,
                       km: float = 10.0) -> Dict:
        """Calculate Michaelis-Menten enzyme kinetics"""
        # Michaelis-Menten equation
        velocity = (vmax * substrate_conc) / (km + substrate_conc)

        # Lineweaver-Burk transformation
        lb_x = 1 / substrate_conc[substrate_conc > 0]
        lb_y = 1 / velocity[substrate_conc > 0]

        # Hill equation for cooperativity
        n_hill = 2.5  # Hill coefficient
        velocity_hill = (vmax * substrate_conc**n_hill) / (km**n_hill + substrate_conc**n_hill)

        results = {
            'substrate': substrate_conc.tolist(),
            'velocity': velocity.tolist(),
            'velocity_hill': velocity_hill.tolist(),
            'vmax': vmax,
            'km': km,
            'kcat': vmax / 1e-6,  # Assuming 1 μM enzyme
            'lineweaver_burk': {
                'x': lb_x.tolist(),
                'y': lb_y.tolist()
            }
        }

        self.results['enzyme_kinetics'] = results
        return results

    def calculate_molecular_weight(self, sequence: str, molecule_type: str = 'DNA') -> float:
        """Calculate molecular weight of DNA, RNA, or protein"""
        if molecule_type == 'DNA':
            # Average MW of dNTPs (with phosphate)
            mw_per_base = {'A': 331.2, 'T': 322.2, 'G': 347.2, 'C': 307.2, 'N': 327.0}
            mw = sum(mw_per_base.get(base, 327.0) for base in sequence)

        elif molecule_type == 'RNA':
            # Average MW of rNTPs
            mw_per_base = {'A': 347.2, 'U': 324.2, 'G': 363.2, 'C': 323.2, 'N': 339.5}
            mw = sum(mw_per_base.get(base, 339.5) for base in sequence)

        elif molecule_type == 'protein':
            # Amino acid molecular weights
            aa_mw = {
                'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
                'Q': 146.2, 'E': 147.1, 'G': 75.1, 'H': 155.2, 'I': 131.2,
                'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
                'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
            }
            mw = sum(aa_mw.get(aa, 128.0) for aa in sequence)
            mw -= 18.0 * (len(sequence) - 1)  # Subtract water for peptide bonds

        else:
            mw = 0

        return mw


def run_demo():
    """Demonstrate molecular biology lab capabilities"""
    print("MOLECULAR BIOLOGY LAB - Production Demo")
    print("=" * 60)

    lab = MolecularBiologyLab()

    # 1. DNA sequence analysis
    print("\n1. DNA SEQUENCE ANALYSIS")
    print("-" * 40)
    test_dna = "ATGGCAAGAAAAGAAAAATCAGATCTTCACAAGTATGCCGAGAATGTGCAAAACTTGGAATTCTAG"
    analysis = lab.analyze_sequence(test_dna, 'DNA')
    print(f"Sequence length: {len(test_dna)} bp")
    print(f"GC content: {analysis['gc_content']:.1f}%")
    print(f"Melting temperature (simple): {analysis['tm_simple']:.1f}°C")
    print(f"Melting temperature (NN): {analysis['tm_nn']:.1f}°C")
    print(f"Protein translation: {analysis['protein'][:20]}...")
    print(f"ORFs found: {len(analysis['orfs'])}")
    print(f"Restriction sites: {list(analysis['restriction_sites'].keys())[:5]}")

    # 2. Primer design
    print("\n2. PCR PRIMER DESIGN")
    print("-" * 40)
    primers = lab.design_primers(test_dna[:100], target_tm=60)
    if primers['forward']:
        print(f"Forward primer: {primers['forward']['sequence']}")
        print(f"  Position: {primers['forward']['position']}")
        print(f"  Tm: {primers['forward']['properties']['tm_nn']:.1f}°C")
        print(f"  Score: {primers['forward']['score']:.1f}")
    if primers['reverse']:
        print(f"Reverse primer: {primers['reverse']['sequence']}")
        print(f"  Position: {primers['reverse']['position']}")
        print(f"  Tm: {primers['reverse']['properties']['tm_nn']:.1f}°C")
        print(f"  Score: {primers['reverse']['score']:.1f}")

    # 3. PCR simulation
    print("\n3. PCR SIMULATION")
    print("-" * 40)
    if primers['forward'] and primers['reverse']:
        pcr = lab.simulate_pcr(
            test_dna,
            primers['forward']['sequence'],
            primers['reverse']['sequence'],
            cycles=30
        )
        print(f"Amplicon length: {pcr['amplicon_length']} bp")
        print(f"Ct value: {pcr['ct_value']:.1f}")
        print(f"Final yield: {pcr['final_yield']:.2e} copies")
        print(f"Fold amplification: {pcr['fold_amplification']:.2e}x")

    # 4. CRISPR guide design
    print("\n4. CRISPR GUIDE RNA DESIGN")
    print("-" * 40)
    guides = lab.design_crispr_guides(test_dna, top_n=3)
    for i, guide in enumerate(guides, 1):
        print(f"Guide {i}: {guide['sequence']}")
        print(f"  Position: {guide['position']}, Strand: {guide['strand']}")
        print(f"  PAM: {guide['pam']}")
        print(f"  On-target score: {guide['scores']['on_target']:.1f}")
        print(f"  Off-target risk: {guide['scores']['off_target_risk']:.2f}")
        print(f"  Overall score: {guide['scores']['overall']:.1f}")

    # 5. Gene expression modeling
    print("\n5. GENE EXPRESSION MODELING")
    print("-" * 40)
    expr = lab.model_gene_expression(promoter_strength=7, rbs_strength=5, duration_hours=12)
    print(f"Steady-state mRNA: {expr['steady_state']['mrna']:.1f} molecules")
    print(f"Steady-state protein: {expr['steady_state']['protein']:.1f} molecules")
    print(f"mRNA half-life: {expr['response_times']['mrna_halflife']:.2f} hours")
    print(f"Protein half-life: {expr['response_times']['protein_halflife']:.2f} hours")

    # 6. Enzyme kinetics
    print("\n6. ENZYME KINETICS")
    print("-" * 40)
    substrate = np.linspace(0, 100, 20)
    kinetics = lab.enzyme_kinetics(substrate, vmax=150, km=25)
    print(f"Vmax: {kinetics['vmax']} μM/min")
    print(f"Km: {kinetics['km']} μM")
    print(f"kcat: {kinetics['kcat']:.1f} /s")
    print(f"Catalytic efficiency (kcat/Km): {kinetics['kcat']/kinetics['km']:.2f} /μM/s")

    print("\n" + "=" * 60)
    print("Demo complete! Lab ready for production use.")


if __name__ == '__main__':
    run_demo()