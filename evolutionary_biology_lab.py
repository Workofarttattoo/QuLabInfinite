"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

EVOLUTIONARY BIOLOGY LAB - Production Ready
Advanced phylogenetics, population genetics, and molecular evolution.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from scipy import stats, optimize
from scipy.spatial.distance import pdist, squareform
from scipy.cluster import hierarchy
import itertools

@dataclass
class EvolutionaryBiologyLab:
    """Production-ready evolutionary biology analysis laboratory."""

    # Genetic parameters
    population_size: int = 1000
    generations: int = 100
    mutation_rate: float = 1e-8  # Per base per generation
    recombination_rate: float = 1e-8  # Per base per generation
    selection_coefficient: float = 0.01
    effective_population_size: int = 500

    # Molecular clock parameters
    SUBSTITUTION_RATES: Dict[str, float] = field(default_factory=lambda: {
        'synonymous': 1e-9,
        'nonsynonymous': 1e-10,
        'mitochondrial': 1e-8,
        'nuclear': 1e-9
    })

    # Transition/transversion matrix
    KIMURA_2P: np.ndarray = field(init=False)

    def __post_init__(self):
        """Initialize evolution matrices and populations."""
        self.initialize_substitution_models()
        self.population = self.initialize_population()
        self.fitness_landscape = None
        self.phylogenetic_tree = None
        self.allele_frequencies = {}

    def initialize_substitution_models(self):
        """Initialize nucleotide substitution models."""
        # Kimura 2-parameter model
        alpha = 0.5  # Transition rate
        beta = 0.25  # Transversion rate

        self.KIMURA_2P = np.array([
            [0, alpha, beta, beta],      # A -> C, G, T
            [alpha, 0, beta, beta],      # C -> A, G, T
            [beta, beta, 0, alpha],      # G -> A, C, T
            [beta, beta, alpha, 0]       # T -> A, C, G
        ])

        # Normalize
        self.KIMURA_2P = self.KIMURA_2P / self.KIMURA_2P.sum()

    def initialize_population(self, genome_length: int = 1000) -> np.ndarray:
        """Initialize population with random genomes."""
        # 0=A, 1=C, 2=G, 3=T
        return np.random.randint(0, 4, size=(self.population_size, genome_length))

    def wright_fisher_model(self, allele_freq: float, Ne: int,
                          selection: float = 0) -> float:
        """Wright-Fisher model with selection."""
        # Fitness values
        w_AA = 1 + selection
        w_Aa = 1 + selection / 2
        w_aa = 1

        # Calculate mean fitness
        p = allele_freq
        q = 1 - p
        w_bar = p * p * w_AA + 2 * p * q * w_Aa + q * q * w_aa

        # Next generation frequency
        p_prime = (p * p * w_AA + p * q * w_Aa) / w_bar

        # Add genetic drift
        if Ne > 0:
            # Binomial sampling
            n_alleles = np.random.binomial(2 * Ne, p_prime)
            p_prime = n_alleles / (2 * Ne)

        return p_prime

    def moran_process(self, population: np.ndarray,
                     fitness_function: callable) -> np.ndarray:
        """Moran process for continuous time evolution."""
        pop = population.copy()
        n = len(pop)

        # Calculate fitness
        fitnesses = np.array([fitness_function(individual) for individual in pop])

        # Normalize fitness
        fitnesses = fitnesses / fitnesses.sum()

        # Birth: select individual proportional to fitness
        birth_idx = np.random.choice(n, p=fitnesses)

        # Death: random individual dies
        death_idx = np.random.randint(n)

        # Replace
        pop[death_idx] = pop[birth_idx].copy()

        # Mutation
        if np.random.random() < self.mutation_rate * len(pop[0]):
            mutation_idx = np.random.randint(len(pop[0]))
            pop[death_idx, mutation_idx] = np.random.randint(4)

        return pop

    def coalescent_simulation(self, n_samples: int,
                            theta: float = 0.01) -> List[Tuple[int, int, float]]:
        """Coalescent simulation for genealogy reconstruction."""
        # theta = 4 * Ne * mu
        coalescent_events = []
        lineages = list(range(n_samples))
        time = 0

        while len(lineages) > 1:
            k = len(lineages)

            # Time to next coalescence
            rate = k * (k - 1) / 2
            waiting_time = np.random.exponential(1 / rate)
            time += waiting_time

            # Choose two lineages to coalesce
            pair = np.random.choice(lineages, 2, replace=False)
            coalescent_events.append((pair[0], pair[1], time))

            # Remove coalesced lineages and add ancestor
            lineages.remove(pair[0])
            lineages.remove(pair[1])
            lineages.append(max(lineages) + 1 if lineages else n_samples)

        return coalescent_events

    def jukes_cantor_distance(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Calculate Jukes-Cantor corrected distance."""
        # Proportion of differences
        p = np.mean(seq1 != seq2)

        # Jukes-Cantor correction
        if p < 0.75:
            d = -3/4 * np.log(1 - 4/3 * p)
        else:
            d = float('inf')  # Sequences too diverged

        return d

    def kimura_2p_distance(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Calculate Kimura 2-parameter distance."""
        n = len(seq1)

        # Count transitions and transversions
        transitions = 0
        transversions = 0

        transition_pairs = [(0, 2), (2, 0), (1, 3), (3, 1)]  # A-G, C-T

        for i in range(n):
            if seq1[i] != seq2[i]:
                if (seq1[i], seq2[i]) in transition_pairs:
                    transitions += 1
                else:
                    transversions += 1

        P = transitions / n  # Proportion of transitions
        Q = transversions / n  # Proportion of transversions

        # Kimura 2-parameter correction
        if P < 1 and Q < 1:
            d = -0.5 * np.log((1 - 2*P - Q) * np.sqrt(1 - 2*Q))
        else:
            d = float('inf')

        return d

    def build_distance_matrix(self, sequences: List[np.ndarray],
                             method: str = 'kimura') -> np.ndarray:
        """Build pairwise distance matrix."""
        n = len(sequences)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                if method == 'kimura':
                    dist = self.kimura_2p_distance(sequences[i], sequences[j])
                elif method == 'jc':
                    dist = self.jukes_cantor_distance(sequences[i], sequences[j])
                else:
                    dist = np.mean(sequences[i] != sequences[j])

                dist_matrix[i, j] = dist_matrix[j, i] = dist

        return dist_matrix

    def neighbor_joining(self, distance_matrix: np.ndarray) -> Dict:
        """Neighbor-joining tree construction algorithm."""
        n = distance_matrix.shape[0]
        tree = {'nodes': list(range(n)), 'edges': [], 'branch_lengths': {}}

        # Working copy of distance matrix
        dist = distance_matrix.copy()
        active_nodes = list(range(n))
        next_node = n

        while len(active_nodes) > 2:
            k = len(active_nodes)

            # Calculate Q-matrix
            q_matrix = np.full((k, k), np.inf)
            total_dist = dist.sum(axis=0)

            for i in range(k):
                for j in range(i + 1, k):
                    q_matrix[i, j] = (k - 2) * dist[i, j] - total_dist[i] - total_dist[j]
                    q_matrix[j, i] = q_matrix[i, j]

            # Find minimum Q value
            min_i, min_j = np.unravel_index(np.argmin(q_matrix), q_matrix.shape)

            # Calculate branch lengths
            branch_i = 0.5 * dist[min_i, min_j] + (total_dist[min_i] - total_dist[min_j]) / (2 * (k - 2))
            branch_j = dist[min_i, min_j] - branch_i

            # Add edges to tree
            node_i, node_j = active_nodes[min_i], active_nodes[min_j]
            tree['edges'].append((node_i, next_node))
            tree['edges'].append((node_j, next_node))
            tree['branch_lengths'][(node_i, next_node)] = max(0, branch_i)
            tree['branch_lengths'][(node_j, next_node)] = max(0, branch_j)

            # Update distance matrix
            new_dist = np.zeros((k - 1, k - 1))

            # Calculate distances to new node
            new_row = []
            idx = 0
            for m in range(k):
                if m not in [min_i, min_j]:
                    d = 0.5 * (dist[min_i, m] + dist[min_j, m] - dist[min_i, min_j])
                    new_row.append(d)

            # Copy unchanged distances
            row_idx = 0
            for i in range(k):
                if i in [min_i, min_j]:
                    continue
                col_idx = 0
                for j in range(k):
                    if j in [min_i, min_j]:
                        continue
                    new_dist[row_idx, col_idx] = dist[i, j]
                    col_idx += 1
                row_idx += 1

            # Add new node distances
            for i, d in enumerate(new_row):
                new_dist[i, -1] = new_dist[-1, i] = d

            dist = new_dist

            # Update active nodes
            active_nodes.remove(node_i)
            active_nodes.remove(node_j)
            active_nodes.append(next_node)
            tree['nodes'].append(next_node)
            next_node += 1

        # Connect final two nodes
        if len(active_nodes) == 2:
            tree['edges'].append((active_nodes[0], active_nodes[1]))
            tree['branch_lengths'][(active_nodes[0], active_nodes[1])] = dist[0, 1]

        self.phylogenetic_tree = tree
        return tree

    def calculate_dn_ds_ratio(self, codon_seq1: str, codon_seq2: str) -> float:
        """Calculate dN/dS ratio for detecting selection."""
        # Genetic code
        genetic_code = {
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

        syn_sites = 0
        non_syn_sites = 0
        syn_substitutions = 0
        non_syn_substitutions = 0

        # Process codons
        for i in range(0, min(len(codon_seq1), len(codon_seq2)) - 2, 3):
            codon1 = codon_seq1[i:i+3]
            codon2 = codon_seq2[i:i+3]

            if codon1 in genetic_code and codon2 in genetic_code:
                aa1 = genetic_code[codon1]
                aa2 = genetic_code[codon2]

                # Count differences
                for j in range(3):
                    if codon1[j] != codon2[j]:
                        # Check if substitution is synonymous
                        test_codon = list(codon1)
                        test_codon[j] = codon2[j]
                        test_codon = ''.join(test_codon)

                        if test_codon in genetic_code:
                            if genetic_code[test_codon] == aa1:
                                syn_substitutions += 1
                                syn_sites += 1
                            else:
                                non_syn_substitutions += 1
                                non_syn_sites += 1

        # Calculate rates
        if syn_sites > 0:
            ds = syn_substitutions / syn_sites
        else:
            ds = 0

        if non_syn_sites > 0:
            dn = non_syn_substitutions / non_syn_sites
        else:
            dn = 0

        # Calculate ratio
        if ds > 0:
            dn_ds = dn / ds
        else:
            dn_ds = float('inf') if dn > 0 else 1.0

        return dn_ds

    def tajima_d_test(self, sequences: List[np.ndarray]) -> float:
        """Tajima's D test for neutral evolution."""
        n = len(sequences)
        if n < 4:
            return 0.0

        # Calculate number of segregating sites
        s = 0
        for site in range(len(sequences[0])):
            alleles = set(seq[site] for seq in sequences)
            if len(alleles) > 1:
                s += 1

        if s == 0:
            return 0.0

        # Calculate average pairwise differences
        pi = 0
        comparisons = 0
        for i in range(n):
            for j in range(i + 1, n):
                pi += np.sum(sequences[i] != sequences[j])
                comparisons += 1
        pi = pi / comparisons

        # Calculate Watterson's theta
        a1 = sum(1 / i for i in range(1, n))
        theta_w = s / a1

        # Calculate variance
        a2 = sum(1 / (i * i) for i in range(1, n))
        b1 = (n + 1) / (3 * (n - 1))
        b2 = 2 * (n * n + n + 3) / (9 * n * (n - 1))

        c1 = b1 - 1 / a1
        c2 = b2 - (n + 2) / (a1 * n) + a2 / (a1 * a1)

        e1 = c1 / a1
        e2 = c2 / (a1 * a1 + a2)

        var_d = e1 * s + e2 * s * (s - 1)

        # Calculate Tajima's D
        if var_d > 0:
            d = (pi - theta_w) / np.sqrt(var_d)
        else:
            d = 0.0

        return d

    def mcdonald_kreitman_test(self, sequences_sp1: List[str],
                              sequences_sp2: List[str]) -> Dict:
        """McDonald-Kreitman test for adaptive evolution."""
        # Count polymorphisms within species
        pn_sp1 = 0  # Non-synonymous polymorphisms
        ps_sp1 = 0  # Synonymous polymorphisms

        for i in range(len(sequences_sp1)):
            for j in range(i + 1, len(sequences_sp1)):
                for k in range(0, min(len(sequences_sp1[i]), len(sequences_sp1[j])) - 2, 3):
                    codon1 = sequences_sp1[i][k:k+3]
                    codon2 = sequences_sp1[j][k:k+3]
                    if codon1 != codon2:
                        dn_ds = self.calculate_dn_ds_ratio(codon1, codon2)
                        if dn_ds > 1:
                            pn_sp1 += 1
                        else:
                            ps_sp1 += 1

        # Count fixed differences between species
        dn = 0  # Fixed non-synonymous
        ds = 0  # Fixed synonymous

        for seq1 in sequences_sp1[:1]:  # Use first sequence as representative
            for seq2 in sequences_sp2[:1]:
                dn_ds = self.calculate_dn_ds_ratio(seq1, seq2)
                if dn_ds > 1:
                    dn += 1
                else:
                    ds += 1

        # Perform Fisher's exact test
        contingency_table = [[pn_sp1, ps_sp1], [dn, ds]]
        odds_ratio, p_value = stats.fisher_exact(contingency_table)

        # Calculate neutrality index
        if dn * ps_sp1 > 0:
            ni = (pn_sp1 * ds) / (dn * ps_sp1)
        else:
            ni = 1.0

        return {
            'polymorphic_nonsyn': pn_sp1,
            'polymorphic_syn': ps_sp1,
            'fixed_nonsyn': dn,
            'fixed_syn': ds,
            'odds_ratio': odds_ratio,
            'p_value': p_value,
            'neutrality_index': ni
        }

    def fst_calculation(self, pop1_alleles: np.ndarray,
                       pop2_alleles: np.ndarray) -> float:
        """Calculate FST for population differentiation."""
        # Allele frequencies in each population
        p1 = np.mean(pop1_alleles)
        p2 = np.mean(pop2_alleles)

        # Total allele frequency
        p_total = (p1 + p2) / 2

        # Expected heterozygosity
        hs = (2 * p1 * (1 - p1) + 2 * p2 * (1 - p2)) / 2
        ht = 2 * p_total * (1 - p_total)

        # FST
        if ht > 0:
            fst = (ht - hs) / ht
        else:
            fst = 0

        return max(0, min(1, fst))

    def hardy_weinberg_test(self, genotypes: np.ndarray) -> Dict:
        """Test for Hardy-Weinberg equilibrium."""
        # Count genotypes (0=AA, 1=Aa, 2=aa)
        n_AA = np.sum(genotypes == 0)
        n_Aa = np.sum(genotypes == 1)
        n_aa = np.sum(genotypes == 2)
        n_total = len(genotypes)

        # Calculate allele frequencies
        p = (2 * n_AA + n_Aa) / (2 * n_total)
        q = 1 - p

        # Expected genotype frequencies
        exp_AA = n_total * p * p
        exp_Aa = n_total * 2 * p * q
        exp_aa = n_total * q * q

        # Chi-square test
        chi_square = ((n_AA - exp_AA)**2 / exp_AA +
                     (n_Aa - exp_Aa)**2 / exp_Aa +
                     (n_aa - exp_aa)**2 / exp_aa)

        p_value = 1 - stats.chi2.cdf(chi_square, df=1)

        # Inbreeding coefficient
        if p * q > 0:
            f_is = 1 - (n_Aa / n_total) / (2 * p * q)
        else:
            f_is = 0

        return {
            'observed': {'AA': n_AA, 'Aa': n_Aa, 'aa': n_aa},
            'expected': {'AA': exp_AA, 'Aa': exp_Aa, 'aa': exp_aa},
            'allele_freq': {'A': p, 'a': q},
            'chi_square': chi_square,
            'p_value': p_value,
            'inbreeding_coefficient': f_is
        }

    def simulate_genetic_drift(self, initial_freq: float, Ne: int,
                              generations: int) -> np.ndarray:
        """Simulate genetic drift over time."""
        frequencies = np.zeros(generations)
        frequencies[0] = initial_freq

        for t in range(1, generations):
            # Binomial sampling
            n_alleles = np.random.binomial(2 * Ne, frequencies[t-1])
            frequencies[t] = n_alleles / (2 * Ne)

            # Check for fixation
            if frequencies[t] == 0 or frequencies[t] == 1:
                frequencies[t:] = frequencies[t]
                break

        return frequencies

    def molecular_clock_dating(self, sequences: List[np.ndarray],
                             calibration_time: float,
                             calibration_nodes: Tuple[int, int]) -> Dict:
        """Estimate divergence times using molecular clock."""
        # Build distance matrix
        dist_matrix = self.build_distance_matrix(sequences)

        # Get calibration distance
        calib_dist = dist_matrix[calibration_nodes[0], calibration_nodes[1]]

        # Calculate substitution rate
        if calib_dist > 0:
            rate = calib_dist / (2 * calibration_time)  # Divide by 2 for time to MRCA
        else:
            rate = 1e-9  # Default rate

        # Estimate all divergence times
        divergence_times = {}
        n = len(sequences)

        for i in range(n):
            for j in range(i + 1, n):
                time = dist_matrix[i, j] / (2 * rate)
                divergence_times[(i, j)] = time

        return {
            'substitution_rate': rate,
            'divergence_times': divergence_times,
            'calibration_distance': calib_dist
        }

    def simulate_adaptive_landscape(self, genome_length: int = 100,
                                  n_peaks: int = 5) -> np.ndarray:
        """Create adaptive landscape with multiple fitness peaks."""
        landscape = np.zeros((4**3, ))  # For 3-locus system

        # Create random fitness peaks
        for _ in range(n_peaks):
            peak_position = np.random.randint(0, 4**3)
            peak_height = np.random.uniform(0.5, 1.0)

            # Add Gaussian peak
            for i in range(len(landscape)):
                # Hamming distance from peak
                dist = bin(i ^ peak_position).count('1')
                landscape[i] += peak_height * np.exp(-dist / 2)

        # Normalize
        landscape = landscape / landscape.max()
        self.fitness_landscape = landscape

        return landscape

    def hill_climbing_evolution(self, initial_genotype: np.ndarray,
                              fitness_function: callable,
                              max_steps: int = 1000) -> List[np.ndarray]:
        """Simulate evolution via hill climbing."""
        trajectory = [initial_genotype.copy()]
        current = initial_genotype.copy()
        current_fitness = fitness_function(current)

        for step in range(max_steps):
            # Generate all single mutants
            improved = False

            for i in range(len(current)):
                for allele in range(4):
                    if allele != current[i]:
                        mutant = current.copy()
                        mutant[i] = allele
                        mutant_fitness = fitness_function(mutant)

                        if mutant_fitness > current_fitness:
                            current = mutant
                            current_fitness = mutant_fitness
                            trajectory.append(current.copy())
                            improved = True
                            break

                if improved:
                    break

            if not improved:
                break  # Local optimum reached

        return trajectory

    def linkage_disequilibrium(self, genotypes: np.ndarray,
                              locus1: int, locus2: int) -> Dict:
        """Calculate linkage disequilibrium between two loci."""
        # Extract alleles at two loci
        alleles1 = genotypes[:, locus1]
        alleles2 = genotypes[:, locus2]

        # Calculate haplotype frequencies
        haplotypes = {}
        for a1, a2 in zip(alleles1, alleles2):
            hap = (a1, a2)
            haplotypes[hap] = haplotypes.get(hap, 0) + 1

        n = len(genotypes)
        for hap in haplotypes:
            haplotypes[hap] /= n

        # Calculate allele frequencies
        p1 = np.mean(alleles1 == 1)
        p2 = np.mean(alleles2 == 1)

        # Calculate D
        f11 = haplotypes.get((1, 1), 0)
        D = f11 - p1 * p2

        # Calculate D'
        if D >= 0:
            D_max = min(p1 * (1 - p2), (1 - p1) * p2)
        else:
            D_max = max(-p1 * p2, -(1 - p1) * (1 - p2))

        if D_max != 0:
            D_prime = D / D_max
        else:
            D_prime = 0

        # Calculate r²
        if p1 * (1 - p1) * p2 * (1 - p2) > 0:
            r_squared = D**2 / (p1 * (1 - p1) * p2 * (1 - p2))
        else:
            r_squared = 0

        return {
            'D': D,
            'D_prime': D_prime,
            'r_squared': r_squared,
            'haplotype_frequencies': haplotypes
        }

    def run_comprehensive_analysis(self, sequences: List[np.ndarray]) -> Dict:
        """Run complete evolutionary analysis pipeline."""
        results = {}

        # Basic statistics
        results['num_sequences'] = len(sequences)
        results['sequence_length'] = len(sequences[0]) if sequences else 0

        # Distance matrix
        dist_matrix = self.build_distance_matrix(sequences)
        results['mean_distance'] = np.mean(dist_matrix[np.triu_indices_from(dist_matrix, k=1)])

        # Phylogenetic tree
        tree = self.neighbor_joining(dist_matrix)
        results['tree_nodes'] = len(tree['nodes'])
        results['tree_edges'] = len(tree['edges'])

        # Selection tests
        tajima_d = self.tajima_d_test(sequences)
        results['tajima_d'] = tajima_d

        # Population genetics
        if len(sequences) >= 2:
            # Simulate two populations
            pop1 = sequences[:len(sequences)//2]
            pop2 = sequences[len(sequences)//2:]

            # FST
            if pop1 and pop2:
                fst = self.fst_calculation(
                    np.array([seq[0] for seq in pop1]),
                    np.array([seq[0] for seq in pop2])
                )
                results['fst'] = fst

        # Genetic drift simulation
        drift_trajectory = self.simulate_genetic_drift(0.5, 100, 50)
        results['final_allele_freq'] = drift_trajectory[-1]

        return results


def generate_demo_sequences(n_sequences: int = 10,
                           sequence_length: int = 100) -> List[np.ndarray]:
    """Generate synthetic DNA sequences for demonstration."""
    sequences = []

    # Create ancestral sequence
    ancestor = np.random.randint(0, 4, sequence_length)

    for i in range(n_sequences):
        # Mutate from ancestor
        seq = ancestor.copy()

        # Add mutations
        n_mutations = np.random.poisson(5)  # Average 5 mutations
        mutation_sites = np.random.choice(sequence_length, n_mutations, replace=False)

        for site in mutation_sites:
            seq[site] = np.random.randint(0, 4)

        sequences.append(seq)

    return sequences


def run_demo():
    """Demonstrate comprehensive evolutionary biology capabilities."""
    print("=" * 80)
    print("EVOLUTIONARY BIOLOGY LAB - Production Demo")
    print("=" * 80)

    lab = EvolutionaryBiologyLab()

    print("\n1. POPULATION INITIALIZATION")
    print("-" * 40)
    population = lab.initialize_population(genome_length=100)
    print(f"Population size: {len(population)}")
    print(f"Genome length: {population.shape[1]}")

    print("\n2. WRIGHT-FISHER MODEL")
    print("-" * 40)
    initial_freq = 0.3
    frequencies = []
    freq = initial_freq

    for gen in range(20):
        freq = lab.wright_fisher_model(freq, Ne=100, selection=0.01)
        frequencies.append(freq)

    print(f"Initial allele frequency: {initial_freq:.3f}")
    print(f"Final allele frequency: {frequencies[-1]:.3f}")
    print(f"Selection coefficient: 0.01")

    print("\n3. GENETIC DISTANCE CALCULATION")
    print("-" * 40)
    sequences = generate_demo_sequences(5, 50)

    # Jukes-Cantor distance
    jc_dist = lab.jukes_cantor_distance(sequences[0], sequences[1])
    print(f"Jukes-Cantor distance: {jc_dist:.4f}")

    # Kimura 2-parameter distance
    k2p_dist = lab.kimura_2p_distance(sequences[0], sequences[1])
    print(f"Kimura 2-parameter distance: {k2p_dist:.4f}")

    print("\n4. PHYLOGENETIC TREE CONSTRUCTION")
    print("-" * 40)
    dist_matrix = lab.build_distance_matrix(sequences)
    tree = lab.neighbor_joining(dist_matrix)

    print(f"Tree nodes: {len(tree['nodes'])}")
    print(f"Tree edges: {len(tree['edges'])}")
    print(f"Branch lengths: {len(tree['branch_lengths'])}")

    print("\n5. SELECTION TESTS")
    print("-" * 40)

    # Tajima's D
    tajima_d = lab.tajima_d_test(sequences)
    print(f"Tajima's D: {tajima_d:.4f}")
    if tajima_d < -2:
        print("  Interpretation: Excess of rare alleles (positive selection or expansion)")
    elif tajima_d > 2:
        print("  Interpretation: Excess of intermediate alleles (balancing selection)")
    else:
        print("  Interpretation: Consistent with neutral evolution")

    # dN/dS ratio
    codon_seq1 = "ATGCGATCGTAGCTAGC"
    codon_seq2 = "ATGCGATCGTTGCTTGC"
    dn_ds = lab.calculate_dn_ds_ratio(codon_seq1, codon_seq2)
    print(f"\ndN/dS ratio: {dn_ds:.4f}")
    if dn_ds > 1:
        print("  Interpretation: Positive selection")
    elif dn_ds < 1:
        print("  Interpretation: Purifying selection")
    else:
        print("  Interpretation: Neutral evolution")

    print("\n6. POPULATION DIFFERENTIATION (FST)")
    print("-" * 40)
    pop1_alleles = np.random.binomial(1, 0.3, 100)
    pop2_alleles = np.random.binomial(1, 0.7, 100)
    fst = lab.fst_calculation(pop1_alleles, pop2_alleles)
    print(f"FST: {fst:.4f}")
    if fst < 0.05:
        print("  Interpretation: Little differentiation")
    elif fst < 0.15:
        print("  Interpretation: Moderate differentiation")
    elif fst < 0.25:
        print("  Interpretation: Large differentiation")
    else:
        print("  Interpretation: Very large differentiation")

    print("\n7. HARDY-WEINBERG EQUILIBRIUM TEST")
    print("-" * 40)
    # Generate genotypes (0=AA, 1=Aa, 2=aa)
    genotypes = np.random.choice([0, 1, 2], 1000, p=[0.25, 0.5, 0.25])
    hw_test = lab.hardy_weinberg_test(genotypes)

    print(f"Observed genotypes: AA={hw_test['observed']['AA']}, "
          f"Aa={hw_test['observed']['Aa']}, aa={hw_test['observed']['aa']}")
    print(f"Expected genotypes: AA={hw_test['expected']['AA']:.1f}, "
          f"Aa={hw_test['expected']['Aa']:.1f}, aa={hw_test['expected']['aa']:.1f}")
    print(f"Chi-square: {hw_test['chi_square']:.4f}")
    print(f"P-value: {hw_test['p_value']:.4f}")
    print(f"Inbreeding coefficient: {hw_test['inbreeding_coefficient']:.4f}")

    print("\n8. GENETIC DRIFT SIMULATION")
    print("-" * 40)
    drift_freq = lab.simulate_genetic_drift(0.5, Ne=50, generations=100)
    print(f"Initial frequency: 0.5")
    print(f"Final frequency: {drift_freq[-1]:.3f}")
    print(f"Fixation reached: {'Yes' if drift_freq[-1] in [0, 1] else 'No'}")

    print("\n9. COALESCENT SIMULATION")
    print("-" * 40)
    coal_events = lab.coalescent_simulation(n_samples=6, theta=0.01)
    print(f"Number of coalescent events: {len(coal_events)}")
    for event in coal_events[:3]:
        print(f"  Lineages {event[0]} and {event[1]} coalesce at time {event[2]:.3f}")

    print("\n10. MOLECULAR CLOCK DATING")
    print("-" * 40)
    dating = lab.molecular_clock_dating(
        sequences, calibration_time=10.0, calibration_nodes=(0, 1)
    )
    print(f"Substitution rate: {dating['substitution_rate']:.2e} per site per time unit")
    print(f"Example divergence times:")
    for pair, time in list(dating['divergence_times'].items())[:3]:
        print(f"  Sequences {pair[0]}-{pair[1]}: {time:.2f} time units ago")

    print("\n11. ADAPTIVE LANDSCAPE")
    print("-" * 40)
    landscape = lab.simulate_adaptive_landscape(genome_length=10, n_peaks=3)
    print(f"Fitness landscape created with {len(landscape)} genotypes")
    print(f"Maximum fitness: {np.max(landscape):.3f}")
    print(f"Minimum fitness: {np.min(landscape):.3f}")
    print(f"Number of local peaks: ~3")

    print("\n12. LINKAGE DISEQUILIBRIUM")
    print("-" * 40)
    genotypes = np.random.randint(0, 2, size=(1000, 10))
    ld = lab.linkage_disequilibrium(genotypes, locus1=0, locus2=1)
    print(f"D = {ld['D']:.4f}")
    print(f"D' = {ld['D_prime']:.4f}")
    print(f"r² = {ld['r_squared']:.4f}")

    print("\n13. COMPREHENSIVE ANALYSIS")
    print("-" * 40)
    results = lab.run_comprehensive_analysis(sequences)
    print("Analysis complete:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("Evolutionary Biology Lab demonstration complete!")
    print("=" * 80)


if __name__ == '__main__':
    run_demo()