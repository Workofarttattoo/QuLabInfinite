"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

GENETICS LAB
Production-ready genetics algorithms for Mendelian inheritance, Hardy-Weinberg equilibrium,
linkage analysis, QTL mapping, population genetics, and mutation rate estimation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import math
from scipy import stats, special
from scipy.stats import chi2, norm, binom
import itertools


@dataclass
class Allele:
    """Represents a genetic allele"""
    name: str
    frequency: float
    dominance: str = 'codominant'  # dominant, recessive, codominant
    fitness: float = 1.0
    mutation_rate: float = 1e-8


@dataclass
class Gene:
    """Represents a gene with multiple alleles"""
    name: str
    chromosome: int
    position: float  # cM or bp
    alleles: List[Allele]

    def get_allele_frequencies(self) -> np.ndarray:
        """Get array of allele frequencies"""
        return np.array([a.frequency for a in self.alleles])


@dataclass
class Genotype:
    """Represents an individual's genotype"""
    allele1: str
    allele2: str
    gene: str

    def is_homozygous(self) -> bool:
        return self.allele1 == self.allele2

    def is_heterozygous(self) -> bool:
        return self.allele1 != self.allele2


class MendelianInheritance:
    """Model Mendelian inheritance patterns"""

    def __init__(self):
        self.inheritance_patterns = {
            'autosomal_dominant': self._autosomal_dominant,
            'autosomal_recessive': self._autosomal_recessive,
            'x_linked_dominant': self._x_linked_dominant,
            'x_linked_recessive': self._x_linked_recessive,
            'codominant': self._codominant,
            'incomplete_dominance': self._incomplete_dominance
        }

    def punnett_square(self, parent1_genotype: str, parent2_genotype: str) -> Dict:
        """Generate Punnett square for single gene cross

        Args:
            parent1_genotype: e.g., 'Aa', 'AA', 'aa'
            parent2_genotype: e.g., 'Aa', 'AA', 'aa'
        Returns:
            Offspring genotype probabilities
        """
        # Extract alleles from each parent
        p1_alleles = list(parent1_genotype)
        p2_alleles = list(parent2_genotype)

        # Generate all possible offspring genotypes
        offspring = {}
        for a1 in p1_alleles:
            for a2 in p2_alleles:
                genotype = ''.join(sorted([a1, a2]))
                offspring[genotype] = offspring.get(genotype, 0) + 0.25

        return offspring

    def dihybrid_cross(self, parent1: str, parent2: str) -> Dict:
        """Perform dihybrid cross (two genes)

        Args:
            parent1: e.g., 'AaBb'
            parent2: e.g., 'AaBb'
        Returns:
            Offspring genotype and phenotype ratios
        """
        # Extract genotypes for each gene
        gene1_p1 = parent1[0:2]
        gene1_p2 = parent2[0:2]
        gene2_p1 = parent1[2:4]
        gene2_p2 = parent2[2:4]

        # Get offspring for each gene
        gene1_offspring = self.punnett_square(gene1_p1, gene1_p2)
        gene2_offspring = self.punnett_square(gene2_p1, gene2_p2)

        # Combine for dihybrid ratios
        dihybrid = {}
        for g1, p1 in gene1_offspring.items():
            for g2, p2 in gene2_offspring.items():
                genotype = g1 + g2
                dihybrid[genotype] = p1 * p2

        # Calculate phenotype ratios (assuming simple dominance)
        phenotypes = {}
        for genotype, prob in dihybrid.items():
            pheno = ''
            pheno += 'Dominant1' if 'A' in genotype[0:2] else 'Recessive1'
            pheno += '_'
            pheno += 'Dominant2' if 'B' in genotype[2:4] else 'Recessive2'
            phenotypes[pheno] = phenotypes.get(pheno, 0) + prob

        return {'genotypes': dihybrid, 'phenotypes': phenotypes}

    def _autosomal_dominant(self, genotype: str) -> str:
        """Determine phenotype for autosomal dominant inheritance"""
        return 'affected' if 'A' in genotype else 'normal'

    def _autosomal_recessive(self, genotype: str) -> str:
        """Determine phenotype for autosomal recessive inheritance"""
        return 'affected' if genotype == 'aa' else 'normal'

    def _x_linked_dominant(self, genotype: str, sex: str) -> str:
        """Determine phenotype for X-linked dominant inheritance"""
        if sex == 'male':
            return 'affected' if genotype[0] == 'A' else 'normal'
        else:  # female
            return 'affected' if 'A' in genotype else 'normal'

    def _x_linked_recessive(self, genotype: str, sex: str) -> str:
        """Determine phenotype for X-linked recessive inheritance"""
        if sex == 'male':
            return 'affected' if genotype[0] == 'a' else 'normal'
        else:  # female
            return 'affected' if genotype == 'aa' else 'normal'

    def _codominant(self, genotype: str) -> str:
        """Determine phenotype for codominant inheritance (e.g., blood type)"""
        if genotype == 'AA':
            return 'Type_A'
        elif genotype == 'BB':
            return 'Type_B'
        elif genotype in ['AB', 'BA']:
            return 'Type_AB'
        else:
            return 'Type_O'

    def _incomplete_dominance(self, genotype: str) -> str:
        """Determine phenotype for incomplete dominance"""
        if genotype == 'AA':
            return 'red'
        elif genotype in ['Aa', 'aA']:
            return 'pink'
        else:
            return 'white'

    def test_cross(self, unknown_genotype: str, tester_genotype: str = 'aa') -> Dict:
        """Perform test cross to determine unknown genotype"""
        possible_genotypes = ['AA', 'Aa']
        results = {}

        for possible in possible_genotypes:
            offspring = self.punnett_square(possible, tester_genotype)
            results[possible] = offspring

        return results

    def pedigree_analysis(self, pedigree: Dict, mode: str = 'autosomal_dominant') -> Dict:
        """Analyze inheritance pattern in pedigree

        Args:
            pedigree: Family tree with affected status
            mode: Inheritance mode to test
        Returns:
            Likelihood of inheritance mode
        """
        # Simplified pedigree analysis
        affected_parents = pedigree.get('affected_parents', 0)
        affected_offspring = pedigree.get('affected_offspring', 0)
        total_offspring = pedigree.get('total_offspring', 1)

        # Calculate expected ratios for different modes
        if mode == 'autosomal_dominant':
            if affected_parents >= 1:
                expected_ratio = 0.5  # Aa x aa cross
            else:
                expected_ratio = 0
        elif mode == 'autosomal_recessive':
            if affected_parents == 2:
                expected_ratio = 1.0  # aa x aa
            elif affected_parents == 0:
                expected_ratio = 0.25  # Aa x Aa
            else:
                expected_ratio = 0
        else:
            expected_ratio = 0.5

        observed_ratio = affected_offspring / total_offspring if total_offspring > 0 else 0

        # Chi-square test
        expected = expected_ratio * total_offspring
        observed = affected_offspring

        if expected > 0:
            chi_sq = (observed - expected) ** 2 / expected
            p_value = 1 - chi2.cdf(chi_sq, df=1)
        else:
            p_value = 0 if observed > 0 else 1

        return {
            'mode': mode,
            'expected_ratio': expected_ratio,
            'observed_ratio': observed_ratio,
            'chi_square': chi_sq if expected > 0 else 0,
            'p_value': p_value,
            'consistent': p_value > 0.05
        }


class HardyWeinbergEquilibrium:
    """Hardy-Weinberg equilibrium calculations and tests"""

    def __init__(self):
        self.assumptions = [
            'No mutations',
            'Random mating',
            'No gene flow',
            'Infinite population size',
            'No selection'
        ]

    def calculate_genotype_frequencies(self, p: float, q: float = None) -> Dict:
        """Calculate expected genotype frequencies under HWE

        Args:
            p: Frequency of allele A
            q: Frequency of allele a (if None, calculated as 1-p)
        Returns:
            Expected genotype frequencies
        """
        if q is None:
            q = 1 - p

        # Hardy-Weinberg frequencies
        frequencies = {
            'AA': p ** 2,
            'Aa': 2 * p * q,
            'aa': q ** 2
        }

        # Verify sum to 1
        total = sum(frequencies.values())
        assert abs(total - 1.0) < 1e-10, f"Frequencies don't sum to 1: {total}"

        return frequencies

    def estimate_allele_frequencies(self, genotype_counts: Dict) -> Dict:
        """Estimate allele frequencies from genotype counts

        Args:
            genotype_counts: {'AA': n_AA, 'Aa': n_Aa, 'aa': n_aa}
        Returns:
            Allele frequencies
        """
        n_AA = genotype_counts.get('AA', 0)
        n_Aa = genotype_counts.get('Aa', 0)
        n_aa = genotype_counts.get('aa', 0)

        total = n_AA + n_Aa + n_aa
        if total == 0:
            return {'p': 0.5, 'q': 0.5}

        # Count alleles
        p = (2 * n_AA + n_Aa) / (2 * total)
        q = (2 * n_aa + n_Aa) / (2 * total)

        return {'p': p, 'q': q}

    def chi_square_test(self, observed_counts: Dict) -> Dict:
        """Test for Hardy-Weinberg equilibrium using chi-square test

        Args:
            observed_counts: {'AA': n_AA, 'Aa': n_Aa, 'aa': n_aa}
        Returns:
            Test statistics and p-value
        """
        # Calculate total and allele frequencies
        total = sum(observed_counts.values())
        if total == 0:
            return {'chi_square': 0, 'p_value': 1, 'df': 1, 'in_equilibrium': True}

        allele_freq = self.estimate_allele_frequencies(observed_counts)
        p, q = allele_freq['p'], allele_freq['q']

        # Calculate expected counts
        expected = {
            'AA': p ** 2 * total,
            'Aa': 2 * p * q * total,
            'aa': q ** 2 * total
        }

        # Chi-square statistic
        chi_sq = 0
        for genotype in ['AA', 'Aa', 'aa']:
            obs = observed_counts.get(genotype, 0)
            exp = expected[genotype]
            if exp > 0:
                chi_sq += (obs - exp) ** 2 / exp

        # Degrees of freedom = number of genotypes - number of alleles
        df = 1  # 3 genotypes - 2 alleles (p and q are not independent)

        # P-value
        p_value = 1 - chi2.cdf(chi_sq, df)

        return {
            'chi_square': chi_sq,
            'p_value': p_value,
            'df': df,
            'in_equilibrium': p_value > 0.05,
            'expected': expected,
            'observed': observed_counts
        }

    def inbreeding_coefficient(self, observed_heterozygosity: float,
                             expected_heterozygosity: float) -> float:
        """Calculate inbreeding coefficient (F)

        F = (He - Ho) / He
        where He = expected heterozygosity, Ho = observed heterozygosity
        """
        if expected_heterozygosity == 0:
            return 0

        F = (expected_heterozygosity - observed_heterozygosity) / expected_heterozygosity
        return F

    def wahlund_effect(self, subpop_frequencies: List[Dict]) -> Dict:
        """Calculate Wahlund effect (deficit of heterozygotes due to population substructure)

        Args:
            subpop_frequencies: List of {'p': p_i, 'weight': w_i} for each subpopulation
        Returns:
            Overall frequencies and heterozygote deficit
        """
        # Calculate weighted average allele frequency
        p_bar = sum(sub['p'] * sub['weight'] for sub in subpop_frequencies)
        q_bar = 1 - p_bar

        # Expected heterozygosity if single population
        He_total = 2 * p_bar * q_bar

        # Actual heterozygosity (average across subpopulations)
        Ho_total = sum(2 * sub['p'] * (1 - sub['p']) * sub['weight']
                      for sub in subpop_frequencies)

        # Variance in allele frequency
        var_p = sum((sub['p'] - p_bar) ** 2 * sub['weight']
                   for sub in subpop_frequencies)

        # F_ST (fixation index)
        F_ST = var_p / (p_bar * q_bar) if p_bar * q_bar > 0 else 0

        return {
            'p_bar': p_bar,
            'He_total': He_total,
            'Ho_total': Ho_total,
            'heterozygote_deficit': He_total - Ho_total,
            'F_ST': F_ST,
            'variance_p': var_p
        }


class LinkageAnalysis:
    """Genetic linkage and recombination analysis"""

    def __init__(self):
        self.mapping_functions = {
            'haldane': self._haldane_function,
            'kosambi': self._kosambi_function
        }

    def recombination_frequency(self, recombinant: int, total: int) -> float:
        """Calculate recombination frequency

        Args:
            recombinant: Number of recombinant offspring
            total: Total number of offspring
        Returns:
            Recombination frequency (0-0.5)
        """
        if total == 0:
            return 0

        freq = recombinant / total
        return min(freq, 0.5)  # Max is 0.5 (unlinked)

    def genetic_distance(self, recombination_freq: float,
                        mapping_function: str = 'haldane') -> float:
        """Convert recombination frequency to genetic distance (centiMorgans)

        Args:
            recombination_freq: Recombination frequency (0-0.5)
            mapping_function: 'haldane' or 'kosambi'
        Returns:
            Genetic distance in cM
        """
        if mapping_function in self.mapping_functions:
            return self.mapping_functions[mapping_function](recombination_freq)
        else:
            # Simple linear approximation
            return recombination_freq * 100

    def _haldane_function(self, r: float) -> float:
        """Haldane mapping function (no interference)"""
        if r <= 0:
            return 0
        elif r >= 0.5:
            return float('inf')
        else:
            return -50 * math.log(1 - 2 * r)

    def _kosambi_function(self, r: float) -> float:
        """Kosambi mapping function (with interference)"""
        if r <= 0:
            return 0
        elif r >= 0.5:
            return float('inf')
        else:
            return 25 * math.log((1 + 2 * r) / (1 - 2 * r))

    def lod_score(self, observed_recombinant: int, observed_parental: int,
                 test_theta: float) -> float:
        """Calculate LOD score for linkage

        LOD = log10(L(theta) / L(0.5))

        Args:
            observed_recombinant: Number of recombinant offspring
            observed_parental: Number of parental offspring
            test_theta: Recombination fraction to test
        Returns:
            LOD score
        """
        n_r = observed_recombinant
        n_p = observed_parental
        n_total = n_r + n_p

        if n_total == 0:
            return 0

        # Likelihood at test theta
        if 0 < test_theta < 1:
            L_theta = (test_theta ** n_r) * ((1 - test_theta) ** n_p)
        else:
            L_theta = 0

        # Likelihood at theta = 0.5 (unlinked)
        L_unlinked = (0.5 ** n_total)

        if L_unlinked > 0:
            lod = math.log10(L_theta / L_unlinked) if L_theta > 0 else -float('inf')
        else:
            lod = 0

        return lod

    def three_point_cross(self, genotype_counts: Dict) -> Dict:
        """Analyze three-point cross to determine gene order and distances

        Args:
            genotype_counts: Counts of each genotype class
        Returns:
            Gene order and map distances
        """
        # Sort genotype classes by frequency
        sorted_classes = sorted(genotype_counts.items(), key=lambda x: x[1], reverse=True)

        # Identify parental and double crossover classes
        parental_classes = sorted_classes[:2]
        double_co_classes = sorted_classes[-2:]

        # Single crossover classes are in between
        single_co_1 = sorted_classes[2:4]
        single_co_2 = sorted_classes[4:6]

        total = sum(genotype_counts.values())

        # Calculate recombination frequencies
        rf_1 = sum(c[1] for c in single_co_1) / total
        rf_2 = sum(c[1] for c in single_co_2) / total
        rf_double = sum(c[1] for c in double_co_classes) / total

        # Coefficient of coincidence
        expected_double = rf_1 * rf_2
        if expected_double > 0:
            coc = rf_double / expected_double
        else:
            coc = 0

        # Interference
        interference = 1 - coc

        return {
            'parental': parental_classes,
            'single_crossover_1': single_co_1,
            'single_crossover_2': single_co_2,
            'double_crossover': double_co_classes,
            'recombination_freq_1': rf_1,
            'recombination_freq_2': rf_2,
            'total_map_distance': (rf_1 + rf_2) * 100,  # in cM
            'coefficient_of_coincidence': coc,
            'interference': interference
        }


class QTLMapping:
    """Quantitative Trait Locus mapping"""

    def __init__(self):
        self.methods = ['single_marker', 'interval_mapping', 'composite_interval']

    def single_marker_analysis(self, marker_genotypes: np.ndarray,
                              phenotypes: np.ndarray) -> Dict:
        """Single marker QTL analysis using ANOVA/regression

        Args:
            marker_genotypes: Array of genotypes (0, 1, 2 for AA, Aa, aa)
            phenotypes: Array of phenotypic values
        Returns:
            QTL statistics
        """
        # Group phenotypes by genotype
        groups = []
        for genotype in [0, 1, 2]:
            group_pheno = phenotypes[marker_genotypes == genotype]
            if len(group_pheno) > 0:
                groups.append(group_pheno)

        if len(groups) < 2:
            return {'significant': False, 'p_value': 1.0}

        # ANOVA
        f_stat, p_value = stats.f_oneway(*groups)

        # Calculate effect sizes
        means = [np.mean(g) for g in groups]

        # Additive effect (a)
        if len(means) >= 3:
            a = (means[2] - means[0]) / 2
            # Dominance effect (d)
            d = means[1] - (means[0] + means[2]) / 2
        else:
            a = means[-1] - means[0] if len(means) > 1 else 0
            d = 0

        # Variance explained (R²)
        total_var = np.var(phenotypes)
        if total_var > 0:
            group_means = np.array([means[int(g)] for g in marker_genotypes])
            explained_var = np.var(group_means)
            r_squared = explained_var / total_var
        else:
            r_squared = 0

        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'means': means,
            'additive_effect': a,
            'dominance_effect': d,
            'r_squared': r_squared
        }

    def interval_mapping(self, markers: List[Dict], phenotypes: np.ndarray,
                        step: float = 1.0) -> List[Dict]:
        """Interval mapping for QTL detection

        Args:
            markers: List of {'position': cM, 'genotypes': array}
            phenotypes: Phenotypic values
            step: Step size in cM
        Returns:
            LOD scores across genome
        """
        results = []

        for i in range(len(markers) - 1):
            left_marker = markers[i]
            right_marker = markers[i + 1]

            # Distance between markers
            distance = right_marker['position'] - left_marker['position']

            # Test positions between markers
            positions = np.arange(left_marker['position'], right_marker['position'], step)

            for pos in positions:
                # Interpolate genotype probabilities
                weight_left = (right_marker['position'] - pos) / distance
                weight_right = (pos - left_marker['position']) / distance

                # Simple weighted average (more sophisticated methods exist)
                interpolated_geno = (weight_left * left_marker['genotypes'] +
                                    weight_right * right_marker['genotypes'])

                # Test for QTL
                qtl_test = self.single_marker_analysis(interpolated_geno, phenotypes)

                # Convert p-value to LOD score
                if qtl_test['p_value'] > 0:
                    lod = -math.log10(qtl_test['p_value'])
                else:
                    lod = 10  # Cap at 10

                results.append({
                    'position': pos,
                    'lod_score': lod,
                    'p_value': qtl_test['p_value'],
                    'additive_effect': qtl_test['additive_effect'],
                    'dominance_effect': qtl_test['dominance_effect']
                })

        return results

    def permutation_threshold(self, markers: List[Dict], phenotypes: np.ndarray,
                            n_permutations: int = 1000, alpha: float = 0.05) -> float:
        """Determine significance threshold using permutation test

        Args:
            markers: Marker data
            phenotypes: Phenotypic values
            n_permutations: Number of permutations
            alpha: Significance level
        Returns:
            LOD threshold for significance
        """
        max_lods = []

        for _ in range(n_permutations):
            # Shuffle phenotypes
            shuffled_pheno = np.random.permutation(phenotypes)

            # Run QTL analysis
            qtl_results = self.interval_mapping(markers, shuffled_pheno)

            # Record maximum LOD
            max_lod = max(r['lod_score'] for r in qtl_results)
            max_lods.append(max_lod)

        # Find threshold at (1-alpha) quantile
        threshold = np.percentile(max_lods, (1 - alpha) * 100)

        return threshold


class PopulationGenetics:
    """Population genetics calculations and simulations"""

    def __init__(self):
        self.forces = ['mutation', 'selection', 'migration', 'drift']

    def wright_fisher_simulation(self, initial_freq: float, population_size: int,
                                generations: int, selection: float = 0,
                                mutation_rate: float = 0) -> np.ndarray:
        """Wright-Fisher simulation of allele frequency evolution

        Args:
            initial_freq: Starting frequency of allele A
            population_size: Effective population size
            generations: Number of generations to simulate
            selection: Selection coefficient (s)
            mutation_rate: Mutation rate per generation
        Returns:
            Allele frequencies over time
        """
        frequencies = np.zeros(generations)
        frequencies[0] = initial_freq

        for gen in range(1, generations):
            p = frequencies[gen - 1]

            # Selection
            if selection != 0:
                # Relative fitness: AA=1+s, Aa=1+s/2, aa=1
                w_bar = p**2 * (1 + selection) + 2*p*(1-p)*(1 + selection/2) + (1-p)**2
                p_prime = (p**2 * (1 + selection) + p*(1-p)*(1 + selection/2)) / w_bar
            else:
                p_prime = p

            # Mutation
            if mutation_rate > 0:
                p_prime = p_prime * (1 - mutation_rate) + (1 - p_prime) * mutation_rate

            # Genetic drift (binomial sampling)
            n_alleles = 2 * population_size
            n_A = np.random.binomial(n_alleles, p_prime)
            frequencies[gen] = n_A / n_alleles

            # Check for fixation
            if frequencies[gen] == 0 or frequencies[gen] == 1:
                frequencies[gen:] = frequencies[gen]
                break

        return frequencies

    def effective_population_size(self, census_size: int, sex_ratio: float = 0.5,
                                variance_offspring: float = 2) -> float:
        """Calculate effective population size

        Args:
            census_size: Actual population size
            sex_ratio: Proportion of males
            variance_offspring: Variance in offspring number
        Returns:
            Effective population size (Ne)
        """
        n_males = int(census_size * sex_ratio)
        n_females = census_size - n_males

        if n_males == 0 or n_females == 0:
            return 0

        # Sex ratio adjustment
        Ne_sex = 4 * n_males * n_females / (n_males + n_females)

        # Variance in offspring adjustment
        Ne_variance = (4 * census_size) / (2 + variance_offspring)

        # Use minimum as conservative estimate
        Ne = min(Ne_sex, Ne_variance)

        return Ne

    def fixation_probability(self, initial_freq: float, selection: float,
                           population_size: int) -> float:
        """Calculate probability of allele fixation

        Args:
            initial_freq: Starting allele frequency
            selection: Selection coefficient
            population_size: Effective population size
        Returns:
            Probability of fixation
        """
        if selection == 0:
            # Neutral allele
            return initial_freq
        else:
            # Selected allele (Kimura formula)
            s = selection
            Ne = population_size
            p = initial_freq

            if abs(4 * Ne * s) < 0.01:
                # Weak selection approximation
                return p * (1 + 2 * Ne * s * p)
            else:
                # Full formula
                numerator = 1 - math.exp(-4 * Ne * s * p)
                denominator = 1 - math.exp(-4 * Ne * s)
                return numerator / denominator

    def fst_calculation(self, subpop_allele_freqs: List[float],
                       subpop_sizes: List[int] = None) -> float:
        """Calculate FST (fixation index) for population structure

        Args:
            subpop_allele_freqs: Allele frequencies in each subpopulation
            subpop_sizes: Size of each subpopulation (if None, assumed equal)
        Returns:
            FST value (0 = no differentiation, 1 = complete differentiation)
        """
        n_pops = len(subpop_allele_freqs)

        if n_pops < 2:
            return 0

        if subpop_sizes is None:
            subpop_sizes = [1] * n_pops

        # Calculate weighted mean allele frequency
        total_size = sum(subpop_sizes)
        p_bar = sum(p * n for p, n in zip(subpop_allele_freqs, subpop_sizes)) / total_size

        # Calculate variance components
        # Expected heterozygosity in total population
        Ht = 2 * p_bar * (1 - p_bar)

        # Average expected heterozygosity within subpopulations
        Hs = sum(2 * p * (1 - p) * n for p, n in zip(subpop_allele_freqs, subpop_sizes)) / total_size

        # FST
        if Ht > 0:
            Fst = (Ht - Hs) / Ht
        else:
            Fst = 0

        return max(0, min(1, Fst))  # Ensure in [0, 1]

    def tajima_d(self, n_sequences: int, n_segregating_sites: int,
                pairwise_differences: float) -> float:
        """Calculate Tajima's D statistic for detecting selection

        Args:
            n_sequences: Number of DNA sequences
            n_segregating_sites: Number of polymorphic sites
            pairwise_differences: Average pairwise differences
        Returns:
            Tajima's D statistic
        """
        n = n_sequences
        S = n_segregating_sites

        if S == 0:
            return 0

        # Calculate a1 and a2
        a1 = sum(1/i for i in range(1, n))
        a2 = sum(1/i**2 for i in range(1, n))

        # Calculate b1 and b2
        b1 = (n + 1) / (3 * (n - 1))
        b2 = 2 * (n**2 + n + 3) / (9 * n * (n - 1))

        # Calculate c1 and c2
        c1 = b1 - 1/a1
        c2 = b2 - (n + 2)/(a1 * n) + a2/a1**2

        # Calculate e1 and e2
        e1 = c1/a1
        e2 = c2/(a1**2 + a2)

        # Calculate theta estimates
        theta_pi = pairwise_differences  # Tajima's estimator
        theta_w = S / a1  # Watterson's estimator

        # Calculate variance
        var_d = math.sqrt(e1 * S + e2 * S * (S - 1))

        # Tajima's D
        if var_d > 0:
            D = (theta_pi - theta_w) / var_d
        else:
            D = 0

        return D


class MutationRateEstimator:
    """Estimate mutation rates from genetic data"""

    def __init__(self):
        self.methods = ['direct', 'phylogenetic', 'pedigree']

    def direct_estimation(self, n_mutations: int, n_sites: int,
                         n_generations: int) -> Dict:
        """Direct estimation of mutation rate from observed mutations

        Args:
            n_mutations: Number of observed mutations
            n_sites: Number of sites examined
            n_generations: Number of generations
        Returns:
            Mutation rate estimates
        """
        if n_sites == 0 or n_generations == 0:
            return {'rate': 0, 'ci_lower': 0, 'ci_upper': 0}

        # Point estimate
        rate = n_mutations / (n_sites * n_generations)

        # Confidence interval (Poisson)
        if n_mutations > 0:
            ci_lower = chi2.ppf(0.025, 2 * n_mutations) / (2 * n_sites * n_generations)
            ci_upper = chi2.ppf(0.975, 2 * (n_mutations + 1)) / (2 * n_sites * n_generations)
        else:
            ci_lower = 0
            ci_upper = -math.log(0.05) / (n_sites * n_generations)

        return {
            'rate': rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'rate_per_site_per_generation': rate
        }

    def drake_equation(self, genome_size: float, mutation_rate_per_site: float) -> float:
        """Drake's equation for genomic mutation rate

        U = μ * G
        where U = genomic mutation rate, μ = per-site rate, G = genome size
        """
        return mutation_rate_per_site * genome_size

    def mutation_accumulation(self, initial_fitness: float, generations: int,
                            beneficial_rate: float = 0.01,
                            deleterious_rate: float = 0.99) -> Dict:
        """Model mutation accumulation over time

        Args:
            initial_fitness: Starting fitness
            generations: Number of generations
            beneficial_rate: Fraction of beneficial mutations
            deleterious_rate: Fraction of deleterious mutations
        Returns:
            Fitness trajectory
        """
        fitness = np.zeros(generations)
        fitness[0] = initial_fitness

        mutations_accumulated = np.zeros(generations)

        for gen in range(1, generations):
            # Random mutations
            n_mutations = np.random.poisson(1.0)  # Average 1 mutation per generation

            for _ in range(n_mutations):
                if np.random.random() < beneficial_rate:
                    # Beneficial mutation
                    effect = np.random.exponential(0.01)
                    fitness[gen] = fitness[gen-1] * (1 + effect)
                elif np.random.random() < deleterious_rate:
                    # Deleterious mutation
                    effect = np.random.exponential(0.02)
                    fitness[gen] = fitness[gen-1] * (1 - effect)
                else:
                    # Neutral mutation
                    fitness[gen] = fitness[gen-1]

            mutations_accumulated[gen] = mutations_accumulated[gen-1] + n_mutations

            # Ensure fitness doesn't go below 0
            fitness[gen] = max(0, fitness[gen])

        # Muller's ratchet (irreversible accumulation of deleterious mutations)
        ratchet_clicks = np.sum(np.diff(fitness) < 0)

        return {
            'fitness': fitness,
            'mutations_accumulated': mutations_accumulated,
            'final_fitness': fitness[-1],
            'fitness_decline': initial_fitness - fitness[-1],
            'mullers_ratchet_clicks': ratchet_clicks,
            'mutation_load': 1 - fitness[-1]/initial_fitness
        }


class GeneticsLab:
    """Main genetics laboratory interface"""

    def __init__(self):
        self.mendelian = MendelianInheritance()
        self.hwe = HardyWeinbergEquilibrium()
        self.linkage = LinkageAnalysis()
        self.qtl = QTLMapping()
        self.population = PopulationGenetics()
        self.mutation = MutationRateEstimator()
        self.results = {}

    def analyze_cross(self, parent1: str, parent2: str, cross_type: str = 'monohybrid') -> Dict:
        """Analyze genetic cross"""
        if cross_type == 'monohybrid':
            results = self.mendelian.punnett_square(parent1, parent2)
        else:  # dihybrid
            results = self.mendelian.dihybrid_cross(parent1, parent2)

        self.results['cross_analysis'] = results
        return results

    def test_hwe(self, genotype_counts: Dict) -> Dict:
        """Test for Hardy-Weinberg equilibrium"""
        results = self.hwe.chi_square_test(genotype_counts)
        self.results['hwe_test'] = results
        return results

    def map_qtl(self, markers: List[Dict], phenotypes: np.ndarray) -> Dict:
        """Map quantitative trait loci"""
        qtl_results = self.qtl.interval_mapping(markers, phenotypes)
        threshold = self.qtl.permutation_threshold(markers, phenotypes, n_permutations=100)

        # Find significant QTLs
        significant_qtls = [q for q in qtl_results if q['lod_score'] > threshold]

        results = {
            'qtl_scan': qtl_results,
            'threshold': threshold,
            'significant_qtls': significant_qtls,
            'n_qtls': len(significant_qtls)
        }

        self.results['qtl_mapping'] = results
        return results

    def simulate_evolution(self, initial_freq: float = 0.5, pop_size: int = 100,
                          generations: int = 100, selection: float = 0.01) -> Dict:
        """Simulate allele frequency evolution"""
        frequencies = self.population.wright_fisher_simulation(
            initial_freq, pop_size, generations, selection
        )

        # Calculate fixation probability
        fix_prob = self.population.fixation_probability(initial_freq, selection, pop_size)

        results = {
            'frequencies': frequencies,
            'initial_freq': initial_freq,
            'final_freq': frequencies[-1],
            'fixed': frequencies[-1] in [0, 1],
            'fixation_probability': fix_prob,
            'generations_to_fixation': np.where(np.isin(frequencies, [0, 1]))[0][0] if frequencies[-1] in [0, 1] else None
        }

        self.results['evolution_simulation'] = results
        return results

    def estimate_mutation_rate(self, n_mutations: int, n_sites: int, n_generations: int) -> Dict:
        """Estimate mutation rate"""
        rate_estimate = self.mutation.direct_estimation(n_mutations, n_sites, n_generations)

        # Simulate mutation accumulation
        accumulation = self.mutation.mutation_accumulation(1.0, 100)

        results = {
            'rate_estimate': rate_estimate,
            'accumulation_simulation': accumulation
        }

        self.results['mutation_analysis'] = results
        return results

    def population_structure_analysis(self, subpop_freqs: List[float]) -> Dict:
        """Analyze population structure"""
        fst = self.population.fst_calculation(subpop_freqs)

        # Test for Wahlund effect
        subpops = [{'p': p, 'weight': 1/len(subpop_freqs)} for p in subpop_freqs]
        wahlund = self.hwe.wahlund_effect(subpops)

        results = {
            'Fst': fst,
            'differentiation_level': 'Low' if fst < 0.05 else 'Moderate' if fst < 0.15 else 'High',
            'wahlund_effect': wahlund
        }

        self.results['population_structure'] = results
        return results


def run_demo():
    """Demonstrate genetics lab capabilities"""
    print("GENETICS LAB - Production Demo")
    print("=" * 60)

    lab = GeneticsLab()

    # 1. Mendelian inheritance
    print("\n1. MENDELIAN INHERITANCE")
    print("-" * 40)
    cross = lab.analyze_cross('Aa', 'Aa', 'monohybrid')
    print(f"Monohybrid cross: Aa × Aa")
    for genotype, prob in cross.items():
        print(f"  {genotype}: {prob:.2%}")

    dihybrid = lab.analyze_cross('AaBb', 'AaBb', 'dihybrid')
    print(f"\nDihybrid cross: AaBb × AaBb")
    print(f"Phenotype ratios: {dihybrid['phenotypes']}")

    # 2. Hardy-Weinberg equilibrium
    print("\n2. HARDY-WEINBERG EQUILIBRIUM")
    print("-" * 40)
    observed = {'AA': 250, 'Aa': 500, 'aa': 250}
    hwe_test = lab.test_hwe(observed)
    print(f"Observed genotypes: {observed}")
    print(f"Chi-square: {hwe_test['chi_square']:.3f}")
    print(f"P-value: {hwe_test['p_value']:.4f}")
    print(f"In equilibrium: {hwe_test['in_equilibrium']}")

    # 3. Linkage analysis
    print("\n3. LINKAGE ANALYSIS")
    print("-" * 40)
    recombinant = 35
    parental = 165
    rf = lab.linkage.recombination_frequency(recombinant, recombinant + parental)
    distance = lab.linkage.genetic_distance(rf)
    lod = lab.linkage.lod_score(recombinant, parental, rf)
    print(f"Recombinant: {recombinant}, Parental: {parental}")
    print(f"Recombination frequency: {rf:.3f}")
    print(f"Genetic distance: {distance:.1f} cM")
    print(f"LOD score: {lod:.2f}")

    # 4. QTL mapping
    print("\n4. QTL MAPPING")
    print("-" * 40)
    # Simulate marker and phenotype data
    np.random.seed(42)
    markers = [
        {'position': 0, 'genotypes': np.random.randint(0, 3, 100)},
        {'position': 10, 'genotypes': np.random.randint(0, 3, 100)},
        {'position': 20, 'genotypes': np.random.randint(0, 3, 100)}
    ]
    phenotypes = np.random.normal(100, 10, 100)
    qtl_results = lab.map_qtl(markers, phenotypes)
    print(f"QTL threshold: {qtl_results['threshold']:.2f}")
    print(f"Significant QTLs found: {qtl_results['n_qtls']}")

    # 5. Population genetics
    print("\n5. POPULATION GENETICS SIMULATION")
    print("-" * 40)
    evolution = lab.simulate_evolution(initial_freq=0.2, pop_size=200,
                                      generations=100, selection=0.05)
    print(f"Initial frequency: {evolution['initial_freq']}")
    print(f"Final frequency: {evolution['final_freq']:.3f}")
    print(f"Fixed: {evolution['fixed']}")
    print(f"Fixation probability: {evolution['fixation_probability']:.3f}")

    # 6. Population structure
    print("\n6. POPULATION STRUCTURE ANALYSIS")
    print("-" * 40)
    subpop_frequencies = [0.2, 0.3, 0.6, 0.5]
    structure = lab.population_structure_analysis(subpop_frequencies)
    print(f"Subpopulation allele frequencies: {subpop_frequencies}")
    print(f"FST: {structure['Fst']:.4f}")
    print(f"Differentiation level: {structure['differentiation_level']}")
    print(f"Heterozygote deficit (Wahlund): {structure['wahlund_effect']['heterozygote_deficit']:.4f}")

    # 7. Mutation rate estimation
    print("\n7. MUTATION RATE ESTIMATION")
    print("-" * 40)
    mutation = lab.estimate_mutation_rate(n_mutations=5, n_sites=1000000, n_generations=100)
    rate = mutation['rate_estimate']
    print(f"Observed mutations: 5")
    print(f"Sites examined: 1,000,000")
    print(f"Generations: 100")
    print(f"Mutation rate: {rate['rate']:.2e} per site per generation")
    print(f"95% CI: [{rate['ci_lower']:.2e}, {rate['ci_upper']:.2e}]")
    print(f"Mutation load after 100 generations: {mutation['accumulation_simulation']['mutation_load']:.3f}")

    print("\n" + "=" * 60)
    print("Demo complete! Lab ready for production use.")


if __name__ == '__main__':
    run_demo()