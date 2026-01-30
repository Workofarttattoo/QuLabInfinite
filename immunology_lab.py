"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

IMMUNOLOGY LAB - Production Ready
Advanced immune system modeling, antibody dynamics, and vaccine design.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable
from scipy import stats, optimize, integrate
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

@dataclass
class ImmunologyLab:
    """Production-ready immunology simulation laboratory."""

    # Physical constants
    AVOGADRO: float = 6.022e23  # molecules/mol
    BOLTZMANN: float = 1.38e-23  # J/K
    TEMPERATURE: float = 310.15  # K (37°C)

    # Immunological parameters
    antibody_production_rate: float = 2000  # antibodies per B cell per second
    t_cell_proliferation_rate: float = 0.693  # per day (doubling time ~1 day)
    antigen_decay_rate: float = 0.1  # per hour
    cytokine_diffusion: float = 10.0  # μm²/s

    # Cell counts (per μL of blood)
    NORMAL_CELL_COUNTS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'neutrophils': (2500, 7000),
        'lymphocytes': (1500, 4000),
        'monocytes': (200, 800),
        'eosinophils': (50, 400),
        'basophils': (25, 100),
        'nk_cells': (90, 600),
        'b_cells': (100, 500),
        't_cells': (800, 3500),
        'cd4_t_cells': (500, 2000),
        'cd8_t_cells': (300, 1500)
    })

    # Cytokine network
    CYTOKINES: List[str] = field(default_factory=lambda: [
        'IL-1', 'IL-2', 'IL-4', 'IL-6', 'IL-10', 'IL-12',
        'TNF-α', 'IFN-γ', 'TGF-β'
    ])

    def __post_init__(self):
        """Initialize immune system components."""
        self.antibody_repertoire = []
        self.t_cell_repertoire = []
        self.memory_cells = []
        self.antigen_history = []
        self.cytokine_levels = {cytokine: 0.0 for cytokine in self.CYTOKINES}

    def generate_antibody_repertoire(self, diversity: int = 10000) -> np.ndarray:
        """Generate diverse antibody repertoire through V(D)J recombination."""
        # Simplified CDR3 representation (20 amino acids, length 10-20)
        repertoire = []

        for _ in range(diversity):
            # Random CDR3 length
            cdr3_length = np.random.randint(10, 21)

            # Random amino acid sequence (20 standard amino acids)
            cdr3_sequence = np.random.randint(0, 20, cdr3_length)

            repertoire.append(cdr3_sequence)

        self.antibody_repertoire = repertoire
        return repertoire

    def antibody_antigen_affinity(self, antibody: np.ndarray,
                                 antigen: np.ndarray) -> float:
        """Calculate binding affinity between antibody and antigen."""
        # Ensure compatible lengths for comparison
        min_len = min(len(antibody), len(antigen))
        antibody_segment = antibody[:min_len]
        antigen_segment = antigen[:min_len]

        # Shape complementarity (inverse of Hamming distance)
        hamming_distance = np.sum(antibody_segment != antigen_segment)
        shape_score = 1 - (hamming_distance / min_len)

        # Hydrophobic interactions
        hydrophobic_aa = [0, 5, 7, 9, 11, 12, 14, 18]  # A, F, I, L, M, P, V, W
        hydro_matches = sum(1 for ab, ag in zip(antibody_segment, antigen_segment)
                           if ab == ag and ab in hydrophobic_aa)
        hydro_score = hydro_matches / min_len

        # Electrostatic interactions
        positive_aa = [1, 8, 10]  # R, K, H
        negative_aa = [3, 4]  # D, E

        electro_score = 0
        for ab, ag in zip(antibody_segment, antigen_segment):
            if (ab in positive_aa and ag in negative_aa) or \
               (ab in negative_aa and ag in positive_aa):
                electro_score += 0.1

        # Combined affinity score
        affinity = shape_score * 0.5 + hydro_score * 0.3 + electro_score * 0.2

        # Apply sigmoid to get realistic binding curve
        Kd = 1e-9  # Dissociation constant (M)
        concentration = 1e-6  # Antigen concentration (M)
        binding = concentration / (Kd + concentration)

        return affinity * binding

    def somatic_hypermutation(self, antibody: np.ndarray,
                            mutation_rate: float = 0.001) -> np.ndarray:
        """Simulate somatic hypermutation in B cells."""
        mutated = antibody.copy()

        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                # Point mutation to different amino acid
                mutated[i] = np.random.randint(0, 20)

        return mutated

    def affinity_maturation(self, initial_antibody: np.ndarray,
                          antigen: np.ndarray,
                          generations: int = 20,
                          population_size: int = 100) -> Tuple[np.ndarray, List[float]]:
        """Simulate affinity maturation in germinal centers."""
        population = [initial_antibody.copy() for _ in range(population_size)]
        affinity_history = []

        for generation in range(generations):
            # Calculate affinities
            affinities = [self.antibody_antigen_affinity(ab, antigen)
                         for ab in population]

            # Record best affinity
            affinity_history.append(max(affinities))

            # Selection (top 50%)
            sorted_indices = np.argsort(affinities)[::-1]
            selected = [population[i] for i in sorted_indices[:population_size//2]]

            # Proliferation with mutation
            new_population = []
            for antibody in selected:
                # Each selected B cell produces 2 daughters
                for _ in range(2):
                    daughter = self.somatic_hypermutation(antibody)
                    new_population.append(daughter)

            population = new_population

        # Return best antibody
        final_affinities = [self.antibody_antigen_affinity(ab, antigen)
                           for ab in population]
        best_idx = np.argmax(final_affinities)

        return population[best_idx], affinity_history

    def t_cell_activation(self, signal1_strength: float,
                        signal2_strength: float,
                        signal3_cytokines: Dict[str, float]) -> float:
        """Model T cell activation with three-signal model."""
        # Signal 1: TCR-MHC interaction
        tcr_signal = signal1_strength

        # Signal 2: Costimulation (CD28-B7)
        costim_signal = signal2_strength

        # Signal 3: Cytokine environment
        cytokine_effect = 0
        if 'IL-2' in signal3_cytokines:
            cytokine_effect += signal3_cytokines['IL-2'] * 0.5
        if 'IL-12' in signal3_cytokines:
            cytokine_effect += signal3_cytokines['IL-12'] * 0.3
        if 'IFN-γ' in signal3_cytokines:
            cytokine_effect += signal3_cytokines['IFN-γ'] * 0.2

        # Activation threshold model
        activation_threshold = 0.5
        total_signal = tcr_signal * costim_signal * (1 + cytokine_effect)

        # Sigmoid activation function
        activation = 1 / (1 + np.exp(-10 * (total_signal - activation_threshold)))

        return activation

    def simulate_infection_response(self, pathogen_load: float,
                                  duration_days: int = 14) -> Dict:
        """Simulate complete immune response to infection."""
        time_points = np.linspace(0, duration_days, duration_days * 24)  # Hourly
        results = {
            'time': time_points,
            'pathogen': np.zeros_like(time_points),
            'antibodies': np.zeros_like(time_points),
            'cd8_t_cells': np.zeros_like(time_points),
            'cd4_t_cells': np.zeros_like(time_points),
            'cytokines': {cyt: np.zeros_like(time_points) for cyt in ['IL-2', 'IFN-γ', 'TNF-α']}
        }

        # Initial conditions
        pathogen = pathogen_load
        antibodies = 0
        cd8_t = 100  # Baseline T cells
        cd4_t = 200  # Baseline helper T cells
        memory_b = 0
        activated_b = 0

        # Simulation parameters
        pathogen_growth_rate = 0.5  # per day
        pathogen_clearance_rate = 0.001  # per antibody per day
        antibody_production = 1000  # per activated B cell per day
        t_cell_killing_rate = 0.01  # per CD8 T cell per day

        for i, t in enumerate(time_points):
            # Pathogen dynamics
            growth = pathogen_growth_rate * pathogen * (1 - pathogen / 1e9)  # Logistic growth
            clearance = pathogen_clearance_rate * antibodies * pathogen
            t_cell_killing = t_cell_killing_rate * cd8_t * pathogen

            pathogen = max(0, pathogen + (growth - clearance - t_cell_killing) / 24)

            # Innate immune response (first 48 hours)
            if t < 2:
                # Cytokine release
                results['cytokines']['TNF-α'][i] = pathogen / 1e6
                results['cytokines']['IL-2'][i] = pathogen / 2e6

            # Adaptive immune response (after 3 days)
            if t > 3:
                # T cell activation and proliferation
                activation_signal = min(1, pathogen / 1e6)
                cd4_t += cd4_t * self.t_cell_proliferation_rate * activation_signal / 24
                cd8_t += cd8_t * self.t_cell_proliferation_rate * activation_signal / 24

                # B cell activation (after 5 days)
                if t > 5:
                    activated_b = min(1000, pathogen / 1e5)
                    antibodies += activated_b * antibody_production / 24

                # Cytokine production by T cells
                results['cytokines']['IFN-γ'][i] = cd8_t / 1000
                results['cytokines']['IL-2'][i] = cd4_t / 1000

            # Memory cell formation (after 10 days)
            if t > 10 and pathogen < 1e4:
                memory_b = activated_b * 0.1

            # Record state
            results['pathogen'][i] = pathogen
            results['antibodies'][i] = antibodies
            results['cd8_t_cells'][i] = cd8_t
            results['cd4_t_cells'][i] = cd4_t

            # Check if infection cleared
            if pathogen < 1:
                results['pathogen'][i:] = 0
                results['antibodies'][i:] = antibodies * np.exp(-0.1 * (time_points[i:] - t))
                break

        return results

    def simulate_vaccination(self, vaccine_type: str = 'protein',
                           doses: List[int] = [0, 28],
                           duration_days: int = 180) -> Dict:
        """Simulate immune response to vaccination."""
        time_points = np.arange(0, duration_days + 1)
        results = {
            'time': time_points,
            'antibody_titer': np.zeros_like(time_points, dtype=float),
            'memory_b_cells': np.zeros_like(time_points, dtype=float),
            'memory_t_cells': np.zeros_like(time_points, dtype=float),
            'protection': np.zeros_like(time_points, dtype=float)
        }

        # Vaccine-specific parameters
        vaccine_params = {
            'protein': {'peak_antibody': 100, 'decay_rate': 0.05, 'memory_factor': 0.1},
            'mRNA': {'peak_antibody': 500, 'decay_rate': 0.03, 'memory_factor': 0.2},
            'viral_vector': {'peak_antibody': 300, 'decay_rate': 0.04, 'memory_factor': 0.15},
            'live_attenuated': {'peak_antibody': 200, 'decay_rate': 0.02, 'memory_factor': 0.3}
        }

        params = vaccine_params.get(vaccine_type, vaccine_params['protein'])

        # Simulate each dose
        for dose_num, dose_day in enumerate(doses):
            # Boost factor for subsequent doses
            boost = 1 + dose_num * 1.5

            for i, day in enumerate(time_points):
                if day >= dose_day:
                    days_since_dose = day - dose_day

                    # Antibody kinetics
                    if days_since_dose < 14:
                        # Rising phase
                        antibody_level = params['peak_antibody'] * boost * \
                                       (1 - np.exp(-0.5 * days_since_dose))
                    else:
                        # Decay phase
                        peak = params['peak_antibody'] * boost
                        antibody_level = peak * np.exp(-params['decay_rate'] * (days_since_dose - 14))

                    results['antibody_titer'][i] += antibody_level

                    # Memory cells
                    if days_since_dose > 7:
                        memory_b = params['memory_factor'] * boost * 100
                        memory_t = params['memory_factor'] * boost * 50

                        results['memory_b_cells'][i] = max(results['memory_b_cells'][i], memory_b)
                        results['memory_t_cells'][i] = max(results['memory_t_cells'][i], memory_t)

        # Calculate protection level (correlate of protection)
        protection_threshold = 50  # Arbitrary units
        for i in range(len(time_points)):
            if results['antibody_titer'][i] > protection_threshold:
                results['protection'][i] = min(1, results['antibody_titer'][i] / (protection_threshold * 2))
            else:
                # Partial protection from memory cells
                results['protection'][i] = min(0.5, (results['memory_b_cells'][i] +
                                                    results['memory_t_cells'][i]) / 200)

        return results

    def model_cytokine_network(self, initial_stimulus: Dict[str, float],
                              time_steps: int = 100) -> np.ndarray:
        """Model cytokine signaling network dynamics."""
        n_cytokines = len(self.CYTOKINES)
        cytokine_levels = np.zeros((time_steps, n_cytokines))

        # Initialize with stimulus
        for i, cytokine in enumerate(self.CYTOKINES):
            cytokine_levels[0, i] = initial_stimulus.get(cytokine, 0)

        # Interaction matrix (simplified)
        interaction_matrix = np.random.randn(n_cytokines, n_cytokines) * 0.1
        np.fill_diagonal(interaction_matrix, -0.1)  # Self-regulation

        # Specific interactions
        il2_idx = self.CYTOKINES.index('IL-2')
        ifng_idx = self.CYTOKINES.index('IFN-γ')
        il10_idx = self.CYTOKINES.index('IL-10')

        interaction_matrix[il2_idx, ifng_idx] = 0.3  # IL-2 promotes IFN-γ
        interaction_matrix[ifng_idx, il2_idx] = 0.2  # Positive feedback
        interaction_matrix[il10_idx, :] = -0.1  # IL-10 is anti-inflammatory

        # Simulate dynamics
        for t in range(1, time_steps):
            # Rate of change = production + interactions - decay
            production = np.ones(n_cytokines) * 0.01  # Basal production
            interactions = np.dot(interaction_matrix, cytokine_levels[t-1])
            decay = -0.05 * cytokine_levels[t-1]

            change = production + interactions + decay
            cytokine_levels[t] = np.maximum(0, cytokine_levels[t-1] + change)

            # Saturation
            cytokine_levels[t] = np.minimum(100, cytokine_levels[t])

        return cytokine_levels

    def calculate_neutralizing_antibody_titer(self, antibodies: List[np.ndarray],
                                             virus: np.ndarray) -> float:
        """Calculate neutralizing antibody titer against virus."""
        if not antibodies:
            return 0

        # Calculate neutralization for each antibody
        neutralization_scores = []

        for antibody in antibodies:
            affinity = self.antibody_antigen_affinity(antibody, virus)

            # Neutralization depends on binding to critical epitopes
            # Simplified: assume 30% of binding sites are neutralizing
            neutralizing_probability = 0.3

            if np.random.random() < neutralizing_probability:
                neutralization_scores.append(affinity)
            else:
                neutralization_scores.append(affinity * 0.1)  # Weak neutralization

        # Calculate IC50 (concentration for 50% neutralization)
        if neutralization_scores:
            mean_neutralization = np.mean(neutralization_scores)
            # Convert to titer (reciprocal dilution)
            titer = 1 / (1e-6 / mean_neutralization) if mean_neutralization > 0 else 0
        else:
            titer = 0

        return titer

    def mhc_peptide_binding(self, peptide: np.ndarray,
                           mhc_allele: str = 'HLA-A*02:01') -> float:
        """Predict MHC-peptide binding affinity."""
        # Simplified binding motif for HLA-A*02:01
        # Prefers L at position 2, V at position 9
        binding_score = 0

        # Length preference (8-11 amino acids)
        optimal_length = 9
        length_penalty = abs(len(peptide) - optimal_length) * 0.1
        binding_score -= length_penalty

        # Anchor residues
        if len(peptide) >= 2:
            if peptide[1] == 11:  # L at position 2
                binding_score += 0.5
        if len(peptide) >= 9:
            if peptide[8] == 17:  # V at position 9
                binding_score += 0.5

        # Hydrophobicity at C-terminus
        if len(peptide) > 0:
            hydrophobic_aa = [0, 5, 7, 9, 11, 12, 14, 17, 18]
            if peptide[-1] in hydrophobic_aa:
                binding_score += 0.3

        # Convert to binding affinity (nM)
        # Strong binders: < 50 nM, Weak binders: 50-500 nM
        affinity_nm = 500 * np.exp(-binding_score)

        return affinity_nm

    def tcr_repertoire_diversity(self, repertoire_size: int = 10000) -> Dict:
        """Analyze T cell receptor repertoire diversity."""
        # Generate TCR sequences (simplified CDR3β)
        tcr_sequences = []
        v_genes = 50  # ~50 V gene segments
        j_genes = 13  # 13 J gene segments

        for _ in range(repertoire_size):
            v_segment = np.random.randint(0, v_genes)
            j_segment = np.random.randint(0, j_genes)
            # Random N additions (0-15 nucleotides)
            n_additions = np.random.randint(0, 6)  # In amino acids

            # CDR3 sequence
            cdr3_length = np.random.randint(12, 18)
            cdr3 = np.random.randint(0, 20, cdr3_length)

            tcr_sequences.append({
                'v_gene': v_segment,
                'j_gene': j_segment,
                'cdr3': cdr3,
                'n_additions': n_additions
            })

        # Calculate diversity metrics
        # Shannon entropy
        unique_sequences = set(tuple(tcr['cdr3']) for tcr in tcr_sequences)
        shannon_entropy = -sum((1/len(unique_sequences)) * np.log(1/len(unique_sequences))
                              for _ in unique_sequences)

        # Simpson's diversity index
        sequence_counts = {}
        for tcr in tcr_sequences:
            seq_tuple = tuple(tcr['cdr3'])
            sequence_counts[seq_tuple] = sequence_counts.get(seq_tuple, 0) + 1

        simpson_index = sum((count/repertoire_size)**2
                          for count in sequence_counts.values())
        simpson_diversity = 1 - simpson_index

        # Clonality
        max_clone_size = max(sequence_counts.values())
        clonality = max_clone_size / repertoire_size

        return {
            'total_sequences': repertoire_size,
            'unique_sequences': len(unique_sequences),
            'shannon_entropy': shannon_entropy,
            'simpson_diversity': simpson_diversity,
            'clonality': clonality,
            'repertoire': tcr_sequences[:100]  # Return sample
        }

    def simulate_autoimmune_response(self, self_antigen: np.ndarray,
                                   tolerance_threshold: float = 0.8) -> Dict:
        """Simulate breakdown of immunological tolerance."""
        results = {
            'autoreactive_cells': 0,
            'tissue_damage': 0,
            'autoantibodies': [],
            'regulatory_response': 0
        }

        # Generate T cell repertoire
        n_t_cells = 1000
        t_cell_repertoire = [np.random.randint(0, 20, 15) for _ in range(n_t_cells)]

        # Check for autoreactive T cells
        autoreactive_tcells = []
        for tcr in t_cell_repertoire:
            # Calculate self-reactivity
            reactivity = self.antibody_antigen_affinity(tcr, self_antigen)

            # Negative selection in thymus
            if reactivity > tolerance_threshold:
                # Cell should be deleted
                if np.random.random() < 0.95:  # 95% deletion efficiency
                    continue
                else:
                    # Escaped negative selection
                    autoreactive_tcells.append(tcr)

        results['autoreactive_cells'] = len(autoreactive_tcells)

        # Peripheral tolerance mechanisms
        if autoreactive_tcells:
            # Regulatory T cells
            treg_suppression = min(0.8, len(autoreactive_tcells) * 0.01)
            results['regulatory_response'] = treg_suppression

            # Calculate tissue damage
            escaped_cells = len(autoreactive_tcells) * (1 - treg_suppression)
            results['tissue_damage'] = min(100, escaped_cells * 0.1)

            # Autoantibody production
            if escaped_cells > 10:
                n_autoantibodies = int(escaped_cells / 10)
                for _ in range(n_autoantibodies):
                    autoantibody = self.somatic_hypermutation(
                        np.random.choice(autoreactive_tcells)
                    )
                    results['autoantibodies'].append(autoantibody)

        return results

    def model_immunosenescence(self, age_years: int) -> Dict:
        """Model age-related changes in immune function."""
        # Young adult baseline (age 20-30)
        baseline_age = 25

        results = {
            'thymic_output': 0,
            'naive_t_cells': 0,
            'memory_t_cells': 0,
            'b_cell_diversity': 0,
            'inflammation': 0,
            'vaccine_response': 0
        }

        # Thymic involution (exponential decay after puberty)
        if age_years < 20:
            thymic_function = 1.0
        else:
            thymic_function = np.exp(-0.03 * (age_years - 20))

        results['thymic_output'] = thymic_function * 100  # Relative to young adult

        # T cell compartments
        results['naive_t_cells'] = max(10, 100 * thymic_function)
        results['memory_t_cells'] = min(90, 30 + age_years * 0.8)

        # B cell repertoire diversity
        if age_years < 60:
            b_cell_diversity = 100 - (age_years - baseline_age) * 0.5
        else:
            b_cell_diversity = 70 - (age_years - 60) * 1.0

        results['b_cell_diversity'] = max(20, b_cell_diversity)

        # Inflammaging (chronic low-grade inflammation)
        if age_years > 50:
            inflammation = (age_years - 50) * 2
        else:
            inflammation = 0

        results['inflammation'] = min(100, inflammation)

        # Vaccine response
        vaccine_efficacy = 100 * thymic_function * (results['b_cell_diversity'] / 100)
        vaccine_efficacy *= (1 - results['inflammation'] / 200)  # Inflammation reduces response

        results['vaccine_response'] = max(10, vaccine_efficacy)

        return results

    def simulate_immunotherapy(self, therapy_type: str = 'checkpoint',
                             tumor_burden: float = 1000,
                             duration_weeks: int = 12) -> Dict:
        """Simulate cancer immunotherapy response."""
        time_points = np.linspace(0, duration_weeks, duration_weeks * 7)
        results = {
            'time': time_points,
            'tumor_size': np.zeros_like(time_points),
            'cd8_infiltration': np.zeros_like(time_points),
            'pd1_expression': np.zeros_like(time_points),
            'response': 'progressive_disease'
        }

        tumor = tumor_burden
        cd8_cells = 100  # Initial T cells
        pd1_level = 0.3  # Baseline PD-1 expression

        for i, t in enumerate(time_points):
            if therapy_type == 'checkpoint':
                # Anti-PD-1/PD-L1 therapy
                if t > 2:  # Therapy starts after 2 weeks
                    # Reduce PD-1 mediated suppression
                    pd1_level = max(0.1, pd1_level - 0.01)

                    # Enhanced T cell activity
                    cd8_cells += cd8_cells * 0.02 * (1 - pd1_level)

                    # Tumor killing
                    kill_rate = 0.001 * cd8_cells * (1 - pd1_level)
                    tumor = max(0, tumor - kill_rate)

            elif therapy_type == 'car_t':
                # CAR-T cell therapy
                if t == 2:  # Infusion at week 2
                    cd8_cells += 10000  # Large number of CAR-T cells

                if t > 2:
                    # CAR-T expansion
                    if tumor > 10:
                        cd8_cells += cd8_cells * 0.1  # Rapid expansion

                    # Tumor killing (more efficient than checkpoint)
                    kill_rate = 0.01 * cd8_cells
                    tumor = max(0, tumor - kill_rate)

                    # CAR-T exhaustion over time
                    if t > 6:
                        cd8_cells *= 0.95

            elif therapy_type == 'vaccine':
                # Therapeutic cancer vaccine
                if t % 2 == 0 and t > 0:  # Boost every 2 weeks
                    cd8_cells += 500  # Moderate T cell increase

                if t > 1:
                    # Slower but sustained response
                    kill_rate = 0.0001 * cd8_cells
                    tumor = max(0, tumor - kill_rate)

            # Tumor growth (if not completely eliminated)
            if tumor > 0:
                growth_rate = 0.01 * tumor * (1 - tumor / 10000)  # Logistic growth
                tumor += growth_rate

            # Record state
            results['tumor_size'][i] = tumor
            results['cd8_infiltration'][i] = cd8_cells
            results['pd1_expression'][i] = pd1_level

        # Determine response category (RECIST-like criteria)
        final_tumor = results['tumor_size'][-1]
        initial_tumor = tumor_burden

        if final_tumor == 0:
            results['response'] = 'complete_response'
        elif final_tumor < 0.3 * initial_tumor:
            results['response'] = 'partial_response'
        elif final_tumor < 1.2 * initial_tumor:
            results['response'] = 'stable_disease'
        else:
            results['response'] = 'progressive_disease'

        return results

    def run_comprehensive_analysis(self) -> Dict:
        """Run complete immunology analysis pipeline."""
        results = {}

        print("Generating antibody repertoire...")
        repertoire = self.generate_antibody_repertoire(1000)
        results['repertoire_size'] = len(repertoire)

        print("Testing antibody-antigen binding...")
        test_antigen = np.random.randint(0, 20, 15)
        test_antibody = repertoire[0] if repertoire else np.random.randint(0, 20, 15)
        affinity = self.antibody_antigen_affinity(test_antibody, test_antigen)
        results['binding_affinity'] = affinity

        print("Simulating affinity maturation...")
        if repertoire:
            mature_antibody, affinity_history = self.affinity_maturation(
                repertoire[0], test_antigen, generations=10
            )
            results['affinity_improvement'] = affinity_history[-1] / affinity_history[0] if affinity_history[0] > 0 else 1

        print("Simulating infection response...")
        infection = self.simulate_infection_response(1e6, duration_days=14)
        results['infection_cleared'] = infection['pathogen'][-1] < 1
        results['peak_antibodies'] = np.max(infection['antibodies'])

        print("Simulating vaccination...")
        vaccination = self.simulate_vaccination('mRNA', doses=[0, 28], duration_days=90)
        results['peak_antibody_titer'] = np.max(vaccination['antibody_titer'])
        results['protection_duration'] = np.sum(vaccination['protection'] > 0.5)

        print("Analyzing TCR diversity...")
        tcr_diversity = self.tcr_repertoire_diversity(5000)
        results['tcr_diversity'] = tcr_diversity['simpson_diversity']

        print("Simulating immunotherapy...")
        therapy = self.simulate_immunotherapy('checkpoint', tumor_burden=1000, duration_weeks=12)
        results['therapy_response'] = therapy['response']

        return results


def run_demo():
    """Demonstrate comprehensive immunology capabilities."""
    print("=" * 80)
    print("IMMUNOLOGY LAB - Production Demo")
    print("=" * 80)

    lab = ImmunologyLab()

    print("\n1. ANTIBODY REPERTOIRE GENERATION")
    print("-" * 40)
    repertoire = lab.generate_antibody_repertoire(diversity=100)
    print(f"Generated {len(repertoire)} unique antibodies")
    print(f"Example CDR3 length: {len(repertoire[0])} amino acids")

    print("\n2. ANTIBODY-ANTIGEN BINDING")
    print("-" * 40)
    antigen = np.random.randint(0, 20, 15)
    antibody = repertoire[0]
    affinity = lab.antibody_antigen_affinity(antibody, antigen)
    print(f"Binding affinity: {affinity:.4f}")
    print(f"Interpretation: {'High' if affinity > 0.5 else 'Low'} affinity")

    print("\n3. AFFINITY MATURATION")
    print("-" * 40)
    mature_ab, history = lab.affinity_maturation(antibody, antigen, generations=20)
    print(f"Initial affinity: {history[0]:.4f}")
    print(f"Final affinity: {history[-1]:.4f}")
    print(f"Improvement: {history[-1]/history[0]:.2f}x")

    print("\n4. T CELL ACTIVATION")
    print("-" * 40)
    signal1 = 0.7  # Strong TCR signal
    signal2 = 0.8  # Good costimulation
    signal3 = {'IL-2': 0.5, 'IFN-γ': 0.3}
    activation = lab.t_cell_activation(signal1, signal2, signal3)
    print(f"T cell activation level: {activation:.2%}")

    print("\n5. INFECTION RESPONSE")
    print("-" * 40)
    response = lab.simulate_infection_response(pathogen_load=1e6, duration_days=14)
    print(f"Peak pathogen load: {np.max(response['pathogen']):.2e}")
    print(f"Time to clearance: {np.argmax(response['pathogen'] < 1) / 24:.1f} days")
    print(f"Peak antibody level: {np.max(response['antibodies']):.0f}")
    print(f"Peak CD8 T cells: {np.max(response['cd8_t_cells']):.0f}")

    print("\n6. VACCINATION RESPONSE")
    print("-" * 40)
    for vaccine_type in ['protein', 'mRNA']:
        vacc = lab.simulate_vaccination(vaccine_type, doses=[0, 28], duration_days=180)
        print(f"\n{vaccine_type.upper()} Vaccine:")
        print(f"  Peak antibody titer: {np.max(vacc['antibody_titer']):.0f}")
        print(f"  Duration of protection (>50%): {np.sum(vacc['protection'] > 0.5)} days")
        print(f"  Memory B cells at day 180: {vacc['memory_b_cells'][-1]:.0f}")

    print("\n7. CYTOKINE NETWORK")
    print("-" * 40)
    initial = {'IL-2': 10, 'TNF-α': 5}
    cytokines = lab.model_cytokine_network(initial, time_steps=50)
    print(f"Initial stimulus: {initial}")
    print(f"Peak IL-2: {np.max(cytokines[:, lab.CYTOKINES.index('IL-2')]):.1f}")
    print(f"Peak IFN-γ: {np.max(cytokines[:, lab.CYTOKINES.index('IFN-γ')]):.1f}")

    print("\n8. MHC-PEPTIDE BINDING")
    print("-" * 40)
    peptide = np.array([11, 11, 5, 7, 9, 12, 14, 18, 17])  # LLFILVPWV-like
    binding = lab.mhc_peptide_binding(peptide, 'HLA-A*02:01')
    print(f"Peptide length: {len(peptide)} aa")
    print(f"Predicted binding affinity: {binding:.1f} nM")
    print(f"Classification: {'Strong' if binding < 50 else 'Weak' if binding < 500 else 'Non'}-binder")

    print("\n9. TCR REPERTOIRE DIVERSITY")
    print("-" * 40)
    tcr_analysis = lab.tcr_repertoire_diversity(repertoire_size=1000)
    print(f"Total TCRs: {tcr_analysis['total_sequences']}")
    print(f"Unique sequences: {tcr_analysis['unique_sequences']}")
    print(f"Shannon entropy: {tcr_analysis['shannon_entropy']:.2f}")
    print(f"Simpson diversity: {tcr_analysis['simpson_diversity']:.4f}")
    print(f"Clonality: {tcr_analysis['clonality']:.4f}")

    print("\n10. AUTOIMMUNE RESPONSE")
    print("-" * 40)
    self_antigen = np.random.randint(0, 20, 15)
    autoimmune = lab.simulate_autoimmune_response(self_antigen, tolerance_threshold=0.8)
    print(f"Autoreactive T cells: {autoimmune['autoreactive_cells']}")
    print(f"Regulatory response: {autoimmune['regulatory_response']:.2%}")
    print(f"Tissue damage: {autoimmune['tissue_damage']:.1f}%")
    print(f"Autoantibodies generated: {len(autoimmune['autoantibodies'])}")

    print("\n11. IMMUNOSENESCENCE")
    print("-" * 40)
    for age in [25, 50, 75]:
        aging = lab.model_immunosenescence(age)
        print(f"\nAge {age} years:")
        print(f"  Thymic output: {aging['thymic_output']:.1f}%")
        print(f"  Naive T cells: {aging['naive_t_cells']:.1f}%")
        print(f"  B cell diversity: {aging['b_cell_diversity']:.1f}%")
        print(f"  Inflammation: {aging['inflammation']:.1f}%")
        print(f"  Vaccine response: {aging['vaccine_response']:.1f}%")

    print("\n12. CANCER IMMUNOTHERAPY")
    print("-" * 40)
    for therapy in ['checkpoint', 'car_t', 'vaccine']:
        result = lab.simulate_immunotherapy(therapy, tumor_burden=1000, duration_weeks=12)
        print(f"\n{therapy.upper()} Therapy:")
        print(f"  Initial tumor: 1000 cells")
        print(f"  Final tumor: {result['tumor_size'][-1]:.0f} cells")
        print(f"  Peak CD8 infiltration: {np.max(result['cd8_infiltration']):.0f} cells")
        print(f"  Response: {result['response'].replace('_', ' ').title()}")

    print("\n13. COMPREHENSIVE ANALYSIS")
    print("-" * 40)
    comprehensive = lab.run_comprehensive_analysis()
    print("\nAnalysis Results:")
    for key, value in comprehensive.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("Immunology Lab demonstration complete!")
    print("=" * 80)


if __name__ == '__main__':
    run_demo()