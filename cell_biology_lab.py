"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

CELL BIOLOGY LAB
Production-ready cell biology algorithms for cell cycle modeling, signaling pathways,
apoptosis kinetics, migration models, proliferation curves, and metabolic flux analysis.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import math
from scipy import integrate, optimize
from scipy.stats import norm, lognorm
import warnings

# Suppress integration warnings for cleaner output
warnings.filterwarnings('ignore', category=integrate.IntegrationWarning)


@dataclass
class CellState:
    """Represents the state of a cell"""
    phase: str = 'G1'  # G1, S, G2, M, G0, Apoptosis
    cycle_time: float = 0.0  # Hours in current phase
    atp_level: float = 100.0  # ATP concentration (arbitrary units)
    calcium_level: float = 0.1  # μM
    ros_level: float = 0.0  # Reactive oxygen species
    survival_signal: float = 1.0
    migration_speed: float = 0.0  # μm/hour
    metabolic_rate: float = 1.0
    dna_damage: float = 0.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))


class CellCycleModel:
    """Model cell cycle progression and checkpoints"""

    def __init__(self, cell_type: str = 'fibroblast'):
        self.cell_type = cell_type

        # Phase durations (hours) for different cell types
        self.phase_durations = {
            'fibroblast': {'G1': 11, 'S': 8, 'G2': 4, 'M': 1},
            'epithelial': {'G1': 12, 'S': 10, 'G2': 5, 'M': 1},
            'stem': {'G1': 8, 'S': 6, 'G2': 3, 'M': 1},
            'cancer': {'G1': 6, 'S': 5, 'G2': 2, 'M': 1}
        }

        # Get durations for specified cell type
        self.durations = self.phase_durations.get(
            cell_type,
            self.phase_durations['fibroblast']
        )

        # Checkpoint thresholds
        self.checkpoints = {
            'G1/S': {'cyclin_D': 0.5, 'p53': 0.3, 'nutrients': 0.4},
            'S/G2': {'dna_replication': 0.9, 'cyclin_A': 0.6},
            'G2/M': {'cyclin_B': 0.7, 'cdc2': 0.6, 'dna_integrity': 0.8},
            'M_exit': {'apc': 0.8, 'separase': 0.7}
        }

        # Cyclin dynamics parameters
        self.cyclin_params = {
            'cyclin_D': {'synthesis': 0.5, 'degradation': 0.1},
            'cyclin_E': {'synthesis': 0.6, 'degradation': 0.15},
            'cyclin_A': {'synthesis': 0.7, 'degradation': 0.2},
            'cyclin_B': {'synthesis': 0.8, 'degradation': 0.25}
        }

    def simulate_cycle(self, duration: float = 48, dt: float = 0.1) -> Dict:
        """Simulate cell cycle progression

        Args:
            duration: Simulation time in hours
            dt: Time step
        Returns:
            Cell cycle dynamics
        """
        steps = int(duration / dt)
        time = np.linspace(0, duration, steps)

        # Initialize arrays
        phase = ['G1']
        cyclin_D = np.zeros(steps)
        cyclin_E = np.zeros(steps)
        cyclin_A = np.zeros(steps)
        cyclin_B = np.zeros(steps)
        cdk_activity = np.zeros(steps)

        # Initial conditions
        cyclin_D[0] = 0.1
        current_phase = 'G1'
        phase_time = 0

        for i in range(1, steps):
            # Update cyclins based on current phase
            if current_phase == 'G1':
                cyclin_D[i] = cyclin_D[i-1] + dt * (
                    self.cyclin_params['cyclin_D']['synthesis'] -
                    self.cyclin_params['cyclin_D']['degradation'] * cyclin_D[i-1]
                )
                cyclin_E[i] = cyclin_E[i-1] + dt * (
                    0.3 * cyclin_D[i-1] - 0.1 * cyclin_E[i-1]
                )

                # G1/S checkpoint
                if cyclin_D[i] > self.checkpoints['G1/S']['cyclin_D'] and phase_time > self.durations['G1']:
                    current_phase = 'S'
                    phase_time = 0

            elif current_phase == 'S':
                cyclin_A[i] = cyclin_A[i-1] + dt * (
                    self.cyclin_params['cyclin_A']['synthesis'] -
                    self.cyclin_params['cyclin_A']['degradation'] * cyclin_A[i-1]
                )
                cyclin_E[i] = max(0, cyclin_E[i-1] - dt * 0.2)

                # S/G2 checkpoint
                if cyclin_A[i] > self.checkpoints['S/G2']['cyclin_A'] and phase_time > self.durations['S']:
                    current_phase = 'G2'
                    phase_time = 0

            elif current_phase == 'G2':
                cyclin_B[i] = cyclin_B[i-1] + dt * (
                    self.cyclin_params['cyclin_B']['synthesis'] -
                    self.cyclin_params['cyclin_B']['degradation'] * cyclin_B[i-1]
                )
                cyclin_A[i] = cyclin_A[i-1] * 0.95  # Gradual degradation

                # G2/M checkpoint
                if cyclin_B[i] > self.checkpoints['G2/M']['cyclin_B'] and phase_time > self.durations['G2']:
                    current_phase = 'M'
                    phase_time = 0

            elif current_phase == 'M':
                # Rapid degradation of cyclins during mitosis
                cyclin_B[i] = cyclin_B[i-1] * 0.8
                cyclin_A[i] = cyclin_A[i-1] * 0.7

                # M exit
                if phase_time > self.durations['M']:
                    current_phase = 'G1'
                    phase_time = 0
                    cyclin_B[i] = 0.1
                    cyclin_A[i] = 0.1
                    cyclin_D[i] = 0.1

            # Update CDK activity (combination of cyclins)
            cdk_activity[i] = (cyclin_D[i] * 0.3 + cyclin_E[i] * 0.3 +
                              cyclin_A[i] * 0.2 + cyclin_B[i] * 0.2)

            phase.append(current_phase)
            phase_time += dt

        return {
            'time': time,
            'phase': phase,
            'cyclin_D': cyclin_D,
            'cyclin_E': cyclin_E,
            'cyclin_A': cyclin_A,
            'cyclin_B': cyclin_B,
            'cdk_activity': cdk_activity
        }

    def calculate_proliferation_rate(self) -> float:
        """Calculate cell doubling time"""
        total_cycle_time = sum(self.durations.values())
        return math.log(2) / total_cycle_time  # Growth rate constant


class SignalingPathway:
    """Model intracellular signaling cascades"""

    def __init__(self, pathway_type: str = 'MAPK'):
        self.pathway_type = pathway_type

        # Michaelis-Menten parameters for kinases
        self.kinase_params = {
            'MAPK': {
                'Raf': {'Vmax': 1.0, 'Km': 0.5, 'ki': 0.1},
                'MEK': {'Vmax': 2.0, 'Km': 0.3, 'ki': 0.2},
                'ERK': {'Vmax': 3.0, 'Km': 0.2, 'ki': 0.3}
            },
            'PI3K': {
                'PI3K': {'Vmax': 1.5, 'Km': 0.4, 'ki': 0.15},
                'AKT': {'Vmax': 2.5, 'Km': 0.25, 'ki': 0.2},
                'mTOR': {'Vmax': 1.8, 'Km': 0.35, 'ki': 0.25}
            },
            'JAK_STAT': {
                'JAK': {'Vmax': 1.2, 'Km': 0.45, 'ki': 0.18},
                'STAT': {'Vmax': 2.2, 'Km': 0.28, 'ki': 0.22}
            }
        }

        # Feedback parameters
        self.feedback_strength = 0.5
        self.crosstalk_strength = 0.2

    def michaelis_menten(self, substrate: float, enzyme: float,
                        vmax: float, km: float) -> float:
        """Calculate reaction rate using Michaelis-Menten kinetics"""
        return (vmax * enzyme * substrate) / (km + substrate)

    def hill_equation(self, ligand: float, kd: float, n: float = 2) -> float:
        """Calculate cooperative binding using Hill equation"""
        return ligand**n / (kd**n + ligand**n)

    def simulate_cascade(self, stimulus: float = 1.0, duration: float = 60,
                        dt: float = 0.1) -> Dict:
        """Simulate signaling cascade dynamics

        Args:
            stimulus: Initial stimulus strength
            duration: Simulation time in minutes
            dt: Time step
        Returns:
            Signaling dynamics
        """
        steps = int(duration / dt)
        time = np.linspace(0, duration, steps)

        if self.pathway_type == 'MAPK':
            return self._simulate_mapk(stimulus, steps, time, dt)
        elif self.pathway_type == 'PI3K':
            return self._simulate_pi3k(stimulus, steps, time, dt)
        else:
            return self._simulate_jak_stat(stimulus, steps, time, dt)

    def _simulate_mapk(self, stimulus: float, steps: int, time: np.ndarray,
                      dt: float) -> Dict:
        """Simulate MAPK/ERK pathway"""
        # Initialize concentrations
        raf = np.zeros(steps)
        mek = np.zeros(steps)
        erk = np.zeros(steps)
        raf_p = np.zeros(steps)  # Phosphorylated forms
        mek_pp = np.zeros(steps)
        erk_pp = np.zeros(steps)

        # Initial conditions
        raf[0] = 1.0
        mek[0] = 1.0
        erk[0] = 1.0

        params = self.kinase_params['MAPK']

        for i in range(1, steps):
            # Raf activation by stimulus
            raf_activation = self.michaelis_menten(
                raf[i-1], stimulus * np.exp(-0.05 * time[i]),
                params['Raf']['Vmax'], params['Raf']['Km']
            )
            raf_p[i] = raf_p[i-1] + dt * (raf_activation - params['Raf']['ki'] * raf_p[i-1])
            raf[i] = raf[0] - raf_p[i]

            # MEK activation by Raf-P
            mek_activation = self.michaelis_menten(
                mek[i-1], raf_p[i-1],
                params['MEK']['Vmax'], params['MEK']['Km']
            )
            mek_pp[i] = mek_pp[i-1] + dt * (mek_activation - params['MEK']['ki'] * mek_pp[i-1])
            mek[i] = mek[0] - mek_pp[i]

            # ERK activation by MEK-PP
            erk_activation = self.michaelis_menten(
                erk[i-1], mek_pp[i-1],
                params['ERK']['Vmax'], params['ERK']['Km']
            )

            # Negative feedback from ERK-PP to Raf
            feedback = self.feedback_strength * erk_pp[i-1]

            erk_pp[i] = erk_pp[i-1] + dt * (
                erk_activation - params['ERK']['ki'] * erk_pp[i-1] - feedback
            )
            erk[i] = erk[0] - erk_pp[i]

        return {
            'time': time,
            'raf': raf,
            'raf_p': raf_p,
            'mek': mek,
            'mek_pp': mek_pp,
            'erk': erk,
            'erk_pp': erk_pp,
            'output': erk_pp
        }

    def _simulate_pi3k(self, stimulus: float, steps: int, time: np.ndarray,
                      dt: float) -> Dict:
        """Simulate PI3K/AKT pathway"""
        # Initialize concentrations
        pi3k = np.zeros(steps)
        pip3 = np.zeros(steps)
        akt = np.zeros(steps)
        akt_p = np.zeros(steps)
        mtor = np.zeros(steps)
        mtor_p = np.zeros(steps)

        # Initial conditions
        pi3k[0] = 1.0
        akt[0] = 1.0
        mtor[0] = 1.0

        params = self.kinase_params['PI3K']

        for i in range(1, steps):
            # PI3K activation
            pi3k_activity = self.michaelis_menten(
                1.0, stimulus * np.exp(-0.03 * time[i]),
                params['PI3K']['Vmax'], params['PI3K']['Km']
            )
            pip3[i] = pip3[i-1] + dt * (pi3k_activity - 0.5 * pip3[i-1])  # PTEN activity

            # AKT activation by PIP3
            akt_activation = self.michaelis_menten(
                akt[i-1], pip3[i-1],
                params['AKT']['Vmax'], params['AKT']['Km']
            )
            akt_p[i] = akt_p[i-1] + dt * (akt_activation - params['AKT']['ki'] * akt_p[i-1])
            akt[i] = akt[0] - akt_p[i]

            # mTOR activation by AKT
            mtor_activation = self.michaelis_menten(
                mtor[i-1], akt_p[i-1],
                params['mTOR']['Vmax'], params['mTOR']['Km']
            )

            # Negative feedback from mTOR to PI3K
            feedback = self.feedback_strength * mtor_p[i-1]

            mtor_p[i] = mtor_p[i-1] + dt * (
                mtor_activation - params['mTOR']['ki'] * mtor_p[i-1]
            )
            mtor[i] = mtor[0] - mtor_p[i]

            # Update PI3K with feedback
            pi3k[i] = pi3k[0] / (1 + feedback)

        return {
            'time': time,
            'pi3k': pi3k,
            'pip3': pip3,
            'akt': akt,
            'akt_p': akt_p,
            'mtor': mtor,
            'mtor_p': mtor_p,
            'output': akt_p
        }

    def _simulate_jak_stat(self, stimulus: float, steps: int, time: np.ndarray,
                          dt: float) -> Dict:
        """Simulate JAK/STAT pathway"""
        jak = np.zeros(steps)
        jak_p = np.zeros(steps)
        stat = np.zeros(steps)
        stat_p = np.zeros(steps)
        socs = np.zeros(steps)  # Negative regulator

        # Initial conditions
        jak[0] = 1.0
        stat[0] = 1.0

        params = self.kinase_params['JAK_STAT']

        for i in range(1, steps):
            # JAK activation by cytokine
            jak_activation = self.michaelis_menten(
                jak[i-1], stimulus * np.exp(-0.02 * time[i]),
                params['JAK']['Vmax'], params['JAK']['Km']
            )

            # SOCS inhibition
            inhibition = 1 / (1 + socs[i-1])

            jak_p[i] = jak_p[i-1] + dt * (
                jak_activation * inhibition - params['JAK']['ki'] * jak_p[i-1]
            )
            jak[i] = jak[0] - jak_p[i]

            # STAT activation by JAK
            stat_activation = self.michaelis_menten(
                stat[i-1], jak_p[i-1],
                params['STAT']['Vmax'], params['STAT']['Km']
            )
            stat_p[i] = stat_p[i-1] + dt * (
                stat_activation - params['STAT']['ki'] * stat_p[i-1]
            )
            stat[i] = stat[0] - stat_p[i]

            # SOCS induction by STAT (delayed negative feedback)
            if i > 50:  # Delay for transcription/translation
                socs[i] = socs[i-1] + dt * (0.5 * stat_p[i-50] - 0.1 * socs[i-1])

        return {
            'time': time,
            'jak': jak,
            'jak_p': jak_p,
            'stat': stat,
            'stat_p': stat_p,
            'socs': socs,
            'output': stat_p
        }

    def calculate_ec50(self, dose_response: np.ndarray, doses: np.ndarray) -> float:
        """Calculate EC50 from dose-response curve"""
        # Fit Hill equation
        def hill_fit(x, ec50, n, vmax):
            return vmax * x**n / (ec50**n + x**n)

        try:
            popt, _ = optimize.curve_fit(hill_fit, doses, dose_response,
                                        p0=[np.median(doses), 1, max(dose_response)])
            return popt[0]  # EC50
        except:
            return np.median(doses)


class ApoptosisModel:
    """Model programmed cell death kinetics"""

    def __init__(self):
        # Apoptosis parameters
        self.death_receptors = ['Fas', 'TNFR', 'TRAIL']
        self.bcl2_family = {
            'anti_apoptotic': ['Bcl2', 'BclXL', 'Mcl1'],
            'pro_apoptotic': ['Bax', 'Bak', 'Bad', 'Bid']
        }

        # Rate constants
        self.k_casp8_activation = 0.5
        self.k_casp9_activation = 0.3
        self.k_casp3_activation = 1.0
        self.k_cytc_release = 0.4
        self.k_parp_cleavage = 2.0

    def simulate_apoptosis(self, death_signal: float = 1.0,
                          mitochondrial_stress: float = 0.5,
                          duration: float = 24, dt: float = 0.1) -> Dict:
        """Simulate apoptosis cascade

        Args:
            death_signal: External death receptor activation
            mitochondrial_stress: Internal stress level
            duration: Time in hours
            dt: Time step
        Returns:
            Apoptosis dynamics
        """
        steps = int(duration / dt)
        time = np.linspace(0, duration, steps)

        # Initialize protein levels
        casp8 = np.ones(steps)
        casp8_active = np.zeros(steps)
        casp9 = np.ones(steps)
        casp9_active = np.zeros(steps)
        casp3 = np.ones(steps)
        casp3_active = np.zeros(steps)
        cytc_cytosol = np.zeros(steps)
        bcl2 = np.ones(steps) * 0.5
        bax = np.ones(steps) * 0.3
        parp = np.ones(steps)
        parp_cleaved = np.zeros(steps)

        # Apoptosis commitment (point of no return)
        commitment = np.zeros(steps)

        for i in range(1, steps):
            # Extrinsic pathway (death receptor)
            casp8_activation = self.k_casp8_activation * death_signal * casp8[i-1]
            casp8_active[i] = casp8_active[i-1] + dt * (
                casp8_activation - 0.1 * casp8_active[i-1]
            )
            casp8[i] = casp8[0] - casp8_active[i]

            # Bcl2/Bax balance determines mitochondrial permeability
            bcl2_bax_ratio = bcl2[i-1] / (bax[i-1] + 0.01)

            # Mitochondrial outer membrane permeabilization (MOMP)
            if bcl2_bax_ratio < 1:
                momp = (1 - bcl2_bax_ratio) * mitochondrial_stress
            else:
                momp = 0

            # Cytochrome c release
            cytc_release = self.k_cytc_release * momp + 0.2 * casp8_active[i-1]  # Bid activation
            cytc_cytosol[i] = cytc_cytosol[i-1] + dt * (
                cytc_release - 0.05 * cytc_cytosol[i-1]
            )

            # Intrinsic pathway (caspase-9)
            casp9_activation = self.k_casp9_activation * cytc_cytosol[i-1] * casp9[i-1]
            casp9_active[i] = casp9_active[i-1] + dt * (
                casp9_activation - 0.1 * casp9_active[i-1]
            )
            casp9[i] = casp9[0] - casp9_active[i]

            # Executioner caspase-3 activation (convergence point)
            casp3_activation = self.k_casp3_activation * (
                0.5 * casp8_active[i-1] + 0.5 * casp9_active[i-1]
            ) * casp3[i-1]

            # Positive feedback amplification
            if casp3_active[i-1] > 0.3:
                casp3_activation *= 2

            casp3_active[i] = casp3_active[i-1] + dt * (
                casp3_activation - 0.05 * casp3_active[i-1]
            )
            casp3[i] = casp3[0] - casp3_active[i]

            # PARP cleavage (apoptosis marker)
            parp_cleavage = self.k_parp_cleavage * casp3_active[i-1] * parp[i-1]
            parp_cleaved[i] = parp_cleaved[i-1] + dt * parp_cleavage
            parp[i] = parp[0] - parp_cleaved[i]

            # Update Bcl2 family (caspase-3 can cleave Bcl2)
            bcl2[i] = bcl2[i-1] * (1 - 0.01 * casp3_active[i-1])
            bax[i] = bax[i-1] * (1 + 0.005 * mitochondrial_stress)

            # Commitment to apoptosis (>50% caspase-3 active)
            commitment[i] = 1 if casp3_active[i] > 0.5 else 0

        # Calculate apoptosis metrics
        time_to_commitment = self._find_commitment_time(commitment, time)

        return {
            'time': time,
            'casp8_active': casp8_active,
            'casp9_active': casp9_active,
            'casp3_active': casp3_active,
            'cytc_cytosol': cytc_cytosol,
            'bcl2': bcl2,
            'bax': bax,
            'parp_cleaved': parp_cleaved,
            'commitment': commitment,
            'time_to_commitment': time_to_commitment,
            'cell_viability': 1 - parp_cleaved  # Inverse of PARP cleavage
        }

    def _find_commitment_time(self, commitment: np.ndarray, time: np.ndarray) -> float:
        """Find time when cell commits to apoptosis"""
        commit_idx = np.where(commitment > 0)[0]
        if len(commit_idx) > 0:
            return time[commit_idx[0]]
        return float('inf')

    def calculate_survival_fraction(self, stress_levels: np.ndarray) -> np.ndarray:
        """Calculate cell survival under different stress levels"""
        survival = np.zeros_like(stress_levels)

        for i, stress in enumerate(stress_levels):
            result = self.simulate_apoptosis(
                death_signal=0,
                mitochondrial_stress=stress,
                duration=24
            )
            # Cell survives if it doesn't commit to apoptosis
            survival[i] = 1 if result['time_to_commitment'] == float('inf') else 0

        return survival


class MigrationModel:
    """Model cell migration and invasion"""

    def __init__(self, substrate: str = 'collagen'):
        self.substrate = substrate

        # Migration parameters for different substrates
        self.substrate_params = {
            'collagen': {'speed': 20, 'persistence': 0.8, 'stiffness': 10},
            'fibronectin': {'speed': 25, 'persistence': 0.7, 'stiffness': 8},
            'laminin': {'speed': 15, 'persistence': 0.9, 'stiffness': 5},
            'matrigel': {'speed': 10, 'persistence': 0.6, 'stiffness': 3}
        }

        self.params = self.substrate_params.get(
            substrate,
            self.substrate_params['collagen']
        )

    def simulate_random_walk(self, n_cells: int = 100, duration: float = 24,
                            dt: float = 0.1) -> Dict:
        """Simulate persistent random walk migration

        Args:
            n_cells: Number of cells to simulate
            duration: Time in hours
            dt: Time step
        Returns:
            Migration trajectories
        """
        steps = int(duration / dt)

        # Initialize positions and velocities
        positions = np.zeros((n_cells, steps, 2))
        velocities = np.zeros((n_cells, steps, 2))

        # Initial random directions
        angles = np.random.uniform(0, 2*np.pi, n_cells)
        velocities[:, 0, 0] = self.params['speed'] * np.cos(angles)
        velocities[:, 0, 1] = self.params['speed'] * np.sin(angles)

        # Persistence parameter
        persistence = self.params['persistence']

        for t in range(1, steps):
            # Update positions
            positions[:, t] = positions[:, t-1] + velocities[:, t-1] * dt

            # Update velocities with persistence
            # New direction = persistence * old + (1-persistence) * random
            random_angles = np.random.uniform(0, 2*np.pi, n_cells)
            random_vel = np.column_stack([
                self.params['speed'] * np.cos(random_angles),
                self.params['speed'] * np.sin(random_angles)
            ])

            velocities[:, t] = (persistence * velocities[:, t-1] +
                               (1 - persistence) * random_vel)

            # Add noise
            velocities[:, t] += np.random.normal(0, 2, (n_cells, 2))

        # Calculate metrics
        msd = self._calculate_msd(positions)
        directionality = self._calculate_directionality(positions)

        return {
            'positions': positions,
            'velocities': velocities,
            'msd': msd,
            'directionality': directionality,
            'mean_speed': np.mean(np.linalg.norm(velocities, axis=2))
        }

    def _calculate_msd(self, positions: np.ndarray) -> np.ndarray:
        """Calculate mean squared displacement"""
        n_cells, n_steps, _ = positions.shape
        msd = np.zeros(n_steps)

        for tau in range(1, n_steps):
            displacements = positions[:, tau:] - positions[:, :-tau]
            squared_disp = np.sum(displacements**2, axis=2)
            msd[tau] = np.mean(squared_disp)

        return msd

    def _calculate_directionality(self, positions: np.ndarray) -> float:
        """Calculate directionality ratio"""
        # Euclidean distance / Path length
        n_cells = positions.shape[0]
        directionality = np.zeros(n_cells)

        for i in range(n_cells):
            euclidean = np.linalg.norm(positions[i, -1] - positions[i, 0])
            path_length = np.sum(np.linalg.norm(np.diff(positions[i], axis=0), axis=1))

            if path_length > 0:
                directionality[i] = euclidean / path_length

        return np.mean(directionality)

    def chemotaxis_gradient(self, gradient_strength: float = 1.0,
                           duration: float = 12) -> Dict:
        """Simulate directed migration in chemotactic gradient"""
        n_cells = 100
        steps = int(duration / 0.1)

        positions = np.zeros((n_cells, steps, 2))

        # Initial random positions
        positions[:, 0] = np.random.normal(0, 10, (n_cells, 2))

        for t in range(1, steps):
            # Calculate gradient direction (pointing to origin)
            directions = -positions[:, t-1] / (np.linalg.norm(positions[:, t-1], axis=1, keepdims=True) + 0.1)

            # Biased random walk
            random_component = np.random.normal(0, 1, (n_cells, 2))

            # Movement combines directed and random components
            movement = (gradient_strength * directions * self.params['speed'] +
                       (1 - gradient_strength) * random_component * self.params['speed']) * 0.1

            positions[:, t] = positions[:, t-1] + movement

        # Calculate chemotactic index
        ci = self._calculate_chemotactic_index(positions)

        return {
            'positions': positions,
            'chemotactic_index': ci,
            'final_distribution': positions[:, -1]
        }

    def _calculate_chemotactic_index(self, positions: np.ndarray) -> float:
        """Calculate chemotactic index (migration toward gradient)"""
        n_cells = positions.shape[0]
        ci_values = np.zeros(n_cells)

        for i in range(n_cells):
            displacement = positions[i, -1] - positions[i, 0]
            # Assuming gradient points to origin
            gradient_direction = -positions[i, 0] / (np.linalg.norm(positions[i, 0]) + 0.1)

            # Dot product gives component along gradient
            ci_values[i] = np.dot(displacement, gradient_direction) / np.linalg.norm(displacement + 0.001)

        return np.mean(ci_values)


class ProliferationCurve:
    """Model cell proliferation dynamics"""

    def __init__(self, growth_model: str = 'exponential'):
        self.growth_model = growth_model
        self.models = ['exponential', 'logistic', 'gompertz', 'baranyi']

    def exponential_growth(self, t: np.ndarray, N0: float, r: float) -> np.ndarray:
        """Exponential growth model"""
        return N0 * np.exp(r * t)

    def logistic_growth(self, t: np.ndarray, N0: float, r: float, K: float) -> np.ndarray:
        """Logistic growth with carrying capacity"""
        return K / (1 + ((K - N0) / N0) * np.exp(-r * t))

    def gompertz_growth(self, t: np.ndarray, N0: float, r: float, K: float) -> np.ndarray:
        """Gompertz growth model"""
        return K * np.exp(np.log(N0 / K) * np.exp(-r * t))

    def baranyi_growth(self, t: np.ndarray, N0: float, r: float, K: float, lag: float) -> np.ndarray:
        """Baranyi model with lag phase"""
        A = t + (1/r) * np.log(np.exp(-r * t) + np.exp(-r * lag) - np.exp(-r * (t + lag)))
        return N0 + r * A - np.log(1 + (np.exp(r * A) - 1) / np.exp(K - N0))

    def simulate_growth(self, initial_cells: float = 1000, duration: float = 168,
                       dt: float = 1, **params) -> Dict:
        """Simulate cell growth

        Args:
            initial_cells: Starting cell number
            duration: Time in hours
            dt: Time step
            **params: Model-specific parameters
        Returns:
            Growth dynamics
        """
        time = np.arange(0, duration, dt)

        if self.growth_model == 'exponential':
            r = params.get('growth_rate', 0.03)
            cells = self.exponential_growth(time, initial_cells, r)
            doubling_time = math.log(2) / r

        elif self.growth_model == 'logistic':
            r = params.get('growth_rate', 0.05)
            K = params.get('carrying_capacity', 1e6)
            cells = self.logistic_growth(time, initial_cells, r, K)
            doubling_time = math.log(2) / r

        elif self.growth_model == 'gompertz':
            r = params.get('growth_rate', 0.02)
            K = params.get('carrying_capacity', 1e6)
            cells = self.gompertz_growth(time, initial_cells, r, K)
            doubling_time = None  # Variable in Gompertz

        else:  # Baranyi
            r = params.get('growth_rate', 0.04)
            K = params.get('carrying_capacity', 1e6)
            lag = params.get('lag_time', 6)
            cells = np.exp(self.baranyi_growth(time, np.log(initial_cells), r, np.log(K), lag))
            doubling_time = math.log(2) / r

        # Calculate growth phases
        phases = self._identify_growth_phases(cells, time)

        return {
            'time': time,
            'cells': cells,
            'model': self.growth_model,
            'doubling_time': doubling_time,
            'phases': phases,
            'specific_growth_rate': np.gradient(np.log(cells)) / dt
        }

    def _identify_growth_phases(self, cells: np.ndarray, time: np.ndarray) -> Dict:
        """Identify growth phases (lag, exponential, stationary)"""
        growth_rate = np.gradient(np.log(cells + 1))

        # Threshold for phase identification
        max_rate = np.max(growth_rate)
        threshold = 0.1 * max_rate

        # Find phase boundaries
        exp_start = np.where(growth_rate > threshold)[0]
        exp_start = exp_start[0] if len(exp_start) > 0 else 0

        stat_start = np.where(growth_rate < threshold)[0]
        stat_start = stat_start[-1] if len(stat_start) > 0 and stat_start[-1] > exp_start else len(time) - 1

        return {
            'lag': (0, exp_start),
            'exponential': (exp_start, stat_start),
            'stationary': (stat_start, len(time) - 1)
        }


class MetabolicFlux:
    """Model cellular metabolic flux"""

    def __init__(self):
        # Metabolic pathways
        self.pathways = {
            'glycolysis': {
                'glucose_uptake': 10,  # mmol/h/g
                'atp_yield': 2,
                'nadh_yield': 2,
                'pyruvate_yield': 2
            },
            'tca_cycle': {
                'acetyl_coa_input': 2,
                'atp_yield': 2,
                'nadh_yield': 6,
                'fadh2_yield': 2
            },
            'oxidative_phosphorylation': {
                'nadh_atp_ratio': 2.5,
                'fadh2_atp_ratio': 1.5,
                'oxygen_consumption': 6
            }
        }

        # Warburg effect parameters (cancer metabolism)
        self.warburg_factor = 0.0  # 0 = normal, 1 = full Warburg

    def calculate_atp_production(self, glucose_uptake: float = 10,
                                oxygen_available: bool = True) -> Dict:
        """Calculate ATP production from glucose"""
        results = {}

        # Glycolysis (always occurs)
        glycolysis_atp = self.pathways['glycolysis']['atp_yield'] * glucose_uptake
        pyruvate = self.pathways['glycolysis']['pyruvate_yield'] * glucose_uptake
        nadh_glycolysis = self.pathways['glycolysis']['nadh_yield'] * glucose_uptake

        if oxygen_available and self.warburg_factor < 1:
            # Aerobic metabolism
            aerobic_fraction = 1 - self.warburg_factor

            # TCA cycle
            tca_turns = pyruvate * aerobic_fraction / 2  # 2 pyruvate per glucose
            tca_atp = self.pathways['tca_cycle']['atp_yield'] * tca_turns
            nadh_tca = self.pathways['tca_cycle']['nadh_yield'] * tca_turns
            fadh2 = self.pathways['tca_cycle']['fadh2_yield'] * tca_turns

            # Oxidative phosphorylation
            total_nadh = nadh_glycolysis + nadh_tca
            oxphos_atp = (total_nadh * self.pathways['oxidative_phosphorylation']['nadh_atp_ratio'] +
                         fadh2 * self.pathways['oxidative_phosphorylation']['fadh2_atp_ratio'])

            total_atp = glycolysis_atp + tca_atp + oxphos_atp
            lactate = pyruvate * self.warburg_factor

            results['pathway'] = 'oxidative_phosphorylation'
            results['oxygen_consumption'] = total_nadh * 0.5 + fadh2 * 0.5

        else:
            # Anaerobic metabolism (fermentation)
            total_atp = glycolysis_atp
            lactate = pyruvate  # All pyruvate to lactate

            results['pathway'] = 'fermentation'
            results['oxygen_consumption'] = 0

        results.update({
            'glucose_consumed': glucose_uptake,
            'atp_produced': total_atp,
            'lactate_produced': lactate,
            'efficiency': total_atp / (38 * glucose_uptake)  # Theoretical max is 38 ATP/glucose
        })

        return results

    def simulate_metabolic_shift(self, duration: float = 24, hypoxia_start: float = 12) -> Dict:
        """Simulate metabolic shift under changing conditions"""
        time = np.linspace(0, duration, 100)

        glucose_uptake = np.ones_like(time) * 10
        atp_production = np.zeros_like(time)
        lactate_production = np.zeros_like(time)
        oxygen_consumption = np.zeros_like(time)

        for i, t in enumerate(time):
            # Simulate hypoxia
            oxygen_available = t < hypoxia_start

            # Increase glucose uptake under hypoxia (Pasteur effect)
            if not oxygen_available:
                glucose_uptake[i] *= 2

            metabolism = self.calculate_atp_production(glucose_uptake[i], oxygen_available)

            atp_production[i] = metabolism['atp_produced']
            lactate_production[i] = metabolism['lactate_produced']
            oxygen_consumption[i] = metabolism['oxygen_consumption']

        return {
            'time': time,
            'glucose_uptake': glucose_uptake,
            'atp_production': atp_production,
            'lactate_production': lactate_production,
            'oxygen_consumption': oxygen_consumption,
            'hypoxia_time': hypoxia_start
        }


class CellBiologyLab:
    """Main cell biology laboratory interface"""

    def __init__(self):
        self.cell_cycle = CellCycleModel()
        self.signaling = SignalingPathway()
        self.apoptosis = ApoptosisModel()
        self.migration = MigrationModel()
        self.proliferation = ProliferationCurve()
        self.metabolism = MetabolicFlux()
        self.results = {}

    def run_cell_cycle_analysis(self, cell_type: str = 'fibroblast', duration: float = 48) -> Dict:
        """Analyze cell cycle progression"""
        self.cell_cycle = CellCycleModel(cell_type)
        results = self.cell_cycle.simulate_cycle(duration)
        results['proliferation_rate'] = self.cell_cycle.calculate_proliferation_rate()
        self.results['cell_cycle'] = results
        return results

    def run_signaling_analysis(self, pathway: str = 'MAPK', stimulus: float = 1.0) -> Dict:
        """Analyze signaling pathway dynamics"""
        self.signaling = SignalingPathway(pathway)
        results = self.signaling.simulate_cascade(stimulus)

        # Calculate dose-response
        doses = np.logspace(-2, 1, 10)
        response = []
        for dose in doses:
            sim = self.signaling.simulate_cascade(dose, duration=30)
            response.append(np.max(sim['output']))

        results['doses'] = doses
        results['dose_response'] = response
        results['ec50'] = self.signaling.calculate_ec50(np.array(response), doses)

        self.results['signaling'] = results
        return results

    def run_apoptosis_analysis(self, death_signal: float = 0.5, stress: float = 0.7) -> Dict:
        """Analyze apoptosis induction"""
        results = self.apoptosis.simulate_apoptosis(death_signal, stress)

        # Calculate survival curve
        stress_levels = np.linspace(0, 2, 20)
        survival = self.apoptosis.calculate_survival_fraction(stress_levels)

        results['stress_levels'] = stress_levels
        results['survival_fraction'] = survival

        self.results['apoptosis'] = results
        return results

    def run_migration_analysis(self, substrate: str = 'collagen', n_cells: int = 50) -> Dict:
        """Analyze cell migration patterns"""
        self.migration = MigrationModel(substrate)

        # Random migration
        random_walk = self.migration.simulate_random_walk(n_cells)

        # Chemotaxis
        chemotaxis = self.migration.chemotaxis_gradient()

        results = {
            'random_walk': random_walk,
            'chemotaxis': chemotaxis,
            'substrate': substrate
        }

        self.results['migration'] = results
        return results

    def run_proliferation_analysis(self, model: str = 'logistic',
                                 initial_cells: float = 1000) -> Dict:
        """Analyze cell proliferation"""
        self.proliferation = ProliferationCurve(model)

        results = self.proliferation.simulate_growth(
            initial_cells=initial_cells,
            growth_rate=0.04,
            carrying_capacity=1e6
        )

        self.results['proliferation'] = results
        return results

    def run_metabolism_analysis(self, warburg_factor: float = 0.0) -> Dict:
        """Analyze metabolic flux"""
        self.metabolism.warburg_factor = warburg_factor

        # Normal vs cancer metabolism
        normal = self.metabolism.calculate_atp_production()
        cancer = self.metabolism.calculate_atp_production()
        cancer['warburg_factor'] = 0.8

        # Metabolic shift simulation
        shift = self.metabolism.simulate_metabolic_shift()

        results = {
            'normal_metabolism': normal,
            'cancer_metabolism': cancer,
            'metabolic_shift': shift
        }

        self.results['metabolism'] = results
        return results

    def integrate_cellular_response(self, stimulus_type: str = 'growth_factor',
                                  stimulus_strength: float = 1.0) -> Dict:
        """Integrate multiple cellular responses to stimulus"""
        integrated = {}

        if stimulus_type == 'growth_factor':
            # Growth factor activates proliferation and migration
            integrated['signaling'] = self.run_signaling_analysis('MAPK', stimulus_strength)
            integrated['cell_cycle'] = self.run_cell_cycle_analysis('fibroblast')
            integrated['migration'] = self.run_migration_analysis('fibronectin')
            integrated['proliferation'] = self.run_proliferation_analysis('exponential')

        elif stimulus_type == 'stress':
            # Stress can induce apoptosis or survival signaling
            integrated['signaling'] = self.run_signaling_analysis('PI3K', 0.5)
            integrated['apoptosis'] = self.run_apoptosis_analysis(0, stimulus_strength)
            integrated['metabolism'] = self.run_metabolism_analysis(0.5)

        elif stimulus_type == 'cytokine':
            # Cytokines activate JAK/STAT and can affect multiple processes
            integrated['signaling'] = self.run_signaling_analysis('JAK_STAT', stimulus_strength)
            integrated['migration'] = self.run_migration_analysis('collagen')
            integrated['proliferation'] = self.run_proliferation_analysis('logistic')

        self.results['integrated_response'] = integrated
        return integrated


def run_demo():
    """Demonstrate cell biology lab capabilities"""
    print("CELL BIOLOGY LAB - Production Demo")
    print("=" * 60)

    lab = CellBiologyLab()

    # 1. Cell cycle analysis
    print("\n1. CELL CYCLE ANALYSIS")
    print("-" * 40)
    cycle = lab.run_cell_cycle_analysis('cancer', duration=48)
    print(f"Cell type: cancer")
    print(f"Proliferation rate: {cycle['proliferation_rate']:.4f} /hour")
    print(f"Doubling time: {math.log(2)/cycle['proliferation_rate']:.1f} hours")
    max_cdk = np.max(cycle['cdk_activity'])
    print(f"Peak CDK activity: {max_cdk:.2f}")

    # 2. Signaling pathway analysis
    print("\n2. SIGNALING PATHWAY ANALYSIS")
    print("-" * 40)
    signaling = lab.run_signaling_analysis('MAPK', stimulus=2.0)
    print(f"Pathway: MAPK")
    print(f"Peak ERK activation: {np.max(signaling['erk_pp']):.2f}")
    print(f"EC50: {signaling['ec50']:.3f}")
    print(f"Response time to peak: {signaling['time'][np.argmax(signaling['erk_pp'])]:.1f} min")

    # 3. Apoptosis analysis
    print("\n3. APOPTOSIS ANALYSIS")
    print("-" * 40)
    apoptosis = lab.run_apoptosis_analysis(death_signal=0.5, stress=0.8)
    print(f"Death signal: 0.5, Stress: 0.8")
    print(f"Time to commitment: {apoptosis['time_to_commitment']:.1f} hours")
    print(f"Final caspase-3 activation: {apoptosis['casp3_active'][-1]:.2f}")
    print(f"Cell viability: {apoptosis['cell_viability'][-1]:.2%}")

    # 4. Migration analysis
    print("\n4. CELL MIGRATION ANALYSIS")
    print("-" * 40)
    migration = lab.run_migration_analysis('fibronectin', n_cells=30)
    print(f"Substrate: fibronectin")
    print(f"Mean speed: {migration['random_walk']['mean_speed']:.1f} μm/hour")
    print(f"Directionality: {migration['random_walk']['directionality']:.3f}")
    print(f"Chemotactic index: {migration['chemotaxis']['chemotactic_index']:.3f}")

    # 5. Proliferation analysis
    print("\n5. PROLIFERATION ANALYSIS")
    print("-" * 40)
    proliferation = lab.run_proliferation_analysis('logistic', initial_cells=1000)
    print(f"Growth model: logistic")
    print(f"Initial cells: 1000")
    print(f"Final cells: {proliferation['cells'][-1]:.0f}")
    print(f"Doubling time: {proliferation['doubling_time']:.1f} hours")
    lag, exp, stat = proliferation['phases'].values()
    print(f"Growth phases - Lag: 0-{lag[1]} h, Exponential: {exp[0]}-{exp[1]} h")

    # 6. Metabolism analysis
    print("\n6. METABOLIC FLUX ANALYSIS")
    print("-" * 40)
    metabolism = lab.run_metabolism_analysis(warburg_factor=0.3)
    normal = metabolism['normal_metabolism']
    print(f"Normal metabolism:")
    print(f"  ATP production: {normal['atp_produced']:.1f} mol/h")
    print(f"  Efficiency: {normal['efficiency']:.2%}")
    print(f"  O2 consumption: {normal['oxygen_consumption']:.1f} mol/h")

    # 7. Integrated response
    print("\n7. INTEGRATED CELLULAR RESPONSE")
    print("-" * 40)
    integrated = lab.integrate_cellular_response('growth_factor', stimulus_strength=1.5)
    print(f"Stimulus: Growth factor (strength=1.5)")
    print(f"Activated pathways: {list(integrated.keys())}")
    print(f"MAPK activation: {np.max(integrated['signaling']['erk_pp']):.2f}")
    print(f"Proliferation rate: {integrated['cell_cycle']['proliferation_rate']:.4f} /hour")
    print(f"Migration speed: {integrated['migration']['random_walk']['mean_speed']:.1f} μm/hour")

    print("\n" + "=" * 60)
    print("Demo complete! Lab ready for production use.")


if __name__ == '__main__':
    run_demo()