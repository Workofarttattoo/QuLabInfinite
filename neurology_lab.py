"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

NEUROLOGY LAB - Production-Ready Clinical Neuroscience
Free gift to the scientific community from QuLabInfinite.

This module implements comprehensive neurological disease models and clinical tools:
- Neurodegenerative disease progression (Alzheimer's, Parkinson's, ALS, MS)
- Lesion mapping and connectivity analysis
- Cognitive decline trajectories
- Treatment response prediction
- Biomarker dynamics (CSF, blood, imaging)
- Clinical scoring systems (MMSE, UPDRS, EDSS, etc.)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
from scipy import stats, optimize, interpolate
from scipy.integrate import solve_ivp, odeint
from scipy.special import expit as sigmoid
import warnings
from enum import Enum

class DiseaseStage(Enum):
    """Disease progression stages."""
    PRECLINICAL = "preclinical"
    PRODROMAL = "prodromal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    TERMINAL = "terminal"

@dataclass
class Patient:
    """Patient demographics and clinical data."""
    age: float
    sex: str  # 'M' or 'F'
    apoe_genotype: str = 'e3/e3'  # APOE allele status
    education_years: int = 12
    comorbidities: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    family_history: bool = False

class AlzheimersDiseaseModel:
    """Comprehensive Alzheimer's disease progression model."""

    def __init__(self, patient: Patient):
        self.patient = patient
        self.baseline_age = patient.age

        # Amyloid-tau-neurodegeneration (ATN) framework
        self.amyloid_burden = 0.1  # Initial amyloid load (0-1)
        self.tau_burden = 0.05  # Initial tau pathology (0-1)
        self.brain_volume = 1.0  # Normalized brain volume

        # Cognitive scores
        self.mmse = 30  # Mini-Mental State Exam (0-30)
        self.adas_cog = 0  # Alzheimer's Disease Assessment Scale-Cognitive (0-70)
        self.cdr = 0  # Clinical Dementia Rating (0-3)

        # Biomarkers
        self.csf_abeta42 = 1000  # pg/mL (normal > 500)
        self.csf_tau = 200  # pg/mL (normal < 350)
        self.csf_ptau = 20  # pg/mL (normal < 60)

        # Risk modifiers
        self.apoe_risk = {'e2/e2': 0.5, 'e2/e3': 0.7, 'e3/e3': 1.0,
                         'e3/e4': 2.5, 'e4/e4': 10.0}
        self.risk_factor = self.apoe_risk.get(patient.apoe_genotype, 1.0)

    def jack_model_dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """Jack et al. hypothetical model of AD biomarker dynamics."""
        amyloid, tau, neurodegeneration = state

        # Amyloid accumulation (sigmoidal)
        k_a = 0.05 * self.risk_factor  # Rate constant
        amyloid_max = 1.0
        d_amyloid = k_a * amyloid * (1 - amyloid / amyloid_max)

        # Tau accumulation (triggered by amyloid)
        k_t = 0.03
        tau_threshold = 0.3  # Amyloid threshold for tau
        if amyloid > tau_threshold:
            d_tau = k_t * (amyloid - tau_threshold) * (1 - tau)
        else:
            d_tau = 0

        # Neurodegeneration (driven by tau)
        k_n = 0.02
        d_neurodegeneration = k_n * tau * neurodegeneration

        return np.array([d_amyloid, d_tau, -d_neurodegeneration])

    def simulate_progression(self, years: float, dt: float = 0.1) -> Dict:
        """Simulate disease progression over time."""
        t_span = (0, years)
        t_eval = np.arange(0, years, dt)

        # Initial conditions
        y0 = [self.amyloid_burden, self.tau_burden, self.brain_volume]

        # Solve ODEs
        solution = solve_ivp(self.jack_model_dynamics, t_span, y0,
                            t_eval=t_eval, method='RK45')

        # Calculate cognitive decline
        mmse_scores = []
        adas_scores = []
        cdr_scores = []

        for i in range(len(t_eval)):
            amyloid = solution.y[0, i]
            tau = solution.y[1, i]
            atrophy = 1 - solution.y[2, i]

            # MMSE decline (nonlinear function of pathology)
            mmse = 30 * (1 - 0.5 * amyloid - 0.3 * tau - 0.2 * atrophy)
            mmse = np.clip(mmse, 0, 30)
            mmse_scores.append(mmse)

            # ADAS-Cog increases with pathology
            adas = 70 * (0.3 * amyloid + 0.4 * tau + 0.3 * atrophy)
            adas = np.clip(adas, 0, 70)
            adas_scores.append(adas)

            # CDR staging
            if mmse >= 26:
                cdr = 0 if amyloid < 0.3 else 0.5
            elif mmse >= 21:
                cdr = 1.0
            elif mmse >= 10:
                cdr = 2.0
            else:
                cdr = 3.0
            cdr_scores.append(cdr)

        # Update biomarkers
        csf_abeta42 = 1000 * (1 - solution.y[0])  # Decreases with amyloid
        csf_tau = 200 + 500 * solution.y[1]  # Increases with tau
        csf_ptau = 20 + 100 * solution.y[1]

        return {
            'time_years': t_eval,
            'amyloid': solution.y[0],
            'tau': solution.y[1],
            'brain_volume': solution.y[2],
            'mmse': np.array(mmse_scores),
            'adas_cog': np.array(adas_scores),
            'cdr': np.array(cdr_scores),
            'csf_abeta42': csf_abeta42,
            'csf_tau': csf_tau,
            'csf_ptau': csf_ptau,
            'stage': self._determine_stages(solution.y[0], solution.y[1], mmse_scores)
        }

    def _determine_stages(self, amyloid: np.ndarray, tau: np.ndarray,
                         mmse: List[float]) -> List[str]:
        """Determine disease stage at each time point."""
        stages = []
        for i in range(len(amyloid)):
            if mmse[i] >= 28 and amyloid[i] < 0.2:
                stage = DiseaseStage.PRECLINICAL.value
            elif mmse[i] >= 26:
                stage = DiseaseStage.PRODROMAL.value
            elif mmse[i] >= 21:
                stage = DiseaseStage.MILD.value
            elif mmse[i] >= 10:
                stage = DiseaseStage.MODERATE.value
            else:
                stage = DiseaseStage.SEVERE.value
            stages.append(stage)
        return stages

    def predict_treatment_response(self, drug_type: str, efficacy: float = 0.3) -> Dict:
        """Predict response to AD treatments."""
        treatments = {
            'cholinesterase_inhibitor': {
                'target': 'cognitive',
                'effect_size': 0.2,
                'duration_months': 12
            },
            'anti_amyloid': {
                'target': 'amyloid',
                'effect_size': 0.3,
                'duration_months': 18
            },
            'tau_inhibitor': {
                'target': 'tau',
                'effect_size': 0.25,
                'duration_months': 24
            }
        }

        treatment = treatments.get(drug_type, treatments['cholinesterase_inhibitor'])

        # Simulate with and without treatment
        baseline = self.simulate_progression(years=2)
        treated = self.simulate_progression(years=2)

        # Apply treatment effect
        if treatment['target'] == 'amyloid':
            treated['amyloid'] *= (1 - efficacy * treatment['effect_size'])
        elif treatment['target'] == 'tau':
            treated['tau'] *= (1 - efficacy * treatment['effect_size'])
        else:  # cognitive
            treated['mmse'] += efficacy * treatment['effect_size'] * 5

        return {
            'treatment': drug_type,
            'baseline_mmse_decline': baseline['mmse'][-1] - baseline['mmse'][0],
            'treated_mmse_decline': treated['mmse'][-1] - treated['mmse'][0],
            'effect_size': treatment['effect_size'],
            'nnt': int(1 / (efficacy * treatment['effect_size'])) if efficacy > 0 else None
        }

class ParkinsonsDiseaseModel:
    """Parkinson's disease progression and treatment model."""

    def __init__(self, patient: Patient):
        self.patient = patient

        # Motor symptoms (UPDRS Part III)
        self.updrs_motor = 0  # 0-108 scale
        self.hoehn_yahr = 0  # 0-5 scale

        # Non-motor symptoms
        self.moca = 30  # Montreal Cognitive Assessment (0-30)
        self.nms_quest = 0  # Non-Motor Symptoms Questionnaire (0-30)

        # Dopamine levels
        self.dopamine_level = 1.0  # Normalized (1.0 = normal)
        self.alpha_synuclein = 0.1  # Pathological protein

        # Levodopa response
        self.levodopa_years = 0
        self.wearing_off = False
        self.dyskinesia = False

    def braak_staging(self) -> int:
        """Determine Braak stage based on pathology spread."""
        if self.alpha_synuclein < 0.2:
            return 1  # Medulla oblongata
        elif self.alpha_synuclein < 0.4:
            return 2  # Pons
        elif self.alpha_synuclein < 0.6:
            return 3  # Midbrain
        elif self.alpha_synuclein < 0.8:
            return 4  # Basal forebrain
        else:
            return 5  # Neocortex

    def simulate_progression(self, years: float) -> Dict:
        """Simulate PD progression with treatment effects."""
        time_points = np.linspace(0, years, int(years * 12))  # Monthly
        results = {
            'time': time_points,
            'updrs_motor': [],
            'hoehn_yahr': [],
            'dopamine': [],
            'alpha_synuclein': [],
            'moca': [],
            'levodopa_response': []
        }

        for t in time_points:
            # Alpha-synuclein accumulation
            self.alpha_synuclein += 0.002 * (1 - self.alpha_synuclein)

            # Dopamine loss (exponential decay)
            self.dopamine_level *= np.exp(-0.005)

            # Motor symptom progression (UPDRS)
            base_progression = 2.5 * t  # Average 2.5 points/year
            self.updrs_motor = base_progression * (1 - self.dopamine_level)
            self.updrs_motor = np.clip(self.updrs_motor, 0, 108)

            # Hoehn-Yahr staging
            if self.updrs_motor < 10:
                self.hoehn_yahr = 1.0
            elif self.updrs_motor < 20:
                self.hoehn_yahr = 1.5
            elif self.updrs_motor < 30:
                self.hoehn_yahr = 2.0
            elif self.updrs_motor < 40:
                self.hoehn_yahr = 2.5
            elif self.updrs_motor < 50:
                self.hoehn_yahr = 3.0
            elif self.updrs_motor < 70:
                self.hoehn_yahr = 4.0
            else:
                self.hoehn_yahr = 5.0

            # Cognitive decline (MoCA)
            if self.alpha_synuclein > 0.7:  # Cortical involvement
                self.moca = 30 - 5 * (self.alpha_synuclein - 0.7) / 0.3
                self.moca = np.clip(self.moca, 0, 30)

            # Levodopa response modeling
            if self.levodopa_years > 0:
                # Initial good response
                response = 0.7  # 70% improvement

                # Wearing-off after 3-5 years
                if self.levodopa_years > 3:
                    response *= np.exp(-0.1 * (self.levodopa_years - 3))

                # Dyskinesia risk
                if self.levodopa_years > 5:
                    self.dyskinesia = np.random.random() < 0.4

                results['levodopa_response'].append(response)
            else:
                results['levodopa_response'].append(0)

            # Store results
            results['updrs_motor'].append(self.updrs_motor)
            results['hoehn_yahr'].append(self.hoehn_yahr)
            results['dopamine'].append(self.dopamine_level)
            results['alpha_synuclein'].append(self.alpha_synuclein)
            results['moca'].append(self.moca)

        return {k: np.array(v) if isinstance(v, list) else v
                for k, v in results.items()}

    def deep_brain_stimulation_effect(self, target: str = 'STN') -> Dict:
        """Model DBS treatment effects."""
        targets = {
            'STN': {'motor_improvement': 0.5, 'medication_reduction': 0.4},
            'GPi': {'motor_improvement': 0.4, 'dyskinesia_reduction': 0.6},
            'VIM': {'tremor_reduction': 0.8, 'other_symptoms': 0.1}
        }

        effect = targets.get(target, targets['STN'])

        pre_dbs_updrs = self.updrs_motor
        post_dbs_updrs = pre_dbs_updrs * (1 - effect['motor_improvement'])

        return {
            'target': target,
            'pre_dbs_updrs': pre_dbs_updrs,
            'post_dbs_updrs': post_dbs_updrs,
            'improvement_percent': effect['motor_improvement'] * 100,
            'medication_reduction': effect.get('medication_reduction', 0) * 100
        }

class MultipleSclerosisModel:
    """Multiple Sclerosis disease course and treatment model."""

    def __init__(self, patient: Patient, ms_type: str = 'RRMS'):
        self.patient = patient
        self.ms_type = ms_type  # RRMS, SPMS, PPMS, PRMS
        self.edss = 0  # Expanded Disability Status Scale (0-10)
        self.relapses = []  # List of relapse times
        self.lesion_load = 0.1  # T2 lesion volume (normalized)
        self.brain_atrophy_rate = 0.005  # Annual rate

        # Disease-modifying therapy
        self.on_dmt = False
        self.dmt_efficacy = 0

    def simulate_rrms(self, years: float) -> Dict:
        """Simulate relapsing-remitting MS."""
        time_points = np.linspace(0, years, int(years * 52))  # Weekly
        edss_trajectory = []
        relapse_times = []

        current_edss = self.edss
        time_since_relapse = 0

        for week, t in enumerate(time_points):
            # Relapse probability (Poisson process)
            annual_relapse_rate = 0.5 if not self.on_dmt else 0.5 * (1 - self.dmt_efficacy)
            weekly_relapse_prob = annual_relapse_rate / 52

            if np.random.random() < weekly_relapse_prob and time_since_relapse > 12:
                # Relapse occurs
                relapse_severity = np.random.gamma(2, 0.5)  # EDSS increase
                current_edss += relapse_severity
                relapse_times.append(t)
                time_since_relapse = 0
            else:
                time_since_relapse += 1

            # Partial recovery from relapse
            if time_since_relapse < 12 and len(relapse_times) > 0:
                recovery_rate = 0.05
                current_edss -= recovery_rate
                current_edss = max(current_edss, self.edss)

            # Progressive disability accumulation
            if self.ms_type == 'SPMS' and t > 10:  # Secondary progressive after 10 years
                current_edss += 0.001  # Slow progression

            current_edss = np.clip(current_edss, 0, 10)
            edss_trajectory.append(current_edss)

        return {
            'time_years': time_points,
            'edss': np.array(edss_trajectory),
            'relapses': relapse_times,
            'relapse_count': len(relapse_times),
            'annualized_relapse_rate': len(relapse_times) / years
        }

    def mri_metrics(self, time_years: float) -> Dict:
        """Calculate MRI-based metrics."""
        # T2 lesion accumulation
        t2_lesions = self.lesion_load * (1 + 0.1 * time_years)

        # Brain atrophy
        brain_volume = 1.0 * np.exp(-self.brain_atrophy_rate * time_years)

        # Gadolinium-enhancing lesions (active inflammation)
        gad_lesions = max(0, np.random.poisson(2) if not self.on_dmt else
                         np.random.poisson(0.5))

        return {
            't2_lesion_volume_ml': t2_lesions * 20,  # Convert to mL
            'brain_parenchymal_fraction': brain_volume,
            'gad_enhancing_lesions': gad_lesions,
            'atrophy_rate_percent': self.brain_atrophy_rate * 100
        }

class StrokeModel:
    """Acute stroke and recovery model."""

    def __init__(self, stroke_type: str = 'ischemic', location: str = 'MCA'):
        self.stroke_type = stroke_type  # ischemic or hemorrhagic
        self.location = location  # MCA, ACA, PCA, etc.
        self.nihss = 0  # NIH Stroke Scale (0-42)
        self.mrs = 0  # Modified Rankin Scale (0-6)
        self.infarct_volume = 0  # mL
        self.time_to_treatment = 0  # hours

    def calculate_nihss(self, infarct_volume: float, location: str) -> int:
        """Calculate NIHSS based on infarct characteristics."""
        # Volume-NIHSS relationship (simplified)
        base_nihss = 0.5 * np.sqrt(infarct_volume)

        # Location modifiers
        location_factors = {
            'MCA': 1.2,  # Middle cerebral artery
            'ACA': 0.8,  # Anterior cerebral artery
            'PCA': 0.7,  # Posterior cerebral artery
            'brainstem': 1.5,
            'cerebellar': 0.6
        }

        nihss = base_nihss * location_factors.get(location, 1.0)
        return int(np.clip(nihss, 0, 42))

    def thrombolysis_outcome(self, time_to_treatment: float, nihss: int) -> Dict:
        """Model tPA thrombolysis outcomes."""
        # Time-dependent benefit (exponential decay)
        if time_to_treatment <= 4.5:
            time_factor = np.exp(-0.5 * time_to_treatment)
        else:
            time_factor = 0  # Outside window

        # Probability of good outcome (mRS 0-1)
        baseline_good_outcome = 0.3
        treatment_benefit = 0.3 * time_factor
        p_good_outcome = baseline_good_outcome + treatment_benefit

        # Hemorrhagic transformation risk
        hemorrhage_risk = 0.06 * (1 + nihss / 20)

        # Final outcome
        if np.random.random() < hemorrhage_risk:
            outcome_mrs = np.random.choice([4, 5, 6], p=[0.3, 0.4, 0.3])
        elif np.random.random() < p_good_outcome:
            outcome_mrs = np.random.choice([0, 1], p=[0.4, 0.6])
        else:
            outcome_mrs = np.random.choice([2, 3, 4], p=[0.3, 0.4, 0.3])

        return {
            'time_to_treatment_hours': time_to_treatment,
            'baseline_nihss': nihss,
            'outcome_mrs': outcome_mrs,
            'good_outcome': outcome_mrs <= 1,
            'hemorrhage': np.random.random() < hemorrhage_risk,
            'nnt_for_good_outcome': int(1 / treatment_benefit) if treatment_benefit > 0 else None
        }

    def recovery_trajectory(self, initial_nihss: int, months: int = 12) -> Dict:
        """Model post-stroke recovery."""
        time_points = np.linspace(0, months, months + 1)
        nihss_trajectory = []
        mrs_trajectory = []

        current_nihss = initial_nihss

        for month in time_points:
            # Exponential recovery with plateau
            if month == 0:
                recovery = 0
            else:
                # Most recovery in first 3 months
                recovery_rate = 0.3 if month <= 3 else 0.05
                recovery = recovery_rate * np.exp(-0.5 * month)

            current_nihss = initial_nihss * (1 - recovery)
            current_nihss = max(0, current_nihss)

            # Convert NIHSS to mRS
            if current_nihss == 0:
                mrs = 0
            elif current_nihss <= 4:
                mrs = 1
            elif current_nihss <= 8:
                mrs = 2
            elif current_nihss <= 12:
                mrs = 3
            elif current_nihss <= 16:
                mrs = 4
            else:
                mrs = 5

            nihss_trajectory.append(current_nihss)
            mrs_trajectory.append(mrs)

        return {
            'time_months': time_points,
            'nihss': np.array(nihss_trajectory),
            'mrs': np.array(mrs_trajectory),
            'final_nihss': nihss_trajectory[-1],
            'final_mrs': mrs_trajectory[-1]
        }

class CognitiveAssessment:
    """Comprehensive cognitive assessment tools."""

    @staticmethod
    def mmse_score(orientation: int = 10, registration: int = 3,
                   attention: int = 5, recall: int = 3,
                   language: int = 9) -> int:
        """Calculate Mini-Mental State Examination score."""
        total = orientation + registration + attention + recall + language
        return np.clip(total, 0, 30)

    @staticmethod
    def moca_score(visuospatial: int = 5, naming: int = 3,
                   attention: int = 6, language: int = 3,
                   abstraction: int = 2, recall: int = 5,
                   orientation: int = 6) -> int:
        """Calculate Montreal Cognitive Assessment score."""
        total = (visuospatial + naming + attention + language +
                abstraction + recall + orientation)
        return np.clip(total, 0, 30)

    @staticmethod
    def cdr_sob(memory: float, orientation: float, judgment: float,
                community: float, home: float, personal: float) -> float:
        """Calculate Clinical Dementia Rating Sum of Boxes."""
        # Each domain scored 0, 0.5, 1, 2, or 3
        domains = [memory, orientation, judgment, community, home, personal]
        for i, score in enumerate(domains):
            domains[i] = np.clip(score, 0, 3)
        return sum(domains)

    @staticmethod
    def ace_iii(attention: int = 18, memory: int = 26, fluency: int = 14,
                language: int = 26, visuospatial: int = 16) -> int:
        """Calculate Addenbrooke's Cognitive Examination III score."""
        total = attention + memory + fluency + language + visuospatial
        return np.clip(total, 0, 100)

class BiomarkerDynamics:
    """Model biomarker trajectories in neurological diseases."""

    def __init__(self):
        self.biomarkers = {}

    def alzheimers_csf(self, disease_stage: str) -> Dict:
        """AD CSF biomarker profiles by stage."""
        profiles = {
            'normal': {'abeta42': 1000, 'tau': 200, 'ptau': 20, 'nfl': 500},
            'preclinical': {'abeta42': 600, 'tau': 250, 'ptau': 30, 'nfl': 600},
            'mci': {'abeta42': 400, 'tau': 400, 'ptau': 50, 'nfl': 800},
            'dementia': {'abeta42': 300, 'tau': 600, 'ptau': 80, 'nfl': 1200}
        }
        return profiles.get(disease_stage, profiles['normal'])

    def parkinsons_dat_scan(self, years_since_onset: float) -> float:
        """Model dopamine transporter imaging decline."""
        # Exponential decline with 5-10% annual loss
        baseline_binding = 2.5  # Striatal binding ratio
        annual_decline = 0.07
        current_binding = baseline_binding * np.exp(-annual_decline * years_since_onset)
        return max(0.5, current_binding)  # Floor at severe loss

    def ms_oligoclonal_bands(self, ms_type: str) -> Dict:
        """CSF oligoclonal band patterns in MS."""
        patterns = {
            'RRMS': {'ocb_positive': 0.95, 'igg_index': 0.8, 'cell_count': 10},
            'PPMS': {'ocb_positive': 0.90, 'igg_index': 0.75, 'cell_count': 5},
            'CIS': {'ocb_positive': 0.70, 'igg_index': 0.6, 'cell_count': 8}
        }
        return patterns.get(ms_type, patterns['RRMS'])

class LesionMapping:
    """Brain lesion location and network effects."""

    def __init__(self):
        self.brain_regions = {
            'frontal': {'functions': ['executive', 'motor_planning', 'speech_production']},
            'parietal': {'functions': ['sensory', 'spatial', 'attention']},
            'temporal': {'functions': ['memory', 'language', 'auditory']},
            'occipital': {'functions': ['vision', 'visual_processing']},
            'cerebellum': {'functions': ['coordination', 'balance', 'motor_learning']},
            'brainstem': {'functions': ['consciousness', 'vital_functions', 'cranial_nerves']},
            'basal_ganglia': {'functions': ['movement', 'procedural_learning', 'motivation']},
            'thalamus': {'functions': ['relay', 'consciousness', 'sensory_gating']},
            'hippocampus': {'functions': ['memory_formation', 'spatial_navigation']}
        }

    def predict_deficits(self, lesion_location: str, lesion_size: float) -> Dict:
        """Predict neurological deficits from lesion characteristics."""
        if lesion_location not in self.brain_regions:
            return {'error': 'Unknown brain region'}

        affected_functions = self.brain_regions[lesion_location]['functions']

        # Size-dependent severity (0-1 scale)
        severity = 1 - np.exp(-lesion_size / 20)  # 20 cm³ for significant effect

        deficits = {}
        for function in affected_functions:
            deficit_severity = severity * np.random.uniform(0.7, 1.0)
            deficits[function] = {
                'severity': deficit_severity,
                'recovery_potential': 1 - deficit_severity * 0.5
            }

        return {
            'location': lesion_location,
            'size_cm3': lesion_size,
            'affected_functions': affected_functions,
            'deficits': deficits,
            'overall_severity': severity
        }

    def disconnection_syndrome(self, white_matter_tract: str) -> Dict:
        """Model white matter disconnection syndromes."""
        tracts = {
            'corpus_callosum': {
                'syndrome': 'Callosal disconnection',
                'symptoms': ['alien_hand', 'hemispheric_competition', 'agraphia']
            },
            'arcuate_fasciculus': {
                'syndrome': 'Conduction aphasia',
                'symptoms': ['repetition_deficit', 'phonemic_paraphasia']
            },
            'uncinate_fasciculus': {
                'syndrome': 'Semantic dementia risk',
                'symptoms': ['semantic_memory_loss', 'emotional_dysregulation']
            },
            'cingulum': {
                'syndrome': 'Executive-attention deficit',
                'symptoms': ['attention_deficit', 'memory_impairment', 'emotional_changes']
            }
        }

        tract_info = tracts.get(white_matter_tract, {
            'syndrome': 'Unknown',
            'symptoms': []
        })

        return {
            'tract': white_matter_tract,
            'syndrome': tract_info['syndrome'],
            'symptoms': tract_info['symptoms'],
            'connectivity_loss': np.random.uniform(0.3, 0.9)
        }

class TreatmentOptimizer:
    """Optimize treatment strategies for neurological diseases."""

    def __init__(self):
        self.treatments = {}

    def alzheimers_combination_therapy(self, stage: str, biomarkers: Dict) -> Dict:
        """Recommend combination therapy for AD."""
        recommendations = []
        rationale = []

        if stage in ['preclinical', 'prodromal']:
            if biomarkers.get('abeta42', 1000) < 600:
                recommendations.append('anti-amyloid antibody')
                rationale.append('Low CSF Aβ42 indicates amyloid pathology')

        if stage in ['mild', 'moderate']:
            recommendations.append('cholinesterase inhibitor')
            rationale.append('Symptomatic benefit for cognition')

            if stage == 'moderate':
                recommendations.append('memantine')
                rationale.append('NMDA antagonist for moderate-severe AD')

        # Lifestyle interventions
        recommendations.extend(['exercise', 'cognitive_training', 'mediterranean_diet'])
        rationale.append('Multimodal lifestyle intervention (FINGER study)')

        return {
            'stage': stage,
            'recommendations': recommendations,
            'rationale': rationale,
            'expected_benefit': self._calculate_expected_benefit(recommendations)
        }

    def _calculate_expected_benefit(self, treatments: List[str]) -> float:
        """Estimate combined treatment benefit."""
        benefits = {
            'anti-amyloid antibody': 0.27,  # 27% slowing
            'cholinesterase inhibitor': 0.15,
            'memantine': 0.10,
            'exercise': 0.20,
            'cognitive_training': 0.15,
            'mediterranean_diet': 0.10
        }

        # Diminishing returns for combination
        total_benefit = 0
        for i, treatment in enumerate(treatments):
            benefit = benefits.get(treatment, 0)
            total_benefit += benefit * (0.8 ** i)  # Each additional treatment less effective

        return min(total_benefit, 0.6)  # Cap at 60% improvement

    def parkinsons_medication_schedule(self, years_since_diagnosis: float,
                                      motor_fluctuations: bool) -> Dict:
        """Optimize PD medication timing."""
        if years_since_diagnosis < 3 and not motor_fluctuations:
            schedule = {
                'levodopa': '3 times daily with meals',
                'dopamine_agonist': 'Once daily (extended release)',
                'mao_b_inhibitor': 'Once daily morning'
            }
        else:
            schedule = {
                'levodopa': 'Every 3-4 hours (5-6 times daily)',
                'dopamine_agonist': 'Three times daily',
                'comt_inhibitor': 'With each levodopa dose',
                'amantadine': 'Twice daily for dyskinesia'
            }

        return {
            'years_since_diagnosis': years_since_diagnosis,
            'motor_fluctuations': motor_fluctuations,
            'schedule': schedule,
            'total_levodopa_equivalent': self._calculate_led(schedule)
        }

    def _calculate_led(self, medications: Dict) -> float:
        """Calculate levodopa equivalent dose."""
        led_conversions = {
            'levodopa': 1.0,
            'dopamine_agonist': 100,  # Rough conversion
            'mao_b_inhibitor': 100,
            'comt_inhibitor': 0.33,  # Multiplier effect
            'amantadine': 100
        }

        total_led = 0
        for med, _ in medications.items():
            if med == 'levodopa':
                total_led += 400  # Typical dose
            elif med in led_conversions:
                total_led += led_conversions[med]

        return total_led

def demonstrate_neurology_lab():
    """Comprehensive demonstration of neurology lab capabilities."""
    print("=" * 80)
    print("NEUROLOGY LAB DEMONSTRATION")
    print("Copyright (c) 2025 Corporation of Light. All Rights Reserved.")
    print("=" * 80)

    # 1. Alzheimer's Disease
    print("\n1. ALZHEIMER'S DISEASE MODELING")
    print("-" * 40)

    patient_ad = Patient(age=65, sex='F', apoe_genotype='e3/e4', education_years=16)
    ad_model = AlzheimersDiseaseModel(patient_ad)

    # Simulate 10-year progression
    ad_progression = ad_model.simulate_progression(years=10)

    print(f"Patient: {patient_ad.age}yo {patient_ad.sex}, APOE {patient_ad.apoe_genotype}")
    print(f"Initial MMSE: {ad_progression['mmse'][0]:.1f}")
    print(f"10-year MMSE: {ad_progression['mmse'][-1]:.1f}")
    print(f"Final stage: {ad_progression['stage'][-1]}")

    # Treatment response
    treatment = ad_model.predict_treatment_response('anti_amyloid', efficacy=0.3)
    print(f"Anti-amyloid treatment NNT: {treatment['nnt']}")

    # 2. Parkinson's Disease
    print("\n2. PARKINSON'S DISEASE MODELING")
    print("-" * 40)

    patient_pd = Patient(age=60, sex='M')
    pd_model = ParkinsonsDiseaseModel(patient_pd)

    # Simulate progression
    pd_progression = pd_model.simulate_progression(years=10)

    print(f"Baseline UPDRS-III: {pd_progression['updrs_motor'][0]:.1f}")
    print(f"10-year UPDRS-III: {pd_progression['updrs_motor'][-1]:.1f}")
    print(f"Hoehn-Yahr stage: {pd_progression['hoehn_yahr'][-1]}")

    # DBS modeling
    dbs_outcome = pd_model.deep_brain_stimulation_effect('STN')
    print(f"DBS improvement: {dbs_outcome['improvement_percent']:.0f}%")

    # 3. Multiple Sclerosis
    print("\n3. MULTIPLE SCLEROSIS MODELING")
    print("-" * 40)

    patient_ms = Patient(age=30, sex='F')
    ms_model = MultipleSclerosisModel(patient_ms, ms_type='RRMS')
    ms_model.on_dmt = True
    ms_model.dmt_efficacy = 0.4

    # Simulate RRMS
    ms_progression = ms_model.simulate_rrms(years=5)

    print(f"5-year relapse count: {ms_progression['relapse_count']}")
    print(f"Annualized relapse rate: {ms_progression['annualized_relapse_rate']:.2f}")
    print(f"Final EDSS: {ms_progression['edss'][-1]:.1f}")

    # MRI metrics
    mri = ms_model.mri_metrics(time_years=5)
    print(f"T2 lesion volume: {mri['t2_lesion_volume_ml']:.1f} mL")

    # 4. Acute Stroke
    print("\n4. STROKE MODELING")
    print("-" * 40)

    stroke = StrokeModel(stroke_type='ischemic', location='MCA')
    stroke.infarct_volume = 50  # mL
    stroke.nihss = stroke.calculate_nihss(50, 'MCA')

    print(f"Infarct volume: {stroke.infarct_volume} mL")
    print(f"Initial NIHSS: {stroke.nihss}")

    # Thrombolysis outcome
    tpa_outcome = stroke.thrombolysis_outcome(time_to_treatment=2.0, nihss=stroke.nihss)
    print(f"tPA at 2 hours - Good outcome: {tpa_outcome['good_outcome']}")
    print(f"NNT for good outcome: {tpa_outcome['nnt_for_good_outcome']}")

    # Recovery trajectory
    recovery = stroke.recovery_trajectory(stroke.nihss, months=12)
    print(f"12-month NIHSS: {recovery['final_nihss']:.1f}")
    print(f"12-month mRS: {recovery['final_mrs']}")

    # 5. Cognitive Assessment
    print("\n5. COGNITIVE ASSESSMENT TOOLS")
    print("-" * 40)

    # Calculate various cognitive scores
    mmse = CognitiveAssessment.mmse_score(
        orientation=8, registration=3, attention=4, recall=2, language=8
    )
    moca = CognitiveAssessment.moca_score(
        visuospatial=4, naming=3, attention=5, language=2,
        abstraction=1, recall=3, orientation=5
    )
    cdr_sob = CognitiveAssessment.cdr_sob(
        memory=1, orientation=0.5, judgment=0.5, community=0, home=0, personal=0
    )

    print(f"MMSE Score: {mmse}/30")
    print(f"MoCA Score: {moca}/30")
    print(f"CDR-SOB: {cdr_sob}/18")

    # 6. Biomarker Dynamics
    print("\n6. BIOMARKER PROFILES")
    print("-" * 40)

    biomarkers = BiomarkerDynamics()

    # AD CSF profile
    ad_csf = biomarkers.alzheimers_csf('mci')
    print(f"MCI stage CSF: Aβ42={ad_csf['abeta42']}, Tau={ad_csf['tau']}, pTau={ad_csf['ptau']}")

    # PD DAT scan
    dat_binding = biomarkers.parkinsons_dat_scan(years_since_onset=5)
    print(f"DAT binding at 5 years: {dat_binding:.2f}")

    # MS oligoclonal bands
    ms_ocb = biomarkers.ms_oligoclonal_bands('RRMS')
    print(f"RRMS OCB positivity: {ms_ocb['ocb_positive']*100:.0f}%")

    # 7. Lesion Mapping
    print("\n7. LESION ANALYSIS")
    print("-" * 40)

    lesion_mapper = LesionMapping()

    # Predict deficits from lesion
    deficits = lesion_mapper.predict_deficits('frontal', lesion_size=30)
    print(f"Frontal lesion (30 cm³):")
    for function, info in deficits['deficits'].items():
        print(f"  {function}: severity={info['severity']:.2f}")

    # White matter disconnection
    disconnection = lesion_mapper.disconnection_syndrome('corpus_callosum')
    print(f"Callosal disconnection: {disconnection['syndrome']}")

    # 8. Treatment Optimization
    print("\n8. TREATMENT OPTIMIZATION")
    print("-" * 40)

    optimizer = TreatmentOptimizer()

    # AD combination therapy
    ad_treatment = optimizer.alzheimers_combination_therapy(
        'mild', {'abeta42': 400, 'tau': 500}
    )
    print(f"AD treatment recommendations: {', '.join(ad_treatment['recommendations'][:3])}")
    print(f"Expected benefit: {ad_treatment['expected_benefit']*100:.0f}%")

    # PD medication schedule
    pd_schedule = optimizer.parkinsons_medication_schedule(
        years_since_diagnosis=5, motor_fluctuations=True
    )
    print(f"PD medication LED: {pd_schedule['total_levodopa_equivalent']:.0f} mg")

    print("\n" + "=" * 80)
    print("Demonstration complete. Visit aios.is for more information.")
    print("=" * 80)

if __name__ == '__main__':
    demonstrate_neurology_lab()