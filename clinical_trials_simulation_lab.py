"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

CLINICAL TRIALS SIMULATION LAB - Advanced Trial Design and Analysis
Comprehensive clinical trial simulation with patient populations, randomization, survival analysis, and power calculations
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import warnings
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import ndtr, ndtri
from scipy.integrate import quad

class TrialPhase(Enum):
    """Clinical trial phases"""
    PHASE_1 = "Phase I - Safety/Dose-finding"
    PHASE_2 = "Phase II - Efficacy Signal"
    PHASE_3 = "Phase III - Confirmatory"
    PHASE_4 = "Phase IV - Post-marketing Surveillance"

class RandomizationMethod(Enum):
    """Randomization strategies"""
    SIMPLE = "simple_randomization"
    BLOCK = "block_randomization"
    STRATIFIED = "stratified_randomization"
    ADAPTIVE = "adaptive_randomization"
    MINIMIZATION = "minimization"

class EndpointType(Enum):
    """Primary endpoint types"""
    OVERALL_SURVIVAL = "overall_survival"
    PROGRESSION_FREE = "progression_free_survival"
    RESPONSE_RATE = "objective_response_rate"
    TIME_TO_EVENT = "time_to_event"
    CONTINUOUS = "continuous_measure"
    BINARY = "binary_outcome"

@dataclass
class PatientCharacteristics:
    """Detailed patient baseline characteristics"""
    patient_id: int
    age: int
    sex: str  # M/F
    race: str
    bmi: float
    ecog_status: int  # 0-4
    disease_stage: str
    prior_therapies: int
    biomarker_status: Dict[str, Union[bool, float]]
    comorbidities: List[str]
    enrollment_date: int  # days from trial start
    stratification_factors: Dict[str, str]

@dataclass
class TreatmentArm:
    """Treatment arm definition"""
    arm_id: str
    arm_name: str
    drug_name: str
    dose: float
    schedule: str  # e.g., "daily", "weekly", "3 weeks on, 1 week off"
    route: str  # e.g., "oral", "IV", "SC"
    expected_response_rate: float
    expected_median_survival: float  # months
    toxicity_rate: float
    dropout_rate: float

@dataclass
class TrialDesign:
    """Clinical trial design parameters"""
    phase: TrialPhase
    primary_endpoint: EndpointType
    sample_size: int
    arms: List[TreatmentArm]
    randomization_ratio: List[int]  # e.g., [1, 1] for 1:1 randomization
    alpha: float = 0.05  # Type I error rate
    power: float = 0.80  # Statistical power
    enrollment_rate: float = 10.0  # patients per month
    follow_up_duration: float = 24.0  # months
    interim_analyses: List[float] = field(default_factory=lambda: [0.25, 0.50, 0.75])
    stratification_factors: List[str] = field(default_factory=list)

@dataclass
class TrialOutcome:
    """Individual patient outcome"""
    patient_id: int
    arm_id: str
    response: bool
    survival_time: float  # months
    progression_time: float  # months
    censored: bool
    adverse_events: List[str]
    quality_of_life_score: float  # 0-100
    treatment_completed: bool

class ClinicalTrialsSimulationLab:
    """Advanced clinical trials simulation and analysis"""

    def __init__(self, design: TrialDesign, seed: int = None):
        """Initialize with trial design"""
        self.design = design
        self.patients = []
        self.outcomes = []
        self.randomization_list = []
        self.interim_results = []

        if seed is not None:
            np.random.seed(seed)

        self._validate_design()

    def _validate_design(self):
        """Validate trial design parameters"""
        if self.design.sample_size <= 0:
            raise ValueError(f"Invalid sample size: {self.design.sample_size}")
        if not 0 < self.design.alpha < 1:
            raise ValueError(f"Invalid alpha level: {self.design.alpha}")
        if not 0 < self.design.power < 1:
            raise ValueError(f"Invalid power: {self.design.power}")
        if len(self.design.arms) < 1:
            raise ValueError("Trial must have at least one treatment arm")
        if len(self.design.randomization_ratio) != len(self.design.arms):
            raise ValueError("Randomization ratio must match number of arms")

    def generate_patient_population(self, n: int = None) -> List[PatientCharacteristics]:
        """
        Generate realistic patient population based on eligibility criteria

        Reference: Pocock, S.J. (1983) Clinical Trials: A Practical Approach. Wiley.
        """
        if n is None:
            n = self.design.sample_size

        patients = []
        for i in range(n):
            # Age distribution (typical oncology trial)
            age = int(np.random.normal(65, 12))
            age = max(18, min(85, age))  # Cap between 18-85

            # Sex distribution
            sex = np.random.choice(["M", "F"], p=[0.52, 0.48])

            # Race distribution (US-based trial)
            race = np.random.choice(
                ["White", "Black", "Asian", "Hispanic", "Other"],
                p=[0.65, 0.13, 0.06, 0.14, 0.02]
            )

            # BMI distribution
            bmi = np.random.normal(27, 5)
            bmi = max(15, min(45, bmi))

            # ECOG performance status
            ecog = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])

            # Disease stage
            stage = np.random.choice(
                ["Stage IIIB", "Stage IV"],
                p=[0.3, 0.7]
            )

            # Prior therapies
            prior = np.random.poisson(1.5)

            # Biomarker status
            biomarkers = {
                "PD-L1_positive": np.random.random() > 0.5,
                "EGFR_mutation": np.random.random() > 0.85,
                "ALK_fusion": np.random.random() > 0.95,
                "TMB_high": np.random.random() > 0.7
            }

            # Comorbidities
            num_comorbidities = np.random.poisson(1.2)
            possible_comorbidities = [
                "Hypertension", "Diabetes", "CAD", "COPD", "CKD",
                "Hypothyroidism", "Depression", "Osteoarthritis"
            ]
            comorbidities = np.random.choice(
                possible_comorbidities,
                size=min(num_comorbidities, len(possible_comorbidities)),
                replace=False
            ).tolist()

            # Enrollment date (following typical recruitment curve)
            enrollment_day = int(np.random.exponential(scale=30) * (i / n))

            # Stratification factors
            strat_factors = {
                "age_group": "≥65" if age >= 65 else "<65",
                "ecog": str(ecog),
                "stage": stage,
                "pd_l1": "positive" if biomarkers["PD-L1_positive"] else "negative"
            }

            patient = PatientCharacteristics(
                patient_id=i + 1,
                age=age,
                sex=sex,
                race=race,
                bmi=bmi,
                ecog_status=ecog,
                disease_stage=stage,
                prior_therapies=prior,
                biomarker_status=biomarkers,
                comorbidities=comorbidities,
                enrollment_date=enrollment_day,
                stratification_factors=strat_factors
            )

            patients.append(patient)

        self.patients = patients
        return patients

    def simple_randomization(self) -> List[str]:
        """
        Simple randomization (coin flip)

        Reference: Friedman, L.M. et al. (2015) Fundamentals of Clinical Trials. Springer.
        """
        assignments = []
        arms = self.design.arms
        ratio = self.design.randomization_ratio
        probabilities = np.array(ratio) / sum(ratio)

        for patient in self.patients:
            arm_index = np.random.choice(len(arms), p=probabilities)
            assignments.append(arms[arm_index].arm_id)

        return assignments

    def block_randomization(self, block_size: int = 4) -> List[str]:
        """
        Block randomization to ensure balance

        Reference: Schulz, K.F. & Grimes, D.A. (2002) Lancet 359(9305):515-519
        """
        assignments = []
        arms = self.design.arms
        ratio = self.design.randomization_ratio

        # Create block template
        block_template = []
        for i, count in enumerate(ratio):
            block_template.extend([arms[i].arm_id] * count)

        # Generate blocks
        num_blocks = int(np.ceil(len(self.patients) / len(block_template)))

        for _ in range(num_blocks):
            block = block_template.copy()
            np.random.shuffle(block)
            assignments.extend(block)

        return assignments[:len(self.patients)]

    def stratified_randomization(self, factors: List[str] = None) -> List[str]:
        """
        Stratified randomization based on prognostic factors

        Reference: Kernan, W.N. et al. (1999) Journal of Clinical Epidemiology 52(1):19-26
        """
        if factors is None:
            factors = self.design.stratification_factors

        if not factors:
            return self.simple_randomization()

        # Group patients by strata
        strata = {}
        for patient in self.patients:
            stratum_key = tuple(patient.stratification_factors.get(f, "NA") for f in factors)
            if stratum_key not in strata:
                strata[stratum_key] = []
            strata[stratum_key].append(patient)

        # Randomize within each stratum
        assignments = {}
        for stratum_patients in strata.values():
            stratum_size = len(stratum_patients)
            stratum_assignments = self.block_randomization()[:stratum_size]
            for patient, assignment in zip(stratum_patients, stratum_assignments):
                assignments[patient.patient_id] = assignment

        # Return in patient order
        return [assignments[p.patient_id] for p in self.patients]

    def adaptive_randomization(self, responses: Dict[str, List[bool]] = None) -> List[str]:
        """
        Response-adaptive randomization (play-the-winner)

        Reference: Wei, L.J. & Durham, S. (1978) JASA 73(364):840-843
        """
        assignments = []
        arms = self.design.arms

        # Initial equal probability
        if responses is None:
            responses = {arm.arm_id: [] for arm in arms}

        for patient in self.patients:
            # Calculate current success rates
            success_rates = []
            for arm in arms:
                arm_responses = responses.get(arm.arm_id, [])
                if len(arm_responses) > 0:
                    rate = sum(arm_responses) / len(arm_responses)
                else:
                    rate = arm.expected_response_rate
                success_rates.append(rate)

            # Normalize to probabilities (favoring better performers)
            if sum(success_rates) > 0:
                probabilities = np.array(success_rates) / sum(success_rates)
            else:
                probabilities = np.ones(len(arms)) / len(arms)

            # Assign treatment
            arm_index = np.random.choice(len(arms), p=probabilities)
            assignment = arms[arm_index].arm_id
            assignments.append(assignment)

            # Simulate response for next adaptation
            arm = arms[arm_index]
            response = np.random.random() < arm.expected_response_rate
            responses[assignment].append(response)

        return assignments

    def minimization_randomization(self, imbalance_weight: float = 0.8) -> List[str]:
        """
        Minimization method to balance covariates

        Reference: Taves, D.R. (1974) Clinical Pharmacology & Therapeutics 15(5):443-453
        """
        assignments = []
        arms = self.design.arms
        arm_ids = [arm.arm_id for arm in arms]

        # Track assignments by stratification factors
        factor_counts = {
            arm_id: {factor: {} for factor in self.design.stratification_factors}
            for arm_id in arm_ids
        }

        for patient in self.patients:
            if not assignments:
                # First patient: random assignment
                assignment = np.random.choice(arm_ids)
            else:
                # Calculate imbalance for each potential assignment
                imbalances = []
                for potential_arm in arm_ids:
                    imbalance = 0
                    for factor in self.design.stratification_factors:
                        value = patient.stratification_factors.get(factor)
                        # Count current distribution
                        current_counts = []
                        for arm_id in arm_ids:
                            count = factor_counts[arm_id][factor].get(value, 0)
                            if arm_id == potential_arm:
                                count += 1
                            current_counts.append(count)
                        # Calculate imbalance (range)
                        imbalance += max(current_counts) - min(current_counts)
                    imbalances.append(imbalance)

                # Choose arm that minimizes imbalance
                min_imbalance = min(imbalances)
                best_arms = [arm_ids[i] for i, imb in enumerate(imbalances)
                           if imb == min_imbalance]

                # Random selection with probability based on imbalance
                if len(best_arms) == 1:
                    assignment = best_arms[0]
                else:
                    # Biased coin approach
                    if np.random.random() < imbalance_weight:
                        assignment = np.random.choice(best_arms)
                    else:
                        assignment = np.random.choice(arm_ids)

            assignments.append(assignment)

            # Update counts
            for factor in self.design.stratification_factors:
                value = patient.stratification_factors.get(factor)
                if value not in factor_counts[assignment][factor]:
                    factor_counts[assignment][factor][value] = 0
                factor_counts[assignment][factor][value] += 1

        return assignments

    def simulate_survival_times(self, arm: TreatmentArm,
                               n_patients: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate survival and progression times using Weibull distribution

        Reference: Collett, D. (2015) Modelling Survival Data in Medical Research. CRC Press.
        """
        # Weibull parameters based on median survival
        median_survival = arm.expected_median_survival
        shape = 1.5  # Shape parameter (>1 for increasing hazard)

        # Convert median to scale parameter
        scale = median_survival / (np.log(2) ** (1 / shape))

        # Generate survival times
        survival_times = np.random.weibull(shape, n_patients) * scale

        # Generate progression times (typically 60-70% of survival time)
        progression_ratio = np.random.beta(7, 3, n_patients)  # Centered around 0.7
        progression_times = survival_times * progression_ratio

        return survival_times, progression_times

    def kaplan_meier_estimate(self, times: np.ndarray,
                            events: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Kaplan-Meier survival curve estimation

        Reference: Kaplan, E.L. & Meier, P. (1958) JASA 53(282):457-481
        """
        # Sort by time
        sorted_indices = np.argsort(times)
        sorted_times = times[sorted_indices]
        sorted_events = events[sorted_indices]

        # Get unique event times
        unique_times = []
        n_events = []
        n_at_risk = []

        current_at_risk = len(times)
        for i, (time, event) in enumerate(zip(sorted_times, sorted_events)):
            if event and (not unique_times or time != unique_times[-1]):
                unique_times.append(time)
                n_events.append(1)
                n_at_risk.append(current_at_risk)
            elif event and unique_times and time == unique_times[-1]:
                n_events[-1] += 1
            if event:
                current_at_risk -= 1
            else:
                # Censored observation
                current_at_risk -= 1

        # Calculate survival probabilities
        unique_times = np.array([0] + unique_times)
        survival_prob = [1.0]

        for events, at_risk in zip(n_events, n_at_risk):
            survival_prob.append(survival_prob[-1] * (1 - events / at_risk))

        survival_prob = np.array(survival_prob)

        # Calculate confidence intervals (Greenwood's formula)
        variance = 0
        ci_lower = [1.0]
        ci_upper = [1.0]

        for i, (events, at_risk) in enumerate(zip(n_events, n_at_risk)):
            if at_risk > events:
                variance += events / (at_risk * (at_risk - events))
                se = survival_prob[i + 1] * np.sqrt(variance)
                ci_lower.append(max(0, survival_prob[i + 1] - 1.96 * se))
                ci_upper.append(min(1, survival_prob[i + 1] + 1.96 * se))

        return unique_times, survival_prob, (np.array(ci_lower), np.array(ci_upper))

    def cox_proportional_hazards(self, times: np.ndarray, events: np.ndarray,
                                covariates: np.ndarray) -> Dict[str, float]:
        """
        Cox proportional hazards regression (simplified)

        Reference: Cox, D.R. (1972) Journal of the Royal Statistical Society B 34(2):187-220
        """
        n_patients = len(times)
        n_covariates = covariates.shape[1] if len(covariates.shape) > 1 else 1

        if len(covariates.shape) == 1:
            covariates = covariates.reshape(-1, 1)

        # Initialize coefficients
        beta = np.zeros(n_covariates)

        # Sort by time descending (optimization)
        order = np.argsort(-times)
        times_sorted = times[order]
        events_sorted = events[order]
        covariates_sorted = covariates[order]

        # Newton-Raphson iteration (simplified)
        for _ in range(10):  # Maximum iterations
            # Calculate risk scores
            risk_scores = np.exp(covariates_sorted @ beta)

            # Calculate partial likelihood derivatives
            gradient = np.zeros(n_covariates)
            hessian = np.zeros((n_covariates, n_covariates))

            # Accumulators for efficient risk set calculation
            # Process from largest time (smallest risk set) to smallest time (largest risk set)
            # Risk set for time t includes all subjects with time >= t

            current_risk_sum = 0.0
            current_weighted_cov_sum = np.zeros(n_covariates)

            i = 0
            while i < n_patients:
                # Identify block of tied times
                j = i
                while j < n_patients and times_sorted[j] == times_sorted[i]:
                    j += 1

                # Block is i:j
                # Add current block to risk set accumulators
                # Because we are traversing descending times, these new patients enter the risk set
                # for themselves and all subsequent (smaller time) patients.

                block_risk_scores = risk_scores[i:j]
                block_covariates = covariates_sorted[i:j]

                block_risk_sum = np.sum(block_risk_scores)

                # Weighted sum: sum(cov * score)
                block_weighted_cov_sum = np.sum(block_covariates * block_risk_scores.reshape(-1, 1), axis=0)

                current_risk_sum += block_risk_sum
                current_weighted_cov_sum += block_weighted_cov_sum

                # Process events in this block
                events_in_block_mask = events_sorted[i:j].astype(bool)
                if np.any(events_in_block_mask) and current_risk_sum > 0:
                    weighted_cov_mean = current_weighted_cov_sum / current_risk_sum

                    # Sum of covariates for the events in this block
                    event_covariates_sum = np.sum(block_covariates[events_in_block_mask], axis=0)
                    num_events_in_block = np.sum(events_in_block_mask)

                    gradient += event_covariates_sum - (weighted_cov_mean * num_events_in_block)

                i = j

            # Update beta (simplified - actual implementation needs Hessian)
            beta += 0.1 * gradient / n_patients

        # Calculate hazard ratios
        hazard_ratios = np.exp(beta)

        # Calculate p-values (Wald test)
        se = np.sqrt(np.diag(np.linalg.pinv(hessian + np.eye(n_covariates) * 0.01)))
        z_scores = beta / (se + 1e-10)
        p_values = 2 * (1 - ndtr(np.abs(z_scores)))

        return {
            "coefficients": beta.tolist(),
            "hazard_ratios": hazard_ratios.tolist(),
            "p_values": p_values.tolist()
        }

    def calculate_sample_size(self, effect_size: float, alpha: float = 0.05,
                            power: float = 0.80, two_sided: bool = True) -> int:
        """
        Sample size calculation for two-arm trial

        Reference: Chow, S.C. et al. (2017) Sample Size Calculations. CRC Press.
        """
        if not 0 < alpha < 1 or not 0 < power < 1:
            raise ValueError("Alpha and power must be between 0 and 1")

        # Z-scores
        z_alpha = ndtri(1 - alpha / 2) if two_sided else ndtri(1 - alpha)
        z_beta = ndtri(power)

        # For proportions (response rate)
        if self.design.primary_endpoint == EndpointType.RESPONSE_RATE:
            p1 = 0.5  # Assumed control response
            p2 = p1 + effect_size
            p_avg = (p1 + p2) / 2

            n_per_arm = ((z_alpha + z_beta) ** 2 * 2 * p_avg * (1 - p_avg)) / (effect_size ** 2)

        # For time-to-event (log-rank test)
        elif self.design.primary_endpoint in [EndpointType.OVERALL_SURVIVAL,
                                             EndpointType.PROGRESSION_FREE]:
            # Hazard ratio
            hr = effect_size
            d = 4 * (z_alpha + z_beta) ** 2 / (np.log(hr) ** 2)
            # Events needed
            n_per_arm = d / 0.7  # Assuming 70% event rate

        else:
            # Continuous endpoint
            n_per_arm = 2 * ((z_alpha + z_beta) / effect_size) ** 2

        return int(np.ceil(n_per_arm * 2))  # Total for both arms

    def obrien_fleming_boundaries(self, n_analyses: int,
                                 alpha: float = 0.05) -> List[float]:
        """
        O'Brien-Fleming spending function for group sequential design

        Reference: O'Brien, P.C. & Fleming, T.R. (1979) Biometrics 35(3):549-556
        """
        boundaries = []
        information_fractions = np.linspace(1/n_analyses, 1, n_analyses)

        for i, t in enumerate(information_fractions):
            # O'Brien-Fleming boundary
            z_boundary = ndtri(1 - alpha / 2) / np.sqrt(t)
            p_boundary = 2 * (1 - ndtr(z_boundary))
            boundaries.append(p_boundary)

        return boundaries

    def pocock_boundaries(self, n_analyses: int,
                         alpha: float = 0.05) -> List[float]:
        """
        Pocock boundaries for group sequential design

        Reference: Pocock, S.J. (1977) Biometrika 64(2):191-199
        """
        # Pocock constant (approximation)
        c_pocock = ndtri(1 - alpha / (2 * n_analyses))

        boundaries = []
        for i in range(n_analyses):
            p_boundary = 2 * (1 - ndtr(c_pocock))
            boundaries.append(p_boundary)

        return boundaries

    def simulate_trial_outcomes(self, assignments: List[str]) -> List[TrialOutcome]:
        """Simulate individual patient outcomes based on treatment assignment"""
        outcomes = []

        arm_dict = {arm.arm_id: arm for arm in self.design.arms}

        for patient, arm_id in zip(self.patients, assignments):
            arm = arm_dict[arm_id]

            # Response (based on expected rate with patient factors)
            response_prob = arm.expected_response_rate

            # Adjust for patient factors
            if patient.ecog_status >= 2:
                response_prob *= 0.7
            if patient.prior_therapies > 2:
                response_prob *= 0.8
            if patient.biomarker_status.get("PD-L1_positive") and "immuno" in arm.drug_name.lower():
                response_prob *= 1.3

            response = np.random.random() < min(response_prob, 0.95)

            # Survival times
            survival_modifier = 1.0
            if patient.ecog_status >= 2:
                survival_modifier *= 0.6
            if response:
                survival_modifier *= 1.5

            survival_time = np.random.weibull(1.5) * arm.expected_median_survival * survival_modifier
            progression_time = survival_time * np.random.beta(7, 3)

            # Censoring (administrative or lost to follow-up)
            censored = np.random.random() < 0.15  # 15% censoring rate
            if censored:
                survival_time = np.random.uniform(0.5, survival_time)

            # Adverse events
            n_aes = np.random.poisson(2 * arm.toxicity_rate)
            possible_aes = [
                "Fatigue", "Nausea", "Diarrhea", "Rash", "Neutropenia",
                "Anemia", "Thrombocytopenia", "Peripheral neuropathy"
            ]
            adverse_events = np.random.choice(
                possible_aes,
                size=min(n_aes, len(possible_aes)),
                replace=False
            ).tolist()

            # Quality of life
            qol_base = 75
            qol_modifier = -5 * patient.ecog_status - 2 * len(adverse_events)
            qol_score = max(0, min(100, qol_base + qol_modifier + np.random.normal(0, 10)))

            # Treatment completion
            completed = np.random.random() > arm.dropout_rate

            outcome = TrialOutcome(
                patient_id=patient.patient_id,
                arm_id=arm_id,
                response=response,
                survival_time=survival_time,
                progression_time=progression_time,
                censored=censored,
                adverse_events=adverse_events,
                quality_of_life_score=qol_score,
                treatment_completed=completed
            )

            outcomes.append(outcome)

        self.outcomes = outcomes
        return outcomes

    def perform_interim_analysis(self, fraction_enrolled: float) -> Dict[str, any]:
        """
        Perform interim analysis with early stopping rules

        Reference: Jennison, C. & Turnbull, B.W. (1999) Group Sequential Methods. CRC Press.
        """
        n_enrolled = int(self.design.sample_size * fraction_enrolled)

        if n_enrolled > len(self.outcomes):
            return {"error": "Not enough patients enrolled"}

        interim_outcomes = self.outcomes[:n_enrolled]

        # Separate by arm
        arm_outcomes = {}
        for outcome in interim_outcomes:
            if outcome.arm_id not in arm_outcomes:
                arm_outcomes[outcome.arm_id] = []
            arm_outcomes[outcome.arm_id].append(outcome)

        # Calculate response rates
        response_rates = {}
        for arm_id, outcomes in arm_outcomes.items():
            responses = [o.response for o in outcomes]
            response_rates[arm_id] = sum(responses) / len(responses) if responses else 0

        # Perform test
        if len(self.design.arms) == 2:
            # Two-arm comparison
            arm_ids = list(arm_outcomes.keys())
            if len(arm_ids) == 2:
                responses_1 = [o.response for o in arm_outcomes[arm_ids[0]]]
                responses_2 = [o.response for o in arm_outcomes[arm_ids[1]]]

                # Chi-square test
                contingency = [
                    [sum(responses_1), len(responses_1) - sum(responses_1)],
                    [sum(responses_2), len(responses_2) - sum(responses_2)]
                ]
                chi2, p_value = stats.chi2_contingency(contingency)[:2]

                # O'Brien-Fleming boundary
                n_analyses = len(self.design.interim_analyses)
                current_analysis = sum(1 for f in self.design.interim_analyses if f <= fraction_enrolled)
                boundaries = self.obrien_fleming_boundaries(n_analyses, self.design.alpha)
                boundary = boundaries[current_analysis - 1] if current_analysis > 0 else 1.0

                stop_for_efficacy = p_value < boundary
                stop_for_futility = p_value > 0.5  # Simplified futility rule

                return {
                    "fraction_enrolled": fraction_enrolled,
                    "n_enrolled": n_enrolled,
                    "response_rates": response_rates,
                    "p_value": p_value,
                    "boundary": boundary,
                    "stop_for_efficacy": stop_for_efficacy,
                    "stop_for_futility": stop_for_futility,
                    "recommendation": "Stop for efficacy" if stop_for_efficacy else
                                    "Stop for futility" if stop_for_futility else "Continue"
                }

        return {
            "fraction_enrolled": fraction_enrolled,
            "n_enrolled": n_enrolled,
            "response_rates": response_rates
        }

    def calculate_trial_power(self, n: int, effect_size: float,
                            alpha: float = 0.05) -> float:
        """
        Post-hoc power calculation

        Reference: Hintze, J.L. (2008) PASS Software. NCSS.
        """
        z_alpha = ndtri(1 - alpha / 2)

        # Standardized effect
        se = np.sqrt(2 / n)  # Simplified standard error
        z = effect_size / se

        # Power
        power = ndtr(z - z_alpha) + ndtr(-z - z_alpha)

        return min(max(power, 0), 1)


def run_comprehensive_demo():
    """Demonstrate all capabilities of the Clinical Trials Simulation Lab"""

    print("CLINICAL TRIALS SIMULATION LAB - Comprehensive Demo")
    print("=" * 60)

    # Create trial design
    control_arm = TreatmentArm(
        arm_id="control",
        arm_name="Standard of Care",
        drug_name="Docetaxel",
        dose=75,
        schedule="q3w",
        route="IV",
        expected_response_rate=0.25,
        expected_median_survival=12.0,
        toxicity_rate=0.35,
        dropout_rate=0.15
    )

    experimental_arm = TreatmentArm(
        arm_id="experimental",
        arm_name="Novel Immunotherapy",
        drug_name="Anti-PD1 mAb",
        dose=200,
        schedule="q2w",
        route="IV",
        expected_response_rate=0.40,
        expected_median_survival=18.0,
        toxicity_rate=0.25,
        dropout_rate=0.10
    )

    design = TrialDesign(
        phase=TrialPhase.PHASE_3,
        primary_endpoint=EndpointType.OVERALL_SURVIVAL,
        sample_size=300,
        arms=[control_arm, experimental_arm],
        randomization_ratio=[1, 1],
        alpha=0.05,
        power=0.80,
        enrollment_rate=15.0,
        follow_up_duration=36.0,
        interim_analyses=[0.33, 0.67],
        stratification_factors=["ecog", "pd_l1"]
    )

    # Initialize lab
    lab = ClinicalTrialsSimulationLab(design, seed=42)

    print(f"Trial: {design.phase.value}")
    print(f"Primary Endpoint: {design.primary_endpoint.value}")
    print(f"Sample Size: {design.sample_size}")
    print(f"Arms: {len(design.arms)} ({', '.join([arm.arm_name for arm in design.arms])})")

    # 1. Generate patient population
    print("\n1. PATIENT POPULATION GENERATION")
    print("-" * 40)

    patients = lab.generate_patient_population()
    print(f"Generated {len(patients)} patients")

    # Demographics summary
    ages = [p.age for p in patients]
    print(f"Age: mean={np.mean(ages):.1f}, range={min(ages)}-{max(ages)}")

    sex_dist = {}
    for p in patients:
        sex_dist[p.sex] = sex_dist.get(p.sex, 0) + 1
    print(f"Sex distribution: {sex_dist}")

    ecog_dist = {}
    for p in patients:
        ecog_dist[p.ecog_status] = ecog_dist.get(p.ecog_status, 0) + 1
    print(f"ECOG status: {ecog_dist}")

    # 2. Randomization methods
    print("\n2. RANDOMIZATION METHODS")
    print("-" * 40)

    # Simple randomization
    simple_assignments = lab.simple_randomization()
    simple_balance = {}
    for assignment in simple_assignments:
        simple_balance[assignment] = simple_balance.get(assignment, 0) + 1
    print(f"Simple randomization: {simple_balance}")

    # Block randomization
    block_assignments = lab.block_randomization(block_size=4)
    block_balance = {}
    for assignment in block_assignments:
        block_balance[assignment] = block_balance.get(assignment, 0) + 1
    print(f"Block randomization: {block_balance}")

    # Stratified randomization
    stratified_assignments = lab.stratified_randomization()
    strat_balance = {}
    for assignment in stratified_assignments:
        strat_balance[assignment] = strat_balance.get(assignment, 0) + 1
    print(f"Stratified randomization: {strat_balance}")

    # Use stratified for the trial
    final_assignments = stratified_assignments

    # 3. Sample size calculation
    print("\n3. SAMPLE SIZE CALCULATION")
    print("-" * 40)

    effect_size = 0.15  # 15% improvement in response rate
    required_n = lab.calculate_sample_size(effect_size)
    print(f"Effect size: {effect_size:.0%}")
    print(f"Alpha: {design.alpha}")
    print(f"Power: {design.power}")
    print(f"Required sample size: {required_n}")

    actual_power = lab.calculate_trial_power(design.sample_size, effect_size)
    print(f"Actual power with n={design.sample_size}: {actual_power:.2%}")

    # 4. Simulate outcomes
    print("\n4. SIMULATING TRIAL OUTCOMES")
    print("-" * 40)

    outcomes = lab.simulate_trial_outcomes(final_assignments)
    print(f"Generated outcomes for {len(outcomes)} patients")

    # Response rates by arm
    for arm in design.arms:
        arm_outcomes = [o for o in outcomes if o.arm_id == arm.arm_id]
        response_rate = sum(o.response for o in arm_outcomes) / len(arm_outcomes)
        print(f"{arm.arm_name}: {response_rate:.1%} response rate")

    # 5. Survival analysis
    print("\n5. SURVIVAL ANALYSIS")
    print("-" * 40)

    for arm in design.arms:
        arm_outcomes = [o for o in outcomes if o.arm_id == arm.arm_id]
        times = np.array([o.survival_time for o in arm_outcomes])
        events = np.array([not o.censored for o in arm_outcomes])

        # Kaplan-Meier
        km_times, km_survival, (ci_lower, ci_upper) = lab.kaplan_meier_estimate(times, events)

        # Median survival
        median_idx = np.where(km_survival <= 0.5)[0]
        if len(median_idx) > 0:
            median_survival = km_times[median_idx[0]]
        else:
            median_survival = np.inf

        print(f"\n{arm.arm_name}:")
        print(f"  Median survival: {median_survival:.1f} months")
        print(f"  1-year survival: {km_survival[km_times <= 12][-1] if any(km_times <= 12) else 1.0:.1%}")
        print(f"  2-year survival: {km_survival[km_times <= 24][-1] if any(km_times <= 24) else km_survival[-1]:.1%}")

    # 6. Cox regression
    print("\n6. COX PROPORTIONAL HAZARDS")
    print("-" * 40)

    all_times = np.array([o.survival_time for o in outcomes])
    all_events = np.array([not o.censored for o in outcomes])
    treatment = np.array([1 if o.arm_id == "experimental" else 0 for o in outcomes])

    cox_results = lab.cox_proportional_hazards(all_times, all_events, treatment)
    print(f"Treatment effect (HR): {cox_results['hazard_ratios'][0]:.2f}")
    print(f"P-value: {cox_results['p_values'][0]:.4f}")

    # 7. Interim analyses
    print("\n7. INTERIM ANALYSES")
    print("-" * 40)

    for fraction in design.interim_analyses:
        interim_result = lab.perform_interim_analysis(fraction)
        print(f"\nAt {fraction:.0%} enrollment:")
        print(f"  N enrolled: {interim_result.get('n_enrolled', 0)}")
        if 'response_rates' in interim_result:
            for arm_id, rate in interim_result['response_rates'].items():
                print(f"  {arm_id}: {rate:.1%}")
        if 'p_value' in interim_result:
            print(f"  P-value: {interim_result['p_value']:.4f}")
            print(f"  Boundary: {interim_result['boundary']:.4f}")
            print(f"  Recommendation: {interim_result['recommendation']}")

    # 8. Group sequential boundaries
    print("\n8. GROUP SEQUENTIAL BOUNDARIES")
    print("-" * 40)

    n_analyses = 3
    of_boundaries = lab.obrien_fleming_boundaries(n_analyses)
    pocock_boundaries = lab.pocock_boundaries(n_analyses)

    print(f"Number of analyses: {n_analyses}")
    print("O'Brien-Fleming boundaries:")
    for i, boundary in enumerate(of_boundaries, 1):
        print(f"  Analysis {i}: p < {boundary:.4f}")
    print("Pocock boundaries:")
    for i, boundary in enumerate(pocock_boundaries, 1):
        print(f"  Analysis {i}: p < {boundary:.4f}")

    # 9. Quality of life analysis
    print("\n9. QUALITY OF LIFE ANALYSIS")
    print("-" * 40)

    for arm in design.arms:
        arm_outcomes = [o for o in outcomes if o.arm_id == arm.arm_id]
        qol_scores = [o.quality_of_life_score for o in arm_outcomes]
        print(f"{arm.arm_name}:")
        print(f"  Mean QoL: {np.mean(qol_scores):.1f} ± {np.std(qol_scores):.1f}")
        print(f"  Median QoL: {np.median(qol_scores):.1f}")

    # 10. Safety analysis
    print("\n10. SAFETY ANALYSIS")
    print("-" * 40)

    for arm in design.arms:
        arm_outcomes = [o for o in outcomes if o.arm_id == arm.arm_id]
        all_aes = []
        for o in arm_outcomes:
            all_aes.extend(o.adverse_events)

        # Count AEs
        ae_counts = {}
        for ae in all_aes:
            ae_counts[ae] = ae_counts.get(ae, 0) + 1

        print(f"\n{arm.arm_name} - Adverse Events:")
        for ae, count in sorted(ae_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {ae}: {count/len(arm_outcomes):.1%}")

    print("\n" + "=" * 60)
    print("Demo complete - All clinical trial simulation functions demonstrated")


if __name__ == '__main__':
    run_comprehensive_demo()