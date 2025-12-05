"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ONCOLOGY LAB - Advanced Cancer Modeling and Treatment Simulation
Comprehensive oncology simulation with real growth models, drug response curves, and combination therapy analysis
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import warnings
from scipy.optimize import minimize_scalar, curve_fit
from scipy.integrate import odeint
from scipy.special import erf

class TumorType(Enum):
    """Standard tumor classification"""
    NSCLC = "non_small_cell_lung_cancer"
    SCLC = "small_cell_lung_cancer"
    BREAST_ERPOS = "breast_er_positive"
    BREAST_TNBC = "breast_triple_negative"
    COLORECTAL = "colorectal"
    PANCREATIC = "pancreatic"
    OVARIAN = "ovarian"
    MELANOMA = "melanoma"
    GLIOBLASTOMA = "glioblastoma"
    PROSTATE = "prostate"

class DrugClass(Enum):
    """Drug mechanism of action classification"""
    ALKYLATING = "alkylating_agent"
    ANTIMETABOLITE = "antimetabolite"
    TAXANE = "taxane"
    PLATINUM = "platinum_compound"
    TARGETED_EGFR = "egfr_inhibitor"
    TARGETED_HER2 = "her2_inhibitor"
    IMMUNOTHERAPY = "checkpoint_inhibitor"
    HORMONE = "hormone_therapy"
    ANGIOGENESIS = "angiogenesis_inhibitor"

@dataclass
class TumorCharacteristics:
    """Comprehensive tumor parameters based on clinical data"""
    initial_volume: float  # cm³
    doubling_time: float  # days (clinical observation)
    carrying_capacity: float  # maximum achievable volume (cm³)
    hypoxic_fraction: float  # fraction of hypoxic cells (0-1)
    proliferation_index: float  # Ki-67 positivity (0-1)
    mutation_burden: float  # mutations per megabase
    angiogenic_switch: bool  # whether tumor has vascularized
    grade: int  # histological grade (1-4)
    tnm_stage: str  # TNM staging (e.g., "T2N1M0")
    biomarkers: Dict[str, float] = field(default_factory=dict)  # e.g., {"HER2": 3.0, "PD-L1": 0.8}

@dataclass
class DrugProperties:
    """Pharmacokinetic/pharmacodynamic properties"""
    name: str
    drug_class: DrugClass
    ic50: float  # μM - concentration for 50% inhibition
    hill_coefficient: float  # steepness of dose-response
    max_efficacy: float  # maximum achievable effect (0-1)
    half_life: float  # hours
    clearance: float  # L/hr
    volume_distribution: float  # L/kg
    protein_binding: float  # fraction bound to plasma proteins
    bioavailability: float  # fraction absorbed (for oral drugs)
    myelosuppression_risk: float  # bone marrow toxicity (0-1)

@dataclass
class PatientProfile:
    """Detailed patient characteristics affecting treatment"""
    age: int
    weight: float  # kg
    height: float  # cm
    bsa: float  # body surface area (m²)
    gender: str
    performance_status: int  # ECOG 0-4
    creatinine_clearance: float  # mL/min
    liver_function: float  # relative to normal (1.0 = normal)
    albumin: float  # g/dL
    prior_therapies: List[str] = field(default_factory=list)
    genetic_mutations: Dict[str, bool] = field(default_factory=dict)

class OncologyLab:
    """Advanced oncology simulation laboratory"""

    def __init__(self, patient: PatientProfile, tumor: TumorCharacteristics):
        """Initialize with patient and tumor data"""
        self.patient = patient
        self.tumor = tumor
        self.treatment_history = []
        self.resistance_factors = {}
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input parameters"""
        if self.patient.age < 0 or self.patient.age > 120:
            raise ValueError(f"Invalid age: {self.patient.age}")
        if self.patient.weight <= 0:
            raise ValueError(f"Invalid weight: {self.patient.weight}")
        if self.tumor.initial_volume <= 0:
            raise ValueError(f"Invalid tumor volume: {self.tumor.initial_volume}")
        if not 0 <= self.tumor.hypoxic_fraction <= 1:
            raise ValueError(f"Invalid hypoxic fraction: {self.tumor.hypoxic_fraction}")
        if self.patient.performance_status not in range(5):
            raise ValueError(f"Invalid ECOG status: {self.patient.performance_status}")

    def gompertz_growth(self, t: np.ndarray, v0: float = None,
                       alpha: float = None, beta: float = None) -> np.ndarray:
        """
        Gompertz tumor growth model - most accurate for solid tumors
        dV/dt = α * V * ln(β/V)

        Reference: Laird, A.K. (1964) British Journal of Cancer 18(3):490-502
        """
        if v0 is None:
            v0 = self.tumor.initial_volume
        if alpha is None:
            # Convert doubling time to growth rate
            alpha = np.log(2) / self.tumor.doubling_time
        if beta is None:
            beta = self.tumor.carrying_capacity

        # Analytical solution: V(t) = β * exp(ln(V0/β) * exp(-α*t))
        return beta * np.exp(np.log(v0/beta) * np.exp(-alpha * t))

    def logistic_growth(self, t: np.ndarray, v0: float = None,
                       r: float = None, k: float = None) -> np.ndarray:
        """
        Logistic tumor growth model
        dV/dt = r * V * (1 - V/K)

        Reference: Verhulst, P.F. (1838) Correspondance mathématique et physique 10:113-121
        """
        if v0 is None:
            v0 = self.tumor.initial_volume
        if r is None:
            r = np.log(2) / self.tumor.doubling_time
        if k is None:
            k = self.tumor.carrying_capacity

        # Analytical solution: V(t) = K / (1 + ((K-V0)/V0) * exp(-r*t))
        return k / (1 + ((k - v0) / v0) * np.exp(-r * t))

    def exponential_growth(self, t: np.ndarray, v0: float = None,
                          r: float = None) -> np.ndarray:
        """
        Simple exponential growth - early stage tumors only
        dV/dt = r * V

        Reference: Skipper, H.E. (1964) Cancer Chemotherapy Reports 45:5-28
        """
        if v0 is None:
            v0 = self.tumor.initial_volume
        if r is None:
            r = np.log(2) / self.tumor.doubling_time

        return v0 * np.exp(r * t)

    def bertalanffy_growth(self, t: np.ndarray, v0: float = None,
                          a: float = None, b: float = None) -> np.ndarray:
        """
        Von Bertalanffy growth model - accounts for metabolism
        dV/dt = a * V^(2/3) - b * V

        Reference: von Bertalanffy, L. (1957) Quarterly Review of Biology 32(3):217-231
        """
        if v0 is None:
            v0 = self.tumor.initial_volume
        if b is None:
            b = np.log(2) / self.tumor.doubling_time
        if a is None:
            # Estimate from carrying capacity
            a = b * (self.tumor.carrying_capacity ** (1/3))

        def bertalanffy_ode(v, t):
            return a * (v ** (2/3)) - b * v

        solution = odeint(bertalanffy_ode, v0, t)
        return solution.flatten()

    def hill_drug_response(self, concentration: float, drug: DrugProperties) -> float:
        """
        Hill equation for drug dose-response
        E = Emax * C^n / (IC50^n + C^n)

        Reference: Hill, A.V. (1910) Journal of Physiology 40:190-224
        """
        if concentration < 0:
            raise ValueError("Drug concentration must be non-negative")

        n = drug.hill_coefficient
        emax = drug.max_efficacy
        ic50 = drug.ic50

        # Account for protein binding
        free_concentration = concentration * (1 - drug.protein_binding)

        # Hill equation
        effect = emax * (free_concentration ** n) / (ic50 ** n + free_concentration ** n)

        # Adjust for hypoxic fraction (reduced efficacy in hypoxic cells)
        hypoxia_penalty = 1 - (self.tumor.hypoxic_fraction * 0.5)

        return effect * hypoxia_penalty

    def emax_model(self, concentration: float, emax: float, ec50: float,
                   gamma: float = 1.0) -> float:
        """
        Emax pharmacodynamic model
        E = Emax * C^γ / (EC50^γ + C^γ)

        Reference: Holford, N.H. & Sheiner, L.B. (1981) Clinical Pharmacokinetics 6(6):429-453
        """
        if concentration < 0:
            return 0.0

        return emax * (concentration ** gamma) / (ec50 ** gamma + concentration ** gamma)

    def bliss_independence(self, drug1_effect: float, drug2_effect: float) -> float:
        """
        Bliss independence model for drug combination
        E_combo = E1 + E2 - E1*E2

        Reference: Bliss, C.I. (1939) Annals of Applied Biology 26(3):585-615
        """
        if not (0 <= drug1_effect <= 1 and 0 <= drug2_effect <= 1):
            raise ValueError("Drug effects must be between 0 and 1")

        return drug1_effect + drug2_effect - (drug1_effect * drug2_effect)

    def loewe_additivity(self, drug1_conc: float, drug1_ic50: float,
                        drug2_conc: float, drug2_ic50: float) -> float:
        """
        Loewe additivity model for drug synergy
        CI = (D1/IC50_1) + (D2/IC50_2)
        CI < 1: synergy, CI = 1: additivity, CI > 1: antagonism

        Reference: Loewe, S. (1953) Arzneimittel-Forschung 3:285-290
        """
        if drug1_ic50 <= 0 or drug2_ic50 <= 0:
            raise ValueError("IC50 values must be positive")

        combination_index = (drug1_conc / drug1_ic50) + (drug2_conc / drug2_ic50)

        # Convert CI to effect
        if combination_index < 0.1:
            return 0.0
        elif combination_index < 1.0:
            # Synergy - enhanced effect
            return 0.5 + 0.5 * combination_index  # Maps 0.1-1 to 0.55-1.0
        elif combination_index == 1.0:
            # Perfect additivity
            return 0.5
        else:
            # Antagonism - reduced effect
            return 0.5 / combination_index  # Decreases as CI increases

    def simulate_resistance_evolution(self, initial_sensitive_fraction: float,
                                    treatment_pressure: float,
                                    mutation_rate: float,
                                    generations: int) -> np.ndarray:
        """
        Model evolution of drug resistance using branching process

        Reference: Iwasa, Y. et al. (2006) Genetics 172(4):2557-2566
        """
        if not 0 <= initial_sensitive_fraction <= 1:
            raise ValueError("Initial sensitive fraction must be between 0 and 1")

        sensitive = np.zeros(generations + 1)
        resistant = np.zeros(generations + 1)

        sensitive[0] = initial_sensitive_fraction
        resistant[0] = 1 - initial_sensitive_fraction

        for gen in range(generations):
            # Sensitive cells can mutate to resistant
            new_resistant = sensitive[gen] * mutation_rate

            # Selection pressure affects growth
            sensitive_growth = (1 - treatment_pressure) * sensitive[gen]
            resistant_growth = resistant[gen]  # Unaffected by treatment

            # Update populations
            sensitive[gen + 1] = sensitive_growth - new_resistant
            resistant[gen + 1] = resistant_growth + new_resistant

            # Normalize to maintain total fraction = 1
            total = sensitive[gen + 1] + resistant[gen + 1]
            if total > 0:
                sensitive[gen + 1] /= total
                resistant[gen + 1] /= total

        return resistant  # Return resistant fraction over time

    def calculate_biomarker_score(self, biomarkers: Dict[str, float],
                                 weights: Dict[str, float] = None) -> float:
        """
        Calculate composite biomarker score for treatment selection

        Reference: Ballman, K.V. (2015) Journal of Clinical Oncology 33(33):3968-3971
        """
        if weights is None:
            # Default weights based on clinical importance
            weights = {
                "HER2": 0.3,
                "ER": 0.25,
                "PR": 0.15,
                "PD-L1": 0.2,
                "Ki-67": 0.1
            }

        score = 0.0
        total_weight = 0.0

        for marker, value in biomarkers.items():
            if marker in weights:
                # Normalize biomarker values to 0-1 range
                if marker == "HER2":
                    # HER2 scored 0-3+
                    normalized = min(value / 3.0, 1.0)
                elif marker in ["ER", "PR", "PD-L1", "Ki-67"]:
                    # These are percentages
                    normalized = min(value / 100.0, 1.0)
                else:
                    normalized = value

                score += normalized * weights[marker]
                total_weight += weights[marker]

        return score / total_weight if total_weight > 0 else 0.0

    def predict_survival_probability(self, time_months: float,
                                    median_survival: float,
                                    shape_param: float = 1.0) -> float:
        """
        Weibull survival model for cancer patients
        S(t) = exp(-(t/λ)^k)

        Reference: Weibull, W. (1951) Journal of Applied Mechanics 18(3):293-297
        """
        if time_months < 0 or median_survival <= 0:
            raise ValueError("Time and median survival must be positive")

        # Convert median to scale parameter
        scale = median_survival / (np.log(2) ** (1 / shape_param))

        # Weibull survival function
        survival_prob = np.exp(-(time_months / scale) ** shape_param)

        # Adjust for performance status
        ps_factor = 1.0 - (self.patient.performance_status * 0.15)

        return min(survival_prob * ps_factor, 1.0)

    def calculate_response_rate(self, tumor_type: TumorType,
                               drug_class: DrugClass) -> Dict[str, float]:
        """
        Clinical response rates based on real trial data

        References: Multiple phase III trials from JCO, NEJM, Lancet Oncology
        """
        # Response rates from clinical trials (simplified subset)
        response_data = {
            (TumorType.NSCLC, DrugClass.PLATINUM): {
                "complete_response": 0.02,
                "partial_response": 0.28,
                "stable_disease": 0.35,
                "progression": 0.35
            },
            (TumorType.NSCLC, DrugClass.TARGETED_EGFR): {
                "complete_response": 0.01,
                "partial_response": 0.71,
                "stable_disease": 0.20,
                "progression": 0.08
            },
            (TumorType.BREAST_TNBC, DrugClass.TAXANE): {
                "complete_response": 0.15,
                "partial_response": 0.35,
                "stable_disease": 0.25,
                "progression": 0.25
            },
            (TumorType.MELANOMA, DrugClass.IMMUNOTHERAPY): {
                "complete_response": 0.19,
                "partial_response": 0.36,
                "stable_disease": 0.15,
                "progression": 0.30
            }
        }

        # Default response if specific combination not in database
        default_response = {
            "complete_response": 0.05,
            "partial_response": 0.25,
            "stable_disease": 0.40,
            "progression": 0.30
        }

        return response_data.get((tumor_type, drug_class), default_response)

    def simulate_tumor_heterogeneity(self, num_clones: int = 5,
                                    time_points: int = 10) -> np.ndarray:
        """
        Simulate clonal evolution and tumor heterogeneity

        Reference: Nowell, P.C. (1976) Science 194(4260):23-28
        """
        if num_clones < 1 or time_points < 1:
            raise ValueError("Need at least 1 clone and 1 time point")

        # Initialize clone populations
        clone_populations = np.zeros((time_points, num_clones))
        clone_populations[0, 0] = 1.0  # Start with single dominant clone

        # Random fitness advantages for each clone
        fitness = np.random.uniform(0.8, 1.2, num_clones)

        for t in range(1, time_points):
            for c in range(num_clones):
                if clone_populations[t-1, c] > 0:
                    # Growth with fitness advantage
                    growth = clone_populations[t-1, c] * fitness[c]

                    # Mutation can create new clones
                    if c < num_clones - 1 and np.random.random() < self.tumor.mutation_burden / 100:
                        mutation_fraction = 0.01
                        clone_populations[t, c+1] += growth * mutation_fraction
                        growth *= (1 - mutation_fraction)

                    clone_populations[t, c] = growth

            # Normalize to sum to 1
            total = clone_populations[t].sum()
            if total > 0:
                clone_populations[t] /= total

        return clone_populations

    def calculate_toxicity_score(self, drug: DrugProperties,
                                dose: float) -> Dict[str, float]:
        """
        Calculate organ-specific toxicity scores

        Reference: Common Terminology Criteria for Adverse Events (CTCAE) v5.0
        """
        # Base toxicity proportional to dose
        dose_factor = min(dose / 100, 2.0)  # Normalize to typical dose

        toxicities = {
            "hematologic": drug.myelosuppression_risk * dose_factor,
            "hepatic": 0.2 * dose_factor / self.patient.liver_function,
            "renal": 0.15 * dose_factor * (100 / self.patient.creatinine_clearance),
            "cardiac": 0.1 * dose_factor * (self.patient.age / 50),
            "neurologic": 0.05 * dose_factor
        }

        # Adjust for patient factors
        if self.patient.performance_status >= 2:
            for key in toxicities:
                toxicities[key] *= 1.3

        # Normalize to 0-1 range
        for key in toxicities:
            toxicities[key] = min(toxicities[key], 1.0)

        return toxicities


def run_comprehensive_demo():
    """Demonstrate all capabilities of the Oncology Lab"""

    print("ONCOLOGY LAB - Comprehensive Cancer Modeling Demo")
    print("=" * 60)

    # Create patient profile
    patient = PatientProfile(
        age=65,
        weight=70,
        height=170,
        bsa=1.8,
        gender="female",
        performance_status=1,
        creatinine_clearance=85,
        liver_function=0.9,
        albumin=3.8,
        prior_therapies=["carboplatin", "paclitaxel"],
        genetic_mutations={"EGFR": True, "KRAS": False}
    )

    # Create tumor characteristics
    tumor = TumorCharacteristics(
        initial_volume=4.2,  # cm³
        doubling_time=90,  # days
        carrying_capacity=500,  # cm³
        hypoxic_fraction=0.15,
        proliferation_index=0.35,
        mutation_burden=10.5,
        angiogenic_switch=True,
        grade=3,
        tnm_stage="T2N1M0",
        biomarkers={"HER2": 0, "ER": 0, "PR": 0, "PD-L1": 45, "Ki-67": 35}
    )

    # Initialize lab
    lab = OncologyLab(patient, tumor)

    print(f"\nPatient: {patient.age}yo {patient.gender}, ECOG {patient.performance_status}")
    print(f"Tumor: {tumor.initial_volume:.1f} cm³, Grade {tumor.grade}, Stage {tumor.tnm_stage}")
    print(f"Biomarkers: PD-L1={tumor.biomarkers['PD-L1']}%, Ki-67={tumor.biomarkers['Ki-67']}%")

    # 1. Tumor growth models
    print("\n1. TUMOR GROWTH MODELS (365 days)")
    print("-" * 40)

    time_points = np.linspace(0, 365, 13)  # Monthly for 1 year

    gompertz = lab.gompertz_growth(time_points)
    logistic = lab.logistic_growth(time_points)
    exponential = lab.exponential_growth(time_points)
    bertalanffy = lab.bertalanffy_growth(time_points)

    print(f"Initial volume: {tumor.initial_volume:.1f} cm³")
    print(f"After 1 year:")
    print(f"  Gompertz model: {gompertz[-1]:.1f} cm³")
    print(f"  Logistic model: {logistic[-1]:.1f} cm³")
    print(f"  Exponential model: {exponential[-1]:.1f} cm³")
    print(f"  Von Bertalanffy model: {bertalanffy[-1]:.1f} cm³")

    # 2. Drug response
    print("\n2. DRUG RESPONSE CURVES")
    print("-" * 40)

    drug = DrugProperties(
        name="Erlotinib",
        drug_class=DrugClass.TARGETED_EGFR,
        ic50=2.0,  # μM
        hill_coefficient=1.5,
        max_efficacy=0.85,
        half_life=36,  # hours
        clearance=4.5,  # L/hr
        volume_distribution=3.5,  # L/kg
        protein_binding=0.95,
        bioavailability=0.6,
        myelosuppression_risk=0.2
    )

    concentrations = np.logspace(-1, 2, 50)  # 0.1 to 100 μM
    responses = [lab.hill_drug_response(c, drug) for c in concentrations]

    ic50_response = lab.hill_drug_response(drug.ic50, drug)
    print(f"Drug: {drug.name} (EGFR inhibitor)")
    print(f"IC50: {drug.ic50} μM")
    print(f"Response at IC50: {ic50_response:.1%}")
    print(f"Maximum efficacy: {drug.max_efficacy:.1%}")

    # 3. Combination therapy
    print("\n3. COMBINATION THERAPY ANALYSIS")
    print("-" * 40)

    drug1_effect = 0.6  # 60% tumor reduction
    drug2_effect = 0.5  # 50% tumor reduction

    bliss_combo = lab.bliss_independence(drug1_effect, drug2_effect)
    print(f"Drug 1 effect: {drug1_effect:.0%}")
    print(f"Drug 2 effect: {drug2_effect:.0%}")
    print(f"Bliss independence prediction: {bliss_combo:.0%}")

    # Loewe additivity
    ci = lab.loewe_additivity(1.5, 2.0, 2.0, 3.0)
    synergy_type = "synergy" if ci < 0.5 else "additivity" if ci < 1.5 else "antagonism"
    print(f"Loewe combination index: {ci:.2f} ({synergy_type})")

    # 4. Resistance evolution
    print("\n4. RESISTANCE EVOLUTION")
    print("-" * 40)

    generations = 20
    resistance = lab.simulate_resistance_evolution(
        initial_sensitive_fraction=0.99,
        treatment_pressure=0.8,
        mutation_rate=0.001,
        generations=generations
    )

    print(f"Initial resistant fraction: {(1-0.99):.1%}")
    print(f"After {generations} cycles:")
    print(f"  Resistant fraction: {resistance[-1]:.1%}")
    print(f"  Sensitive fraction: {(1-resistance[-1]):.1%}")

    # 5. Biomarker scoring
    print("\n5. BIOMARKER ANALYSIS")
    print("-" * 40)

    biomarker_score = lab.calculate_biomarker_score(tumor.biomarkers)
    print(f"Composite biomarker score: {biomarker_score:.2f}")
    print("Interpretation: " +
          ("High" if biomarker_score > 0.6 else "Moderate" if biomarker_score > 0.3 else "Low") +
          " likelihood of immunotherapy response")

    # 6. Survival prediction
    print("\n6. SURVIVAL ANALYSIS")
    print("-" * 40)

    median_survival = 14.5  # months for this tumor type

    survival_6mo = lab.predict_survival_probability(6, median_survival)
    survival_12mo = lab.predict_survival_probability(12, median_survival)
    survival_24mo = lab.predict_survival_probability(24, median_survival)

    print(f"Median survival: {median_survival} months")
    print(f"6-month survival probability: {survival_6mo:.1%}")
    print(f"12-month survival probability: {survival_12mo:.1%}")
    print(f"24-month survival probability: {survival_24mo:.1%}")

    # 7. Clinical response rates
    print("\n7. CLINICAL RESPONSE RATES")
    print("-" * 40)

    response_rates = lab.calculate_response_rate(TumorType.NSCLC, DrugClass.TARGETED_EGFR)
    print("NSCLC with EGFR inhibitor:")
    for response_type, rate in response_rates.items():
        print(f"  {response_type.replace('_', ' ').title()}: {rate:.1%}")

    # 8. Tumor heterogeneity
    print("\n8. TUMOR HETEROGENEITY SIMULATION")
    print("-" * 40)

    clone_evolution = lab.simulate_tumor_heterogeneity(num_clones=5, time_points=10)

    print("Clonal evolution over time:")
    print(f"Initial: 100% Clone 1")
    print(f"Final distribution:")
    for i, fraction in enumerate(clone_evolution[-1]):
        if fraction > 0.01:
            print(f"  Clone {i+1}: {fraction:.1%}")

    # 9. Toxicity assessment
    print("\n9. TOXICITY ASSESSMENT")
    print("-" * 40)

    toxicities = lab.calculate_toxicity_score(drug, dose=150)  # mg

    print(f"Predicted toxicities for {drug.name} at 150mg:")
    for organ, score in toxicities.items():
        severity = "Severe" if score > 0.7 else "Moderate" if score > 0.4 else "Mild"
        print(f"  {organ.title()}: {score:.2f} ({severity})")

    print("\n" + "=" * 60)
    print("Demo complete - All oncology lab functions demonstrated")


if __name__ == '__main__':
    run_comprehensive_demo()