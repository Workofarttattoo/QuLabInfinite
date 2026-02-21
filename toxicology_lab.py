"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

TOXICOLOGY LAB - Advanced Drug Safety Assessment
Comprehensive toxicology simulation with LD50/LC50, dose-response, PBPK modeling, and ADMET prediction
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import warnings
from scipy.optimize import minimize, curve_fit
from scipy.integrate import odeint, solve_ivp
from scipy.special import ndtr, ndtri, erf
from scipy import stats

class ToxicityEndpoint(Enum):
    """Standard toxicity endpoints"""
    MORTALITY = "mortality"
    ORGAN_DAMAGE = "organ_damage"
    CARCINOGENICITY = "carcinogenicity"
    GENOTOXICITY = "genotoxicity"
    REPRODUCTIVE = "reproductive_toxicity"
    NEUROTOXICITY = "neurotoxicity"
    IMMUNOTOXICITY = "immunotoxicity"
    DEVELOPMENTAL = "developmental_toxicity"

class ExposureRoute(Enum):
    """Routes of drug/chemical exposure"""
    ORAL = "oral"
    INHALATION = "inhalation"
    DERMAL = "dermal"
    INTRAVENOUS = "intravenous"
    INTRAMUSCULAR = "intramuscular"
    SUBCUTANEOUS = "subcutaneous"
    INTRAPERITONEAL = "intraperitoneal"

class Species(Enum):
    """Test species for toxicology studies"""
    MOUSE = "mouse"
    RAT = "rat"
    RABBIT = "rabbit"
    DOG = "dog"
    MONKEY = "monkey"
    HUMAN = "human"
    ZEBRAFISH = "zebrafish"

@dataclass
class CompoundProperties:
    """Physicochemical properties of test compound"""
    name: str
    molecular_weight: float  # g/mol
    log_p: float  # octanol-water partition coefficient
    pka: float  # acid dissociation constant
    solubility: float  # mg/mL in water
    plasma_protein_binding: float  # fraction bound (0-1)
    hydrogen_bond_donors: int
    hydrogen_bond_acceptors: int
    rotatable_bonds: int
    polar_surface_area: float  # Ų
    melting_point: float  # °C
    bioavailability: float  # fraction absorbed (0-1)

@dataclass
class PhysiologicalParameters:
    """Species-specific physiological parameters for PBPK"""
    species: Species
    body_weight: float  # kg
    cardiac_output: float  # L/h
    blood_volume: float  # L
    organ_volumes: Dict[str, float]  # L
    organ_blood_flows: Dict[str, float]  # L/h
    glomerular_filtration_rate: float  # mL/min
    hepatic_clearance: float  # L/h
    breathing_rate: float  # breaths/min
    tidal_volume: float  # L

@dataclass
class ToxicityData:
    """Container for toxicity study results"""
    endpoint: ToxicityEndpoint
    species: Species
    route: ExposureRoute
    doses: np.ndarray  # mg/kg
    responses: np.ndarray  # proportion responding
    n_animals: np.ndarray  # number per dose group
    duration: float  # days
    ld50: Optional[float] = None
    noael: Optional[float] = None  # No Observed Adverse Effect Level
    loael: Optional[float] = None  # Lowest Observed Adverse Effect Level
    bmd: Optional[float] = None  # Benchmark Dose
    confidence_interval: Optional[Tuple[float, float]] = None

@dataclass
class ADMETProfile:
    """ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) predictions"""
    absorption_rate: float  # hr^-1
    bioavailability: float  # fraction
    volume_distribution: float  # L/kg
    clearance: float  # L/h/kg
    half_life: float  # hours
    protein_binding: float  # fraction
    metabolic_stability: float  # % remaining after 1 hour
    cyp_inhibition: Dict[str, float]  # IC50 values for CYP enzymes
    herg_ic50: float  # μM (cardiac safety)
    ames_positive: bool  # mutagenicity
    hepatotoxicity_risk: float  # probability 0-1
    nephrotoxicity_risk: float  # probability 0-1

class ToxicologyLab:
    """Advanced toxicology assessment laboratory"""

    def __init__(self, compound: CompoundProperties):
        """Initialize with compound properties"""
        self.compound = compound
        self.studies = []
        self.pbpk_model = None
        self._validate_compound()

    def _validate_compound(self):
        """Validate compound properties"""
        if self.compound.molecular_weight <= 0:
            raise ValueError(f"Invalid molecular weight: {self.compound.molecular_weight}")
        if not -5 <= self.compound.log_p <= 10:
            warnings.warn(f"Unusual log P value: {self.compound.log_p}")
        if not 0 <= self.compound.plasma_protein_binding <= 1:
            raise ValueError(f"Invalid protein binding: {self.compound.plasma_protein_binding}")

    def probit_analysis(self, doses: np.ndarray, responses: np.ndarray,
                       n_animals: np.ndarray) -> Dict[str, float]:
        """
        Probit analysis for LD50/LC50 calculation

        Reference: Finney, D.J. (1971) Probit Analysis, 3rd ed. Cambridge University Press.
        """
        if len(doses) != len(responses) or len(doses) != len(n_animals):
            raise ValueError("Arrays must have same length")

        # Remove 0% and 100% responses for probit transformation
        valid_idx = (responses > 0) & (responses < 1)
        if sum(valid_idx) < 2:
            raise ValueError("Need at least 2 intermediate response values")

        doses_valid = doses[valid_idx]
        responses_valid = responses[valid_idx]
        n_valid = n_animals[valid_idx]

        # Log transform doses
        log_doses = np.log10(doses_valid)

        # Probit transformation
        probits = ndtri(responses_valid)

        # Weighted linear regression
        weights = n_valid * responses_valid * (1 - responses_valid)

        def probit_model(x, a, b):
            return a + b * x

        # Fit model
        popt, pcov = curve_fit(probit_model, log_doses, probits, sigma=1/np.sqrt(weights))
        slope, intercept = popt[1], popt[0]

        # Calculate LD50 (probit = 5 corresponds to 50% response)
        log_ld50 = (5 - intercept) / slope
        ld50 = 10 ** log_ld50

        # Calculate confidence interval (Fieller's method)
        se_slope = np.sqrt(pcov[1, 1])
        se_intercept = np.sqrt(pcov[0, 0])
        se_ld50 = np.sqrt((se_intercept / slope) ** 2 +
                         ((5 - intercept) * se_slope / slope ** 2) ** 2)

        ci_lower = 10 ** (log_ld50 - 1.96 * se_ld50)
        ci_upper = 10 ** (log_ld50 + 1.96 * se_ld50)

        # Calculate LD10 and LD90
        ld10 = 10 ** ((ndtri(0.1) - intercept) / slope)
        ld90 = 10 ** ((ndtri(0.9) - intercept) / slope)

        return {
            "ld50": ld50,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ld10": ld10,
            "ld90": ld90,
            "slope": slope,
            "intercept": intercept,
            "r_squared": 1 - np.sum((probits - probit_model(log_doses, *popt)) ** 2) /
                           np.sum((probits - np.mean(probits)) ** 2)
        }

    def logit_analysis(self, doses: np.ndarray, responses: np.ndarray) -> Dict[str, float]:
        """
        Logistic regression for dose-response analysis

        Reference: Berkson, J. (1944) JASA 39(227):357-365
        """
        # Log transform doses
        log_doses = np.log10(doses + 1e-10)

        def logistic(x, ld50, slope, bottom=0, top=1):
            return bottom + (top - bottom) / (1 + 10 ** ((ld50 - x) * slope))

        # Initial parameter estimates
        p0 = [np.median(log_doses), 1, min(responses), max(responses)]

        # Fit curve
        try:
            popt, pcov = curve_fit(logistic, log_doses, responses, p0=p0,
                                 bounds=([min(log_doses), 0.1, 0, 0.5],
                                        [max(log_doses), 10, 0.5, 1]))

            log_ld50, hill_slope, bottom, top = popt
            ld50 = 10 ** log_ld50

            # Calculate IC values
            ic20 = 10 ** (log_ld50 - np.log10(4) / hill_slope)
            ic80 = 10 ** (log_ld50 + np.log10(4) / hill_slope)

            # Goodness of fit
            predicted = logistic(log_doses, *popt)
            r_squared = 1 - np.sum((responses - predicted) ** 2) / np.sum((responses - np.mean(responses)) ** 2)

            return {
                "ld50": ld50,
                "hill_slope": hill_slope,
                "bottom": bottom,
                "top": top,
                "ic20": ic20,
                "ic80": ic80,
                "r_squared": r_squared
            }
        except:
            return {"ld50": np.nan, "error": "Curve fitting failed"}

    def hill_equation(self, concentration: float, ic50: float,
                     hill_coefficient: float, top: float = 1.0,
                     bottom: float = 0.0) -> float:
        """
        Hill equation for dose-response modeling

        Reference: Hill, A.V. (1910) Journal of Physiology 40:190-224
        """
        if concentration < 0 or ic50 <= 0:
            raise ValueError("Concentration and IC50 must be positive")

        return bottom + (top - bottom) / (1 + (ic50 / concentration) ** hill_coefficient)

    def benchmark_dose(self, doses: np.ndarray, responses: np.ndarray,
                      bmr: float = 0.1) -> Dict[str, float]:
        """
        Benchmark Dose (BMD) calculation using exponential model

        Reference: Crump, K.S. (1984) Fundamental and Applied Toxicology 4(5):854-871
        """
        # Remove control group if present
        if doses[0] == 0:
            control_response = responses[0]
            doses = doses[1:]
            responses = responses[1:] - control_response
        else:
            control_response = 0

        # Exponential model: P(d) = 1 - exp(-b0 - b1*d)
        def exponential_model(dose, b1):
            return 1 - np.exp(-b1 * dose)

        # Fit model
        popt, pcov = curve_fit(exponential_model, doses, responses, p0=[0.001])
        b1 = popt[0]

        # Calculate BMD for specified benchmark response
        bmd = -np.log(1 - bmr) / b1

        # Calculate BMDL (lower confidence limit)
        se_b1 = np.sqrt(pcov[0, 0])
        b1_lower = b1 - 1.96 * se_b1
        if b1_lower > 0:
            bmdl = -np.log(1 - bmr) / b1_lower
        else:
            bmdl = np.inf

        return {
            "bmd": bmd,
            "bmdl": bmdl,
            "bmr": bmr,
            "model_parameter": b1,
            "standard_error": se_b1
        }

    def pbpk_model(self, time: np.ndarray, dose: float, route: ExposureRoute,
                  params: PhysiologicalParameters) -> Dict[str, np.ndarray]:
        """
        Physiologically-Based Pharmacokinetic (PBPK) modeling

        Reference: Reddy, M. et al. (2005) Physiologically Based Pharmacokinetic Modeling. Wiley.
        """
        # Compartments: blood, liver, kidney, lung, brain, fat, muscle, rest
        n_compartments = 8

        # Partition coefficients (tissue:blood) - estimated from log P
        kp_liver = 2.0 * (10 ** (0.3 * self.compound.log_p))
        kp_kidney = 1.5 * (10 ** (0.25 * self.compound.log_p))
        kp_lung = 0.8 * (10 ** (0.2 * self.compound.log_p))
        kp_brain = 0.5 * (10 ** (0.35 * self.compound.log_p))
        kp_fat = 10 ** (0.7 * self.compound.log_p)
        kp_muscle = 0.6 * (10 ** (0.2 * self.compound.log_p))
        kp_rest = 1.0

        partition_coefficients = np.array([1.0, kp_liver, kp_kidney, kp_lung,
                                         kp_brain, kp_fat, kp_muscle, kp_rest])

        # Volumes (L) - typical for 70 kg human
        volumes = np.array([
            params.blood_volume,
            params.organ_volumes.get("liver", 1.5),
            params.organ_volumes.get("kidney", 0.3),
            params.organ_volumes.get("lung", 0.5),
            params.organ_volumes.get("brain", 1.4),
            params.organ_volumes.get("fat", 10.0),
            params.organ_volumes.get("muscle", 30.0),
            params.organ_volumes.get("rest", 25.0)
        ])

        # Blood flows (L/h) - typical for 70 kg human
        flows = np.array([
            params.cardiac_output,
            params.organ_blood_flows.get("liver", 90),
            params.organ_blood_flows.get("kidney", 72),
            params.organ_blood_flows.get("lung", params.cardiac_output),
            params.organ_blood_flows.get("brain", 45),
            params.organ_blood_flows.get("fat", 15),
            params.organ_blood_flows.get("muscle", 75),
            params.organ_blood_flows.get("rest", 93)
        ])

        # Clearances
        cl_hepatic = params.hepatic_clearance * self.compound.metabolic_stability
        cl_renal = params.glomerular_filtration_rate * 0.06 * (1 - self.compound.plasma_protein_binding)

        # Absorption parameters
        if route == ExposureRoute.ORAL:
            ka = 1.0  # absorption rate constant (h^-1)
            f_abs = self.compound.bioavailability
        elif route == ExposureRoute.INTRAVENOUS:
            ka = 0
            f_abs = 1.0
        else:
            ka = 0.5
            f_abs = 0.5

        def pbpk_ode(t, y):
            """PBPK differential equations"""
            # Unpack concentrations
            c_blood = y[0]
            c_tissues = y[1:n_compartments]
            amount_gut = y[n_compartments] if route == ExposureRoute.ORAL else 0

            # Calculate tissue concentrations
            c_tissue_unbound = c_tissues / partition_coefficients[1:]

            # Blood concentration
            venous_return = np.sum(flows[1:] * c_tissue_unbound) / flows[0]
            arterial = c_blood

            # Derivatives
            dy = np.zeros(n_compartments + 1)

            # Blood
            dy[0] = venous_return - arterial - (cl_hepatic + cl_renal) * c_blood / volumes[0]

            # Tissues
            for i in range(1, n_compartments):
                dy[i] = flows[i] * (arterial - c_tissue_unbound[i-1]) / volumes[i]

            # Gut compartment for oral dosing
            if route == ExposureRoute.ORAL:
                dy[n_compartments] = -ka * amount_gut
                dy[0] += ka * amount_gut * f_abs / volumes[0]

            return dy

        # Initial conditions
        y0 = np.zeros(n_compartments + 1)
        if route == ExposureRoute.ORAL:
            y0[n_compartments] = dose * params.body_weight  # mg
        elif route == ExposureRoute.INTRAVENOUS:
            y0[0] = dose * params.body_weight / volumes[0]  # mg/L

        # Solve ODE
        solution = solve_ivp(pbpk_ode, [time[0], time[-1]], y0, t_eval=time,
                           method='RK45', rtol=1e-6)

        # Extract results
        results = {
            "time": solution.t,
            "blood": solution.y[0],
            "liver": solution.y[1],
            "kidney": solution.y[2],
            "lung": solution.y[3],
            "brain": solution.y[4],
            "fat": solution.y[5],
            "muscle": solution.y[6],
            "rest": solution.y[7]
        }

        # Calculate pharmacokinetic parameters
        auc = np.trapz(results["blood"], results["time"])
        cmax = np.max(results["blood"])
        tmax = results["time"][np.argmax(results["blood"])]

        results["auc"] = auc
        results["cmax"] = cmax
        results["tmax"] = tmax

        return results

    def predict_admet(self) -> ADMETProfile:
        """
        Predict ADMET properties using QSAR models

        Reference: Lipinski, C.A. et al. (2001) Advanced Drug Delivery Reviews 46(1-3):3-26
        """
        # Lipinski's Rule of Five
        lipinski_violations = 0
        if self.compound.molecular_weight > 500:
            lipinski_violations += 1
        if self.compound.log_p > 5:
            lipinski_violations += 1
        if self.compound.hydrogen_bond_donors > 5:
            lipinski_violations += 1
        if self.compound.hydrogen_bond_acceptors > 10:
            lipinski_violations += 1

        # Absorption prediction (empirical model)
        if lipinski_violations <= 1:
            absorption_rate = 2.0 * np.exp(-0.1 * self.compound.molecular_weight / 100)
            bioavailability = 0.8 * (1 - 0.1 * lipinski_violations)
        else:
            absorption_rate = 0.5
            bioavailability = 0.3

        # Volume of distribution (Oie-Tozer equation)
        fu = 1 - self.compound.plasma_protein_binding
        vd = 0.07 + 0.7 * fu + 42 * fu / (1 + 10 ** (self.compound.log_p - 2))

        # Clearance prediction (allometric scaling)
        clearance = 0.5 * (1 + 0.1 * self.compound.log_p) * fu

        # Half-life
        half_life = 0.693 * vd / clearance

        # Metabolic stability (simplified)
        metabolic_stability = 100 * np.exp(-0.2 * self.compound.log_p)

        # CYP inhibition prediction (IC50 in μM)
        cyp_inhibition = {
            "CYP3A4": 10 * np.exp(0.3 * self.compound.log_p),
            "CYP2D6": 15 * np.exp(0.2 * self.compound.log_p),
            "CYP2C9": 20 * np.exp(0.25 * self.compound.log_p),
            "CYP1A2": 25 * np.exp(0.15 * self.compound.log_p),
            "CYP2C19": 18 * np.exp(0.22 * self.compound.log_p)
        }

        # hERG inhibition (cardiac safety)
        herg_ic50 = 5 * np.exp(0.4 * self.compound.log_p)

        # Genotoxicity (Ames test) - simplified prediction
        ames_positive = (self.compound.molecular_weight < 200 and
                        self.compound.log_p > 2 and
                        self.compound.hydrogen_bond_acceptors > 3)

        # Organ toxicity risk (logistic model)
        hepatotox_score = (0.1 * self.compound.log_p +
                          0.05 * self.compound.molecular_weight / 100 -
                          0.2 * self.compound.polar_surface_area / 100)
        hepatotoxicity_risk = 1 / (1 + np.exp(-hepatotox_score))

        nephrotox_score = (0.15 * self.compound.log_p +
                          0.1 * (1 - self.compound.plasma_protein_binding) -
                          0.3)
        nephrotoxicity_risk = 1 / (1 + np.exp(-nephrotox_score))

        return ADMETProfile(
            absorption_rate=absorption_rate,
            bioavailability=bioavailability,
            volume_distribution=vd,
            clearance=clearance,
            half_life=half_life,
            protein_binding=self.compound.plasma_protein_binding,
            metabolic_stability=metabolic_stability,
            cyp_inhibition=cyp_inhibition,
            herg_ic50=herg_ic50,
            ames_positive=ames_positive,
            hepatotoxicity_risk=hepatotoxicity_risk,
            nephrotoxicity_risk=nephrotoxicity_risk
        )

    def organ_toxicity_index(self, dose: float, duration_days: float,
                            organ: str = "liver") -> float:
        """
        Calculate organ-specific toxicity index

        Reference: Krewski, D. et al. (2010) Journal of Toxicology and Environmental Health B 13(2-4):51-138
        """
        # Base toxicity score
        base_score = dose / 100  # Normalize to typical dose

        # Time factor (cumulative effect)
        time_factor = 1 - np.exp(-duration_days / 30)

        # Organ-specific susceptibility
        organ_factors = {
            "liver": 1.0,  # Reference organ
            "kidney": 0.8,
            "heart": 0.6,
            "lung": 0.7,
            "brain": 0.4,  # Blood-brain barrier protection
            "bone_marrow": 1.2  # High susceptibility
        }
        organ_factor = organ_factors.get(organ, 0.5)

        # Compound-specific factors
        lipophilicity_factor = 1 + 0.1 * self.compound.log_p if self.compound.log_p > 0 else 1
        protein_binding_factor = 1 - 0.5 * self.compound.plasma_protein_binding

        # Calculate toxicity index (0-1 scale)
        toxicity_index = min(1.0, base_score * time_factor * organ_factor *
                           lipophilicity_factor * protein_binding_factor)

        return toxicity_index

    def calculate_noael(self, doses: np.ndarray, responses: np.ndarray,
                       control_response: float = 0, alpha: float = 0.05) -> Dict[str, float]:
        """
        Calculate NOAEL (No Observed Adverse Effect Level) and LOAEL

        Reference: OECD (2008) Guidelines for the Testing of Chemicals
        """
        noael = None
        loael = None

        for i, (dose, response) in enumerate(zip(doses, responses)):
            # Statistical test vs control (Dunnett's test approximation)
            if response > control_response:
                # Simple t-test as approximation
                z_score = (response - control_response) / np.sqrt(control_response * (1 - control_response) / 10)
                p_value = 2 * (1 - ndtr(abs(z_score)))

                if p_value < alpha and loael is None:
                    loael = dose
                    if i > 0:
                        noael = doses[i - 1]
                    break

        # If no significant effect found
        if loael is None:
            noael = doses[-1]
            loael = None

        return {
            "noael": noael,
            "loael": loael,
            "control_response": control_response,
            "alpha": alpha
        }

    def therapeutic_index(self, ld50: float, ed50: float) -> float:
        """
        Calculate therapeutic index (safety margin)

        Reference: Muller, P.Y. & Milton, M.N. (2012) Nature Reviews Drug Discovery 11(10):751-761
        """
        if ed50 <= 0:
            raise ValueError("ED50 must be positive")

        return ld50 / ed50

    def allometric_scaling(self, dose_animal: float, weight_animal: float,
                         species_from: Species, species_to: Species) -> float:
        """
        Interspecies dose scaling using allometry

        Reference: Reagan-Shaw, S. et al. (2008) FASEB Journal 22(3):659-661
        """
        # Body surface area normalization factors (km)
        km_factors = {
            Species.MOUSE: 3,
            Species.RAT: 6,
            Species.RABBIT: 12,
            Species.DOG: 20,
            Species.MONKEY: 12,
            Species.HUMAN: 37
        }

        km_from = km_factors.get(species_from, 6)
        km_to = km_factors.get(species_to, 37)

        # Human equivalent dose = Animal dose * (Animal Km / Human Km)
        scaled_dose = dose_animal * (km_from / km_to)

        return scaled_dose

    def margin_of_safety(self, noael: float, human_dose: float,
                        safety_factors: Dict[str, float] = None) -> float:
        """
        Calculate margin of safety with uncertainty factors

        Reference: Dourson, M.L. & Stara, J.F. (1983) Regulatory Toxicology and Pharmacology 3(3):224-238
        """
        if safety_factors is None:
            safety_factors = {
                "interspecies": 10,  # Animal to human
                "intraspecies": 10,  # Human variability
                "subchronic_to_chronic": 1,  # Study duration
                "loael_to_noael": 1,  # If using LOAEL instead of NOAEL
                "database_completeness": 1  # Data quality
            }

        total_uncertainty_factor = np.prod(list(safety_factors.values()))

        reference_dose = noael / total_uncertainty_factor
        mos = reference_dose / human_dose

        return mos

    def genotoxicity_battery(self) -> Dict[str, bool]:
        """
        Predict standard genotoxicity test battery results

        Reference: ICH S2(R1) Guidance on Genotoxicity Testing
        """
        # Simplified predictions based on structural features
        results = {}

        # Ames test (bacterial mutation)
        results["ames"] = (self.compound.molecular_weight < 300 and
                          self.compound.log_p > 1.5 and
                          self.compound.hydrogen_bond_acceptors > 2)

        # Chromosome aberration
        results["chromosome_aberration"] = (self.compound.log_p > 3 and
                                           self.compound.molecular_weight > 200)

        # Micronucleus test
        results["micronucleus"] = results["chromosome_aberration"]

        # Gene mutation in mammalian cells
        results["mammalian_gene_mutation"] = (results["ames"] or
                                             self.compound.molecular_weight < 250)

        # Overall genotoxicity risk
        results["overall_risk"] = any([results["ames"],
                                      results["chromosome_aberration"],
                                      results["micronucleus"]])

        return results


def run_comprehensive_demo():
    """Demonstrate all capabilities of the Toxicology Lab"""

    print("TOXICOLOGY LAB - Comprehensive Drug Safety Assessment Demo")
    print("=" * 60)

    # Create test compound
    compound = CompoundProperties(
        name="TestCompound-X",
        molecular_weight=385.5,
        log_p=3.2,
        pka=8.4,
        solubility=5.0,
        plasma_protein_binding=0.92,
        hydrogen_bond_donors=2,
        hydrogen_bond_acceptors=5,
        rotatable_bonds=7,
        polar_surface_area=78.3,
        melting_point=156.0,
        bioavailability=0.65
    )

    # Initialize lab
    lab = ToxicologyLab(compound)

    print(f"Compound: {compound.name}")
    print(f"MW: {compound.molecular_weight} g/mol")
    print(f"Log P: {compound.log_p}")
    print(f"Protein binding: {compound.plasma_protein_binding:.0%}")

    # 1. LD50 calculation using probit analysis
    print("\n1. LD50 CALCULATION (PROBIT ANALYSIS)")
    print("-" * 40)

    doses = np.array([10, 30, 100, 300, 1000])  # mg/kg
    responses = np.array([0.05, 0.15, 0.40, 0.75, 0.95])  # mortality proportion
    n_animals = np.array([20, 20, 20, 20, 20])

    probit_results = lab.probit_analysis(doses, responses, n_animals)
    print(f"LD50: {probit_results['ld50']:.1f} mg/kg")
    print(f"95% CI: [{probit_results['ci_lower']:.1f}, {probit_results['ci_upper']:.1f}] mg/kg")
    print(f"LD10: {probit_results['ld10']:.1f} mg/kg")
    print(f"LD90: {probit_results['ld90']:.1f} mg/kg")
    print(f"Probit slope: {probit_results['slope']:.3f}")

    # 2. Dose-response curve (logistic)
    print("\n2. DOSE-RESPONSE MODELING (LOGISTIC)")
    print("-" * 40)

    logit_results = lab.logit_analysis(doses, responses)
    if not np.isnan(logit_results['ld50']):
        print(f"IC50: {logit_results['ld50']:.1f} mg/kg")
        print(f"Hill slope: {logit_results['hill_slope']:.2f}")
        print(f"R²: {logit_results['r_squared']:.3f}")

    # 3. Benchmark dose
    print("\n3. BENCHMARK DOSE (BMD)")
    print("-" * 40)

    bmd_results = lab.benchmark_dose(doses, responses, bmr=0.1)
    print(f"BMD10: {bmd_results['bmd']:.1f} mg/kg")
    print(f"BMDL10: {bmd_results['bmdl']:.1f} mg/kg")
    print(f"Model parameter: {bmd_results['model_parameter']:.4f}")

    # 4. NOAEL/LOAEL determination
    print("\n4. NOAEL/LOAEL DETERMINATION")
    print("-" * 40)

    noael_results = lab.calculate_noael(doses, responses, control_response=0.05)
    print(f"NOAEL: {noael_results['noael']:.1f} mg/kg" if noael_results['noael'] else "NOAEL: Not determined")
    print(f"LOAEL: {noael_results['loael']:.1f} mg/kg" if noael_results['loael'] else "LOAEL: Not determined")

    # 5. PBPK modeling
    print("\n5. PBPK MODELING")
    print("-" * 40)

    # Human physiological parameters
    human_params = PhysiologicalParameters(
        species=Species.HUMAN,
        body_weight=70,
        cardiac_output=360,  # L/h
        blood_volume=5.0,
        organ_volumes={
            "liver": 1.5, "kidney": 0.3, "lung": 0.5,
            "brain": 1.4, "fat": 10.0, "muscle": 30.0, "rest": 25.0
        },
        organ_blood_flows={
            "liver": 90, "kidney": 72, "lung": 360,
            "brain": 45, "fat": 15, "muscle": 75, "rest": 93
        },
        glomerular_filtration_rate=120,
        hepatic_clearance=50,
        breathing_rate=12,
        tidal_volume=0.5
    )

    # Simulate single oral dose
    time_points = np.linspace(0, 24, 100)  # 24 hours
    pbpk_results = lab.pbpk_model(time_points, dose=10, route=ExposureRoute.ORAL, params=human_params)

    print(f"Oral dose: 10 mg/kg")
    print(f"Cmax: {pbpk_results['cmax']:.2f} mg/L")
    print(f"Tmax: {pbpk_results['tmax']:.1f} hours")
    print(f"AUC: {pbpk_results['auc']:.1f} mg·h/L")
    print(f"Brain Cmax: {np.max(pbpk_results['brain']):.2f} mg/L")
    print(f"Liver Cmax: {np.max(pbpk_results['liver']):.2f} mg/L")

    # 6. ADMET predictions
    print("\n6. ADMET PREDICTIONS")
    print("-" * 40)

    admet = lab.predict_admet()
    print(f"Bioavailability: {admet.bioavailability:.0%}")
    print(f"Volume of distribution: {admet.volume_distribution:.1f} L/kg")
    print(f"Clearance: {admet.clearance:.2f} L/h/kg")
    print(f"Half-life: {admet.half_life:.1f} hours")
    print(f"Metabolic stability: {admet.metabolic_stability:.0f}% remaining")

    print("\nCYP Inhibition IC50 values:")
    for cyp, ic50 in admet.cyp_inhibition.items():
        risk = "High" if ic50 < 1 else "Moderate" if ic50 < 10 else "Low"
        print(f"  {cyp}: {ic50:.1f} μM ({risk} risk)")

    print(f"\nhERG IC50: {admet.herg_ic50:.1f} μM")
    print(f"Ames test: {'Positive' if admet.ames_positive else 'Negative'}")
    print(f"Hepatotoxicity risk: {admet.hepatotoxicity_risk:.0%}")
    print(f"Nephrotoxicity risk: {admet.nephrotoxicity_risk:.0%}")

    # 7. Organ toxicity assessment
    print("\n7. ORGAN TOXICITY INDEX")
    print("-" * 40)

    organs = ["liver", "kidney", "heart", "lung", "brain", "bone_marrow"]
    dose_test = 100  # mg/kg
    duration = 28  # days

    print(f"Dose: {dose_test} mg/kg for {duration} days")
    for organ in organs:
        tox_index = lab.organ_toxicity_index(dose_test, duration, organ)
        severity = "Severe" if tox_index > 0.7 else "Moderate" if tox_index > 0.4 else "Mild"
        print(f"  {organ.title()}: {tox_index:.2f} ({severity})")

    # 8. Therapeutic index
    print("\n8. THERAPEUTIC INDEX")
    print("-" * 40)

    ed50 = 15  # mg/kg (effective dose)
    ld50 = probit_results['ld50']
    ti = lab.therapeutic_index(ld50, ed50)
    print(f"ED50: {ed50} mg/kg")
    print(f"LD50: {ld50:.1f} mg/kg")
    print(f"Therapeutic Index: {ti:.1f}")
    print(f"Safety assessment: {'Safe' if ti > 10 else 'Caution' if ti > 3 else 'Narrow margin'}")

    # 9. Allometric scaling
    print("\n9. ALLOMETRIC SCALING")
    print("-" * 40)

    rat_dose = 50  # mg/kg
    rat_weight = 0.25  # kg
    human_dose = lab.allometric_scaling(rat_dose, rat_weight, Species.RAT, Species.HUMAN)
    print(f"Rat dose: {rat_dose} mg/kg")
    print(f"Human equivalent dose: {human_dose:.1f} mg/kg")

    # Apply safety factor
    safety_factor = 10
    safe_human_dose = human_dose / safety_factor
    print(f"Safe starting dose (10x safety): {safe_human_dose:.2f} mg/kg")

    # 10. Margin of safety
    print("\n10. MARGIN OF SAFETY")
    print("-" * 40)

    if noael_results['noael']:
        clinical_dose = 5  # mg/kg
        mos = lab.margin_of_safety(noael_results['noael'], clinical_dose)
        print(f"NOAEL: {noael_results['noael']:.1f} mg/kg")
        print(f"Clinical dose: {clinical_dose} mg/kg")
        print(f"Margin of Safety: {mos:.1f}")
        print(f"Assessment: {'Adequate' if mos > 1 else 'Insufficient'}")

    # 11. Genotoxicity battery
    print("\n11. GENOTOXICITY ASSESSMENT")
    print("-" * 40)

    genotox = lab.genotoxicity_battery()
    print("Genotoxicity test battery predictions:")
    print(f"  Ames test: {'Positive' if genotox['ames'] else 'Negative'}")
    print(f"  Chromosome aberration: {'Positive' if genotox['chromosome_aberration'] else 'Negative'}")
    print(f"  Micronucleus: {'Positive' if genotox['micronucleus'] else 'Negative'}")
    print(f"  Mammalian gene mutation: {'Positive' if genotox['mammalian_gene_mutation'] else 'Negative'}")
    print(f"  Overall genotoxicity risk: {'High' if genotox['overall_risk'] else 'Low'}")

    # 12. Create toxicity study record
    print("\n12. TOXICITY STUDY SUMMARY")
    print("-" * 40)

    study = ToxicityData(
        endpoint=ToxicityEndpoint.MORTALITY,
        species=Species.RAT,
        route=ExposureRoute.ORAL,
        doses=doses,
        responses=responses,
        n_animals=n_animals,
        duration=28,
        ld50=probit_results['ld50'],
        noael=noael_results['noael'],
        loael=noael_results['loael'],
        bmd=bmd_results['bmd'],
        confidence_interval=(probit_results['ci_lower'], probit_results['ci_upper'])
    )

    print(f"Study type: {study.endpoint.value}")
    print(f"Species: {study.species.value}")
    print(f"Route: {study.route.value}")
    print(f"Duration: {study.duration} days")
    print(f"Key findings:")
    print(f"  LD50: {study.ld50:.1f} mg/kg")
    print(f"  NOAEL: {study.noael:.1f} mg/kg" if study.noael else "  NOAEL: Not determined")
    print(f"  BMD: {study.bmd:.1f} mg/kg")

    print("\n" + "=" * 60)
    print("Demo complete - All toxicology lab functions demonstrated")


if __name__ == '__main__':
    run_comprehensive_demo()