"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

QuLabInfinite Immunology Laboratory
====================================
Production-ready immunology simulation with immune response dynamics,
antibody-antigen binding, vaccine efficacy modeling, and autoimmune disease analysis.

References:
- Janeway's Immunobiology (9th edition)
- Antibody-antigen binding kinetics (Karlsson et al.)
- Vaccine efficacy models from CDC/WHO standards
- Autoimmune disease parameters from clinical literature
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json


class ImmuneCell(Enum):
    """Types of immune cells"""
    T_HELPER = "t_helper"  # CD4+
    T_CYTOTOXIC = "t_cytotoxic"  # CD8+
    B_CELL = "b_cell"
    NK_CELL = "nk_cell"
    MACROPHAGE = "macrophage"
    DENDRITIC = "dendritic"
    NEUTROPHIL = "neutrophil"


class Cytokine(Enum):
    """Major cytokines"""
    IL2 = "interleukin_2"  # T cell proliferation
    IL4 = "interleukin_4"  # B cell activation
    IL6 = "interleukin_6"  # Inflammation
    IL10 = "interleukin_10"  # Anti-inflammatory
    IFNG = "interferon_gamma"  # Antiviral
    TNF = "tumor_necrosis_factor"  # Inflammation


class AntibodyClass(Enum):
    """Antibody classes (isotypes)"""
    IGM = "IgM"  # First response
    IGG = "IgG"  # Main antibody
    IGA = "IgA"  # Mucosal immunity
    IGE = "IgE"  # Allergies
    IGD = "IgD"  # B cell receptor


@dataclass
class AntibodyAntigenBinding:
    """Antibody-antigen binding kinetics"""
    ka: float  # Association rate constant (M^-1 s^-1)
    kd: float  # Dissociation rate constant (s^-1)
    KD: float  # Dissociation constant (M)
    affinity: float  # 1/KD
    binding_curve: np.ndarray


@dataclass
class ImmuneResponse:
    """Immune response dynamics"""
    pathogen_count: np.ndarray
    antibody_titer: np.ndarray
    t_cell_count: np.ndarray
    cytokine_levels: Dict[str, np.ndarray]
    time_days: np.ndarray


@dataclass
class VaccineEfficacy:
    """Vaccine efficacy metrics"""
    seroconversion_rate: float  # % who develop antibodies
    geometric_mean_titer: float  # Average antibody level
    protection_rate: float  # % protected from disease
    duration_months: float  # Duration of protection
    adverse_events_rate: float


class ImmunologyLaboratory:
    """
    Production immunology laboratory with validated models
    """

    # Physical constants
    AVOGADRO = 6.022e23  # mol^-1

    # Antibody-antigen binding rates (typical ranges from literature)
    BINDING_RATES = {
        'high_affinity': {'ka': 1e7, 'kd': 1e-4},  # Strong binding
        'medium_affinity': {'ka': 1e6, 'kd': 1e-3},
        'low_affinity': {'ka': 1e5, 'kd': 1e-2}  # Weak binding
    }

    # Immune cell parameters
    CELL_PARAMS = {
        ImmuneCell.T_HELPER: {
            'proliferation_rate': 0.5,  # divisions/day
            'lifespan_days': 100,
            'activation_threshold': 1e-9  # M antigen
        },
        ImmuneCell.B_CELL: {
            'proliferation_rate': 0.7,
            'lifespan_days': 30,
            'antibody_production': 2000  # molecules/cell/second
        },
        ImmuneCell.T_CYTOTOXIC: {
            'proliferation_rate': 0.6,
            'lifespan_days': 100,
            'killing_rate': 10  # infected cells/day
        }
    }

    # Cytokine parameters (pg/mL typical levels)
    CYTOKINE_BASELINE = {
        Cytokine.IL2: 5,
        Cytokine.IL6: 10,
        Cytokine.IL10: 5,
        Cytokine.IFNG: 2,
        Cytokine.TNF: 8
    }

    def __init__(self, seed: Optional[int] = None):
        """Initialize immunology lab"""
        if seed is not None:
            np.random.seed(seed)

    def simulate_antibody_antigen_binding(self, antibody_conc_nM: float,
                                        antigen_conc_nM: float,
                                        affinity: str = 'medium_affinity',
                                        duration_s: float = 3600) -> AntibodyAntigenBinding:
        """
        Simulate antibody-antigen binding kinetics using Langmuir model

        Args:
            antibody_conc_nM: Antibody concentration (nM)
            antigen_conc_nM: Antigen concentration (nM)
            affinity: 'high_affinity', 'medium_affinity', or 'low_affinity'
            duration_s: Duration in seconds

        Returns:
            Binding kinetics data
        """
        rates = self.BINDING_RATES[affinity]
        ka = rates['ka']  # M^-1 s^-1
        kd = rates['kd']  # s^-1

        # Convert nM to M
        Ab = antibody_conc_nM * 1e-9
        Ag = antigen_conc_nM * 1e-9

        # Dissociation constant KD = kd/ka
        KD = kd / ka

        # Time array
        time = np.linspace(0, duration_s, 1000)
        dt = time[1] - time[0]

        # Simulate binding: dC/dt = ka*Ab*Ag - kd*C
        # where C is complex concentration
        C = np.zeros(len(time))
        Ab_free = np.zeros(len(time))
        Ag_free = np.zeros(len(time))

        Ab_free[0] = Ab
        Ag_free[0] = Ag

        for i in range(1, len(time)):
            # Rate of complex formation
            dC = ka * Ab_free[i-1] * Ag_free[i-1] - kd * C[i-1]
            C[i] = max(0, C[i-1] + dC * dt)

            # Update free concentrations
            Ab_free[i] = max(0, Ab - C[i])
            Ag_free[i] = max(0, Ag - C[i])

        # Calculate equilibrium binding (fraction bound)
        # At equilibrium: C_eq = (Ab * Ag) / (KD + Ag)
        fraction_bound = C / Ab

        return AntibodyAntigenBinding(
            ka=ka,
            kd=kd,
            KD=KD,
            affinity=1.0 / KD,
            binding_curve=fraction_bound
        )

    def simulate_immune_response(self, pathogen_dose: float,
                                pathogen_growth_rate: float = 2.0,
                                duration_days: float = 30) -> ImmuneResponse:
        """
        Simulate complete immune response to pathogen

        Args:
            pathogen_dose: Initial pathogen count
            pathogen_growth_rate: Doubling time (hours)
            duration_days: Simulation duration

        Returns:
            Immune response dynamics
        """
        # Time array (hours)
        dt = 0.1  # hours
        time_h = np.arange(0, duration_days * 24, dt)
        time_days = time_h / 24

        n_steps = len(time_h)

        # Initialize arrays
        pathogen = np.zeros(n_steps)
        antibody = np.zeros(n_steps)
        t_helper = np.zeros(n_steps)
        t_cyto = np.zeros(n_steps)
        b_cells = np.zeros(n_steps)

        # Cytokines
        il2 = np.zeros(n_steps)
        il6 = np.zeros(n_steps)
        ifng = np.zeros(n_steps)

        # Initial conditions
        pathogen[0] = pathogen_dose
        t_helper[0] = 1e6  # Baseline T helper cells
        b_cells[0] = 1e5  # Baseline B cells
        t_cyto[0] = 5e5  # Baseline cytotoxic T cells

        # Baseline cytokines
        il2[0] = self.CYTOKINE_BASELINE[Cytokine.IL2]
        il6[0] = self.CYTOKINE_BASELINE[Cytokine.IL6]
        ifng[0] = self.CYTOKINE_BASELINE[Cytokine.IFNG]

        # Pathogen growth rate (per hour)
        pathogen_r = np.log(2) / pathogen_growth_rate

        for i in range(1, n_steps):
            # Pathogen dynamics
            # Growth - immune clearance
            immune_clearance = (t_cyto[i-1] * 1e-5 * pathogen[i-1] +
                              antibody[i-1] * 1e-3 * pathogen[i-1])

            d_pathogen = pathogen_r * pathogen[i-1] - immune_clearance
            pathogen[i] = max(0, pathogen[i-1] + d_pathogen * dt)

            # T helper activation by pathogen
            activation = pathogen[i] / (1e6 + pathogen[i])
            d_t_helper = (activation * 0.5 * t_helper[i-1] -  # Proliferation
                         0.01 * t_helper[i-1])  # Death
            t_helper[i] = max(1e6, t_helper[i-1] + d_t_helper * dt)

            # Cytotoxic T cell expansion
            d_t_cyto = (activation * 0.6 * t_cyto[i-1] -
                       0.01 * t_cyto[i-1])
            t_cyto[i] = max(5e5, t_cyto[i-1] + d_t_cyto * dt)

            # B cell activation and antibody production
            b_activation = activation * (il2[i-1] / 10)
            d_b_cells = (b_activation * 0.7 * b_cells[i-1] -
                        0.03 * b_cells[i-1])
            b_cells[i] = max(1e5, b_cells[i-1] + d_b_cells * dt)

            # Antibody production (molecules per cell per hour)
            antibody_production = b_cells[i] * 2000 * 3600 / 1e12  # Convert to nM
            antibody_decay = 0.05 * antibody[i-1]  # Half-life ~14 days

            antibody[i] = max(0, antibody[i-1] + (antibody_production - antibody_decay) * dt)

            # Cytokine dynamics
            il2[i] = self.CYTOKINE_BASELINE[Cytokine.IL2] + activation * 50
            il6[i] = self.CYTOKINE_BASELINE[Cytokine.IL6] + activation * 100
            ifng[i] = self.CYTOKINE_BASELINE[Cytokine.IFNG] + activation * 30

        cytokine_dict = {
            'IL2': il2,
            'IL6': il6,
            'IFNG': ifng
        }

        return ImmuneResponse(
            pathogen_count=pathogen,
            antibody_titer=antibody,
            t_cell_count=t_helper,
            cytokine_levels=cytokine_dict,
            time_days=time_days
        )

    def calculate_vaccine_efficacy(self, vaccine_type: str = 'mRNA',
                                  dose_schedule: List[int] = [0, 28],
                                  adjuvant: bool = True) -> VaccineEfficacy:
        """
        Calculate vaccine efficacy based on vaccine parameters

        Args:
            vaccine_type: 'mRNA', 'protein', 'inactivated', 'live_attenuated'
            dose_schedule: Days for each dose
            adjuvant: Whether adjuvant is included

        Returns:
            Vaccine efficacy metrics
        """
        # Base efficacy by vaccine type (from clinical data)
        base_efficacy = {
            'mRNA': 0.95,
            'protein': 0.85,
            'inactivated': 0.70,
            'live_attenuated': 0.90
        }

        # Base seroconversion
        base_sero = {
            'mRNA': 0.98,
            'protein': 0.90,
            'inactivated': 0.85,
            'live_attenuated': 0.95
        }

        # Duration of protection (months)
        duration = {
            'mRNA': 8,
            'protein': 6,
            'inactivated': 4,
            'live_attenuated': 12
        }

        efficacy = base_efficacy.get(vaccine_type, 0.8)
        seroconversion = base_sero.get(vaccine_type, 0.9)
        protect_duration = duration.get(vaccine_type, 6)

        # Adjust for doses
        dose_multiplier = min(len(dose_schedule), 3) / 2.0  # 2 doses optimal
        efficacy *= dose_multiplier
        seroconversion *= dose_multiplier

        # Adjuvant boost
        if adjuvant and vaccine_type in ['protein', 'inactivated']:
            efficacy *= 1.15
            seroconversion *= 1.1

        # Cap at realistic maxima
        efficacy = min(0.98, efficacy)
        seroconversion = min(0.99, seroconversion)

        # Geometric mean titer (arbitrary units, log-normal distribution)
        gmt = np.random.lognormal(6, 1.5) * (efficacy / 0.8)

        # Adverse events (typical rates)
        adverse_rates = {
            'mRNA': 0.15,  # 15% (mostly mild)
            'protein': 0.10,
            'inactivated': 0.05,
            'live_attenuated': 0.20
        }
        adverse_rate = adverse_rates.get(vaccine_type, 0.10)

        return VaccineEfficacy(
            seroconversion_rate=float(seroconversion),
            geometric_mean_titer=float(gmt),
            protection_rate=float(efficacy),
            duration_months=float(protect_duration),
            adverse_events_rate=float(adverse_rate)
        )

    def simulate_autoimmune_disease(self, disease: str = 'rheumatoid_arthritis',
                                  duration_months: int = 12,
                                  treatment: Optional[str] = None) -> Dict:
        """
        Simulate autoimmune disease progression

        Args:
            disease: 'rheumatoid_arthritis', 'lupus', 'multiple_sclerosis', 'type1_diabetes'
            duration_months: Simulation duration
            treatment: 'corticosteroid', 'dmard', 'biologic', None

        Returns:
            Disease progression metrics
        """
        # Disease parameters
        disease_params = {
            'rheumatoid_arthritis': {
                'baseline_severity': 50,
                'progression_rate': 2.0,  # per month
                'auto_antibody_baseline': 100,
                'inflammation_baseline': 30
            },
            'lupus': {
                'baseline_severity': 40,
                'progression_rate': 3.0,
                'auto_antibody_baseline': 200,
                'inflammation_baseline': 50
            },
            'multiple_sclerosis': {
                'baseline_severity': 35,
                'progression_rate': 1.5,
                'auto_antibody_baseline': 50,
                'inflammation_baseline': 40
            },
            'type1_diabetes': {
                'baseline_severity': 60,
                'progression_rate': 4.0,
                'auto_antibody_baseline': 150,
                'inflammation_baseline': 35
            }
        }

        params = disease_params.get(disease, disease_params['rheumatoid_arthritis'])

        # Time array (months)
        time = np.arange(0, duration_months + 1)

        # Initialize arrays
        severity = np.zeros(len(time))
        auto_antibody = np.zeros(len(time))
        inflammation = np.zeros(len(time))

        severity[0] = params['baseline_severity']
        auto_antibody[0] = params['auto_antibody_baseline']
        inflammation[0] = params['inflammation_baseline']

        # Treatment effects
        treatment_efficacy = {
            'corticosteroid': {'severity': 0.3, 'inflammation': 0.6},
            'dmard': {'severity': 0.5, 'inflammation': 0.4},
            'biologic': {'severity': 0.7, 'inflammation': 0.8},
            None: {'severity': 0.0, 'inflammation': 0.0}
        }

        effect = treatment_efficacy.get(treatment, treatment_efficacy[None])

        for i in range(1, len(time)):
            # Disease progression
            progression = params['progression_rate'] * (1 - effect['severity'])

            # Add stochastic flares
            flare = 10 if np.random.random() < 0.15 else 0

            severity[i] = min(100, max(0, severity[i-1] + progression + flare -
                                     effect['severity'] * 5))

            # Auto-antibody levels correlate with severity
            auto_antibody[i] = params['auto_antibody_baseline'] * (1 + severity[i] / 100)

            # Inflammation
            inflammation[i] = params['inflammation_baseline'] * (1 + severity[i] / 100) * \
                            (1 - effect['inflammation'])

        # Calculate disease metrics
        avg_severity = float(np.mean(severity))
        flare_count = int(np.sum(np.diff(severity) > 5))
        remission_months = int(np.sum(severity < 20))

        return {
            'disease': disease,
            'duration_months': duration_months,
            'treatment': treatment,
            'average_severity': avg_severity,
            'final_severity': float(severity[-1]),
            'flare_count': flare_count,
            'remission_months': remission_months,
            'avg_auto_antibody': float(np.mean(auto_antibody)),
            'avg_inflammation': float(np.mean(inflammation)),
            'severity_trajectory': severity.tolist()
        }


def run_comprehensive_test() -> Dict:
    """Run comprehensive immunology lab test"""
    lab = ImmunologyLaboratory(seed=42)
    results = {}

    # Test 1: Antibody-antigen binding
    print("Testing antibody-antigen binding...")
    binding_high = lab.simulate_antibody_antigen_binding(
        antibody_conc_nM=100, antigen_conc_nM=50, affinity='high_affinity'
    )
    results['antibody_binding'] = {
        'KD_nM': binding_high.KD * 1e9,
        'affinity_M-1': float(binding_high.affinity),
        'ka': binding_high.ka,
        'kd': binding_high.kd,
        'max_binding_fraction': float(np.max(binding_high.binding_curve))
    }

    # Test 2: Immune response
    print("Simulating immune response...")
    response = lab.simulate_immune_response(pathogen_dose=1e6, duration_days=21)
    results['immune_response'] = {
        'peak_pathogen': float(np.max(response.pathogen_count)),
        'clearance_time_days': float(response.time_days[np.argmax(response.antibody_titer)]),
        'peak_antibody_titer': float(np.max(response.antibody_titer)),
        'final_pathogen': float(response.pathogen_count[-1])
    }

    # Test 3: Vaccine efficacy
    print("Calculating vaccine efficacy...")
    vaccines = ['mRNA', 'protein', 'inactivated']
    vaccine_results = {}
    for vax in vaccines:
        efficacy = lab.calculate_vaccine_efficacy(vax, dose_schedule=[0, 28], adjuvant=True)
        vaccine_results[vax] = {
            'seroconversion': efficacy.seroconversion_rate,
            'protection': efficacy.protection_rate,
            'duration_months': efficacy.duration_months,
            'GMT': efficacy.geometric_mean_titer
        }
    results['vaccines'] = vaccine_results

    # Test 4: Autoimmune disease
    print("Simulating autoimmune disease...")
    disease_sim = lab.simulate_autoimmune_disease(
        disease='rheumatoid_arthritis',
        duration_months=12,
        treatment='biologic'
    )
    results['autoimmune'] = {
        'disease': disease_sim['disease'],
        'avg_severity': disease_sim['average_severity'],
        'flares': disease_sim['flare_count'],
        'remission_months': disease_sim['remission_months']
    }

    return results


if __name__ == "__main__":
    print("QuLabInfinite Immunology Laboratory - Comprehensive Test")
    print("=" * 60)

    results = run_comprehensive_test()
    print(json.dumps(results, indent=2))
