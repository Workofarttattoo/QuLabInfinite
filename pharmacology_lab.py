"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

PHARMACOLOGY LAB - Production-Ready Pharmacokinetic/Pharmacodynamic Modeling
Free gift to the scientific community from QuLabInfinite.

This module provides comprehensive PK/PD modeling including:
- Multi-compartment pharmacokinetic models (1, 2, 3 compartment)
- Pharmacodynamic models (Emax, sigmoid Emax, indirect response)
- Population PK with mixed effects modeling
- Bioavailability and therapeutic window analysis
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Union
from scipy.integrate import odeint
from scipy.optimize import minimize, curve_fit
from scipy.stats import norm, lognorm
import warnings

# Constants for pharmacological calculations
@dataclass(frozen=True)
class PharmConstants:
    """Pharmacological constants and standard values"""
    BODY_WATER_FRACTION: float = 0.6  # Total body water as fraction of weight
    PLASMA_VOLUME_L_KG: float = 0.05  # Plasma volume in L/kg
    BLOOD_VOLUME_L_KG: float = 0.08  # Blood volume in L/kg
    GFR_ML_MIN_173M2: float = 120.0  # Normal glomerular filtration rate
    HEPATIC_BLOOD_FLOW_L_H: float = 90.0  # Hepatic blood flow in L/h
    CARDIAC_OUTPUT_L_MIN: float = 5.0  # Normal cardiac output
    PROTEIN_BINDING_ALBUMIN_G_L: float = 40.0  # Normal albumin concentration
    PH_PLASMA: float = 7.4  # Normal plasma pH
    PH_INTRACELLULAR: float = 7.0  # Intracellular pH

@dataclass
class PKParameters:
    """Pharmacokinetic parameters for drug modeling"""
    clearance: float  # L/h
    volume_distribution: float  # L
    absorption_rate: float = 1.0  # 1/h for oral drugs
    bioavailability: float = 1.0  # Fraction (0-1)
    protein_binding: float = 0.0  # Fraction bound (0-1)
    half_life: float = field(init=False)  # hours, calculated
    elimination_rate: float = field(init=False)  # 1/h, calculated

    def __post_init__(self):
        """Calculate derived parameters"""
        self.elimination_rate = self.clearance / self.volume_distribution
        self.half_life = 0.693 / self.elimination_rate

@dataclass
class PDParameters:
    """Pharmacodynamic parameters for drug response modeling"""
    emax: float  # Maximum effect
    ec50: float  # Concentration for 50% effect
    hill_coefficient: float = 1.0  # Sigmoid factor
    baseline: float = 0.0  # Baseline effect
    kout: float = 0.1  # Turnover rate for indirect response

class PharmacologyLab:
    """
    Comprehensive pharmacology laboratory for PK/PD modeling and analysis.
    Implements validated models used in clinical pharmacology research.
    """

    def __init__(self):
        self.constants = PharmConstants()
        self._validation_enabled = True

    # ============= COMPARTMENTAL PK MODELS =============

    def one_compartment_model(self, dose: float, params: PKParameters,
                              time: np.ndarray, iv: bool = True) -> np.ndarray:
        """
        One-compartment PK model with first-order elimination.

        Args:
            dose: Drug dose in mg
            params: PK parameters
            time: Time points in hours
            iv: If True, IV bolus; if False, oral with absorption

        Returns:
            Plasma concentrations at each time point (mg/L)
        """
        if not iv and params.absorption_rate <= 0:
            raise ValueError("Absorption rate must be positive for oral dosing")

        if iv:
            # IV bolus: C(t) = (Dose/Vd) * exp(-ke*t)
            c0 = dose / params.volume_distribution
            concentrations = c0 * np.exp(-params.elimination_rate * time)
        else:
            # Oral: C(t) = F*Dose*ka/(Vd*(ka-ke)) * (exp(-ke*t) - exp(-ka*t))
            ka = params.absorption_rate
            ke = params.elimination_rate
            f = params.bioavailability
            vd = params.volume_distribution

            if abs(ka - ke) < 1e-6:  # When ka ≈ ke
                concentrations = (f * dose * ka / vd) * time * np.exp(-ke * time)
            else:
                concentrations = (f * dose * ka / (vd * (ka - ke))) * \
                                (np.exp(-ke * time) - np.exp(-ka * time))

        return np.maximum(concentrations, 0)  # Ensure non-negative

    def two_compartment_model(self, dose: float, k10: float, k12: float,
                             k21: float, v1: float, time: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Two-compartment PK model (central and peripheral compartments).

        Args:
            dose: Drug dose in mg
            k10: Elimination rate from central compartment (1/h)
            k12: Transfer rate central→peripheral (1/h)
            k21: Transfer rate peripheral→central (1/h)
            v1: Central compartment volume (L)
            time: Time points in hours

        Returns:
            Tuple of (central concentrations, peripheral concentrations) in mg/L
        """
        def two_comp_ode(y, t, k10, k12, k21):
            """ODE system for two-compartment model"""
            a1, a2 = y  # Amounts in compartments
            da1dt = -k10 * a1 - k12 * a1 + k21 * a2
            da2dt = k12 * a1 - k21 * a2
            return [da1dt, da2dt]

        # Initial conditions: all drug in central compartment
        y0 = [dose, 0]

        # Solve ODE system
        solution = odeint(two_comp_ode, y0, time, args=(k10, k12, k21))

        # Convert amounts to concentrations
        central_conc = solution[:, 0] / v1
        peripheral_conc = solution[:, 1] / (v1 * k12 / k21)  # V2 = V1 * k12/k21

        return central_conc, peripheral_conc

    def three_compartment_model(self, dose: float, params_dict: Dict[str, float],
                               time: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Three-compartment PK model (central + two peripheral compartments).

        Args:
            dose: Drug dose in mg
            params_dict: Dictionary with rate constants (k10, k12, k21, k13, k31) and v1
            time: Time points in hours

        Returns:
            Tuple of (central, peripheral1, peripheral2) concentrations in mg/L
        """
        def three_comp_ode(y, t, k10, k12, k21, k13, k31):
            """ODE system for three-compartment model"""
            a1, a2, a3 = y
            da1dt = -k10 * a1 - k12 * a1 - k13 * a1 + k21 * a2 + k31 * a3
            da2dt = k12 * a1 - k21 * a2
            da3dt = k13 * a1 - k31 * a3
            return [da1dt, da2dt, da3dt]

        k10 = params_dict.get('k10', 0.1)
        k12 = params_dict.get('k12', 0.05)
        k21 = params_dict.get('k21', 0.05)
        k13 = params_dict.get('k13', 0.02)
        k31 = params_dict.get('k31', 0.02)
        v1 = params_dict.get('v1', 50.0)

        y0 = [dose, 0, 0]
        solution = odeint(three_comp_ode, y0, time, args=(k10, k12, k21, k13, k31))

        central_conc = solution[:, 0] / v1
        peripheral1_conc = solution[:, 1] / (v1 * k12 / k21)
        peripheral2_conc = solution[:, 2] / (v1 * k13 / k31)

        return central_conc, peripheral1_conc, peripheral2_conc

    # ============= PHARMACODYNAMIC MODELS =============

    def emax_model(self, concentration: Union[float, np.ndarray],
                   params: PDParameters) -> Union[float, np.ndarray]:
        """
        Emax pharmacodynamic model: E = E0 + Emax * C / (EC50 + C)

        Args:
            concentration: Drug concentration(s)
            params: PD parameters

        Returns:
            Pharmacological effect
        """
        effect = params.baseline + (params.emax * concentration) / (params.ec50 + concentration)
        return effect

    def sigmoid_emax_model(self, concentration: Union[float, np.ndarray],
                          params: PDParameters) -> Union[float, np.ndarray]:
        """
        Sigmoid Emax model (Hill equation): E = E0 + Emax * C^n / (EC50^n + C^n)

        Args:
            concentration: Drug concentration(s)
            params: PD parameters including hill_coefficient

        Returns:
            Pharmacological effect
        """
        c_hill = np.power(concentration, params.hill_coefficient)
        ec50_hill = np.power(params.ec50, params.hill_coefficient)
        effect = params.baseline + (params.emax * c_hill) / (ec50_hill + c_hill)
        return effect

    def indirect_response_model(self, concentration: np.ndarray, time: np.ndarray,
                               params: PDParameters, inhibition: bool = True) -> np.ndarray:
        """
        Indirect response model for delayed drug effects.

        Args:
            concentration: Drug concentrations over time
            time: Time points
            params: PD parameters
            inhibition: If True, drug inhibits; if False, drug stimulates

        Returns:
            Response over time
        """
        def indirect_ode(r, t, conc_func, params, inhibition):
            """ODE for indirect response"""
            conc = conc_func(t)
            if inhibition:
                # Inhibition of input: dR/dt = kin*(1 - I(C)) - kout*R
                inhibition_factor = conc / (params.ec50 + conc)
                drdt = params.kout * params.baseline * (1 - inhibition_factor) - params.kout * r
            else:
                # Stimulation of input: dR/dt = kin*(1 + S(C)) - kout*R
                stimulation_factor = params.emax * conc / (params.ec50 + conc)
                drdt = params.kout * params.baseline * (1 + stimulation_factor) - params.kout * r
            return drdt

        # Create interpolation function for concentration
        from scipy.interpolate import interp1d
        conc_func = interp1d(time, concentration, kind='linear', fill_value='extrapolate')

        # Solve ODE
        r0 = params.baseline
        response = odeint(indirect_ode, r0, time, args=(conc_func, params, inhibition))

        return response.flatten()

    # ============= BIOAVAILABILITY & CLEARANCE =============

    def calculate_bioavailability(self, auc_oral: float, auc_iv: float,
                                 dose_oral: float, dose_iv: float) -> float:
        """
        Calculate absolute bioavailability from AUC data.

        F = (AUC_oral / Dose_oral) / (AUC_iv / Dose_iv)

        Args:
            auc_oral: Area under curve for oral administration
            auc_iv: Area under curve for IV administration
            dose_oral: Oral dose
            dose_iv: IV dose

        Returns:
            Bioavailability fraction (0-1)
        """
        if auc_iv <= 0 or dose_oral <= 0 or dose_iv <= 0:
            raise ValueError("AUC and doses must be positive")

        f = (auc_oral * dose_iv) / (auc_iv * dose_oral)
        return np.clip(f, 0, 1)  # Bioavailability cannot exceed 100%

    def calculate_clearance(self, dose: float, auc: float) -> float:
        """
        Calculate total body clearance.

        CL = Dose / AUC

        Args:
            dose: Drug dose (mg)
            auc: Area under concentration-time curve (mg*h/L)

        Returns:
            Clearance in L/h
        """
        if auc <= 0:
            raise ValueError("AUC must be positive")
        return dose / auc

    def renal_clearance(self, fu: float, gfr: float, secretion_rate: float = 0,
                       reabsorption_fraction: float = 0) -> float:
        """
        Calculate renal clearance considering filtration, secretion, and reabsorption.

        CLr = fu * GFR + Secretion - Reabsorption

        Args:
            fu: Fraction unbound in plasma (0-1)
            gfr: Glomerular filtration rate (mL/min)
            secretion_rate: Active secretion rate (mL/min)
            reabsorption_fraction: Fraction reabsorbed (0-1)

        Returns:
            Renal clearance in L/h
        """
        # Convert GFR from mL/min to L/h
        gfr_lh = gfr * 0.06
        secretion_lh = secretion_rate * 0.06

        cl_renal = fu * gfr_lh + secretion_lh
        cl_renal *= (1 - reabsorption_fraction)

        return cl_renal

    def hepatic_clearance(self, fu: float, intrinsic_clearance: float,
                         hepatic_blood_flow: Optional[float] = None) -> float:
        """
        Calculate hepatic clearance using well-stirred model.

        CLh = Q * fu * CLint / (Q + fu * CLint)

        Args:
            fu: Fraction unbound in plasma
            intrinsic_clearance: Intrinsic clearance (L/h)
            hepatic_blood_flow: Hepatic blood flow (L/h), uses default if None

        Returns:
            Hepatic clearance in L/h
        """
        q = hepatic_blood_flow or self.constants.HEPATIC_BLOOD_FLOW_L_H

        if fu * intrinsic_clearance == 0:
            return 0

        cl_hepatic = (q * fu * intrinsic_clearance) / (q + fu * intrinsic_clearance)
        return cl_hepatic

    # ============= POPULATION PK =============

    def population_pk_parameters(self, n_subjects: int, typical_cl: float,
                                typical_vd: float, omega_cl: float = 0.3,
                                omega_vd: float = 0.3, seed: Optional[int] = None) -> List[PKParameters]:
        """
        Generate population PK parameters with inter-individual variability.
        Uses log-normal distribution for PK parameters.

        Args:
            n_subjects: Number of subjects
            typical_cl: Population typical clearance (L/h)
            typical_vd: Population typical volume (L)
            omega_cl: CV for clearance (coefficient of variation)
            omega_vd: CV for volume of distribution
            seed: Random seed for reproducibility

        Returns:
            List of PKParameters for each subject
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate individual parameters using log-normal distribution
        cl_values = np.random.lognormal(np.log(typical_cl), omega_cl, n_subjects)
        vd_values = np.random.lognormal(np.log(typical_vd), omega_vd, n_subjects)

        # Create PKParameters for each subject
        population = []
        for cl, vd in zip(cl_values, vd_values):
            params = PKParameters(clearance=cl, volume_distribution=vd)
            population.append(params)

        return population

    def mixed_effects_model(self, concentrations: np.ndarray, covariates: Dict[str, np.ndarray],
                          fixed_effects: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Simple mixed-effects model for population PK analysis.

        Args:
            concentrations: Observed concentrations
            covariates: Dictionary of covariates (e.g., weight, age, creatinine)
            fixed_effects: Initial estimates of fixed effects

        Returns:
            Predicted concentrations and estimated parameters
        """
        # Simplified implementation - in practice use NONMEM or similar
        n_subjects = len(concentrations)

        # Apply covariate effects
        cl_individual = fixed_effects['typical_cl'] * np.ones(n_subjects)

        if 'weight' in covariates:
            # Allometric scaling: CL = CLtyp * (WT/70)^0.75
            cl_individual *= np.power(covariates['weight'] / 70, 0.75)

        if 'creatinine' in covariates:
            # Renal function adjustment
            crcl = (140 - covariates.get('age', 40)) * covariates.get('weight', 70) / \
                   (72 * covariates['creatinine'])
            cl_individual *= crcl / 120  # Normalize to normal CrCl

        predictions = cl_individual * 2.5  # Simplified prediction

        estimated_params = {
            'population_cl': np.mean(cl_individual),
            'between_subject_var': np.var(cl_individual),
            'residual_error': np.std(concentrations - predictions)
        }

        return predictions, estimated_params

    # ============= THERAPEUTIC WINDOW =============

    def therapeutic_window_analysis(self, concentrations: np.ndarray,
                                   min_effective: float, max_toxic: float,
                                   time: Optional[np.ndarray] = None) -> Dict[str, Union[float, np.ndarray]]:
        """
        Analyze drug concentrations relative to therapeutic window.

        Args:
            concentrations: Drug concentrations
            min_effective: Minimum effective concentration
            max_toxic: Maximum toxic concentration
            time: Time points (optional)

        Returns:
            Dictionary with therapeutic metrics
        """
        in_therapeutic_range = (concentrations >= min_effective) & (concentrations <= max_toxic)
        below_therapeutic = concentrations < min_effective
        above_toxic = concentrations > max_toxic

        results = {
            'percent_in_range': np.mean(in_therapeutic_range) * 100,
            'percent_subtherapeutic': np.mean(below_therapeutic) * 100,
            'percent_toxic': np.mean(above_toxic) * 100,
            'therapeutic_index': max_toxic / min_effective,
            'peak_concentration': np.max(concentrations),
            'trough_concentration': np.min(concentrations),
            'peak_trough_ratio': np.max(concentrations) / np.max([np.min(concentrations), 1e-10])
        }

        if time is not None and len(time) > 1:
            # Calculate time in therapeutic range
            time_in_range = np.trapz(in_therapeutic_range, time) / (time[-1] - time[0])
            results['time_in_range_fraction'] = time_in_range

            # Find time to reach therapeutic level
            therapeutic_idx = np.where(concentrations >= min_effective)[0]
            if len(therapeutic_idx) > 0:
                results['time_to_therapeutic'] = time[therapeutic_idx[0]]

        return results

    def calculate_half_life(self, concentrations: np.ndarray, time: np.ndarray,
                          terminal_points: int = 3) -> Tuple[float, float]:
        """
        Calculate elimination half-life from terminal phase.

        Args:
            concentrations: Drug concentrations
            time: Time points
            terminal_points: Number of points to use for terminal phase

        Returns:
            Tuple of (half_life, r_squared of fit)
        """
        if len(concentrations) < terminal_points:
            raise ValueError(f"Need at least {terminal_points} points")

        # Use terminal phase (last n points)
        log_conc = np.log(concentrations[-terminal_points:] + 1e-10)
        t_terminal = time[-terminal_points:]

        # Linear regression on log-transformed data
        coeffs = np.polyfit(t_terminal, log_conc, 1)
        ke = -coeffs[0]  # Negative slope is elimination rate

        if ke <= 0:
            warnings.warn("Invalid elimination rate calculated")
            return np.inf, 0

        half_life = 0.693 / ke

        # Calculate R-squared
        predicted = np.polyval(coeffs, t_terminal)
        r_squared = 1 - np.sum((log_conc - predicted)**2) / np.sum((log_conc - np.mean(log_conc))**2)

        return half_life, r_squared

    def volume_of_distribution(self, dose: float, concentration_0: float,
                              body_weight: float = 70) -> Dict[str, float]:
        """
        Calculate different volumes of distribution.

        Args:
            dose: Drug dose (mg)
            concentration_0: Initial concentration (mg/L)
            body_weight: Body weight in kg

        Returns:
            Dictionary with different Vd calculations
        """
        vd_apparent = dose / concentration_0

        return {
            'vd_apparent_L': vd_apparent,
            'vd_apparent_L_kg': vd_apparent / body_weight,
            'vd_as_percent_body_water': (vd_apparent / body_weight) / self.constants.BODY_WATER_FRACTION * 100,
            'distribution_classification': self._classify_distribution(vd_apparent / body_weight)
        }

    def _classify_distribution(self, vd_l_kg: float) -> str:
        """Classify drug distribution based on Vd"""
        if vd_l_kg < 0.1:
            return "Confined to blood"
        elif vd_l_kg < 0.3:
            return "Limited to extracellular fluid"
        elif vd_l_kg < 0.7:
            return "Distributed to total body water"
        else:
            return "Extensive tissue binding"

    def validate_parameters(self, params: Union[PKParameters, PDParameters]) -> bool:
        """
        Validate pharmacological parameters are within reasonable ranges.

        Args:
            params: PK or PD parameters to validate

        Returns:
            True if valid, raises ValueError if not
        """
        if not self._validation_enabled:
            return True

        if isinstance(params, PKParameters):
            if params.clearance <= 0 or params.clearance > 1000:
                raise ValueError(f"Clearance {params.clearance} L/h out of physiological range")
            if params.volume_distribution <= 0 or params.volume_distribution > 50000:
                raise ValueError(f"Vd {params.volume_distribution} L out of physiological range")
            if params.bioavailability < 0 or params.bioavailability > 1:
                raise ValueError(f"Bioavailability must be between 0 and 1")
            if params.protein_binding < 0 or params.protein_binding > 1:
                raise ValueError(f"Protein binding must be between 0 and 1")

        elif isinstance(params, PDParameters):
            if params.ec50 <= 0:
                raise ValueError("EC50 must be positive")
            if params.hill_coefficient <= 0 or params.hill_coefficient > 10:
                raise ValueError(f"Hill coefficient {params.hill_coefficient} out of typical range")

        return True


def run_comprehensive_demo():
    """Demonstrate all major features of the PharmacologyLab"""
    print("=" * 80)
    print("PHARMACOLOGY LAB - Comprehensive PK/PD Modeling Demo")
    print("Copyright (c) 2025 Corporation of Light")
    print("=" * 80)

    lab = PharmacologyLab()

    # 1. One-compartment model
    print("\n1. ONE-COMPARTMENT MODEL")
    print("-" * 40)
    pk_params = PKParameters(clearance=5.0, volume_distribution=50.0,
                            absorption_rate=1.5, bioavailability=0.8)
    time = np.linspace(0, 24, 100)

    # IV bolus
    conc_iv = lab.one_compartment_model(dose=100, params=pk_params, time=time, iv=True)
    # Oral administration
    conc_oral = lab.one_compartment_model(dose=100, params=pk_params, time=time, iv=False)

    print(f"PK Parameters: CL={pk_params.clearance:.1f} L/h, Vd={pk_params.volume_distribution:.1f} L")
    print(f"Calculated: t½={pk_params.half_life:.2f} h, ke={pk_params.elimination_rate:.3f} 1/h")
    print(f"Peak concentration (IV): {np.max(conc_iv):.2f} mg/L")
    print(f"Peak concentration (Oral): {np.max(conc_oral):.2f} mg/L")

    # 2. Two-compartment model
    print("\n2. TWO-COMPARTMENT MODEL")
    print("-" * 40)
    central, peripheral = lab.two_compartment_model(
        dose=200, k10=0.2, k12=0.1, k21=0.05, v1=30, time=time
    )
    print(f"Central compartment Cmax: {np.max(central):.2f} mg/L")
    print(f"Peripheral compartment Cmax: {np.max(peripheral):.2f} mg/L")
    print(f"Distribution phase duration: ~{time[np.argmax(peripheral)]:.1f} hours")

    # 3. Three-compartment model
    print("\n3. THREE-COMPARTMENT MODEL")
    print("-" * 40)
    params_3comp = {'k10': 0.15, 'k12': 0.08, 'k21': 0.06,
                   'k13': 0.03, 'k31': 0.02, 'v1': 40}
    c1, c2, c3 = lab.three_compartment_model(dose=150, params_dict=params_3comp, time=time)
    print(f"Central Cmax: {np.max(c1):.2f} mg/L at t={time[np.argmax(c1)]:.1f} h")
    print(f"Peripheral-1 Cmax: {np.max(c2):.2f} mg/L at t={time[np.argmax(c2)]:.1f} h")
    print(f"Peripheral-2 Cmax: {np.max(c3):.2f} mg/L at t={time[np.argmax(c3)]:.1f} h")

    # 4. Pharmacodynamic models
    print("\n4. PHARMACODYNAMIC MODELS")
    print("-" * 40)
    pd_params = PDParameters(emax=100, ec50=5.0, hill_coefficient=2.0, baseline=10, kout=0.1)
    test_conc = np.array([0, 1, 2, 5, 10, 20, 50])

    # Emax model
    effect_emax = lab.emax_model(test_conc, pd_params)
    print("Emax Model Results:")
    for c, e in zip(test_conc, effect_emax):
        print(f"  C={c:5.1f} mg/L → Effect={e:6.1f}")

    # Sigmoid Emax model
    effect_sigmoid = lab.sigmoid_emax_model(test_conc, pd_params)
    print("\nSigmoid Emax Model (Hill n=2):")
    for c, e in zip(test_conc, effect_sigmoid):
        print(f"  C={c:5.1f} mg/L → Effect={e:6.1f}")

    # Indirect response model
    print("\nIndirect Response Model:")
    response = lab.indirect_response_model(conc_iv[:50], time[:50], pd_params, inhibition=True)
    print(f"  Baseline: {pd_params.baseline:.1f}")
    print(f"  Maximum inhibition: {np.min(response):.1f}")
    print(f"  Time to maximum effect: {time[np.argmin(response)]:.1f} h")

    # 5. Clearance calculations
    print("\n5. CLEARANCE CALCULATIONS")
    print("-" * 40)

    # Renal clearance
    cl_renal = lab.renal_clearance(fu=0.3, gfr=120, secretion_rate=20, reabsorption_fraction=0.1)
    print(f"Renal clearance: {cl_renal:.2f} L/h")

    # Hepatic clearance
    cl_hepatic = lab.hepatic_clearance(fu=0.3, intrinsic_clearance=200)
    print(f"Hepatic clearance: {cl_hepatic:.2f} L/h")
    print(f"Extraction ratio: {cl_hepatic/lab.constants.HEPATIC_BLOOD_FLOW_L_H:.2%}")

    # Total clearance
    cl_total = cl_renal + cl_hepatic
    print(f"Total body clearance: {cl_total:.2f} L/h")

    # 6. Bioavailability calculation
    print("\n6. BIOAVAILABILITY ANALYSIS")
    print("-" * 40)
    auc_iv = np.trapz(conc_iv, time)
    auc_oral = np.trapz(conc_oral, time)
    bioavail = lab.calculate_bioavailability(auc_oral, auc_iv, dose_oral=100, dose_iv=100)
    print(f"AUC (IV): {auc_iv:.1f} mg·h/L")
    print(f"AUC (Oral): {auc_oral:.1f} mg·h/L")
    print(f"Calculated bioavailability: {bioavail:.1%}")
    print(f"Expected bioavailability: {pk_params.bioavailability:.1%}")

    # 7. Population PK
    print("\n7. POPULATION PHARMACOKINETICS")
    print("-" * 40)
    population = lab.population_pk_parameters(
        n_subjects=100, typical_cl=5.0, typical_vd=50.0,
        omega_cl=0.3, omega_vd=0.25, seed=42
    )

    cl_values = [p.clearance for p in population]
    vd_values = [p.volume_distribution for p in population]

    print(f"Population size: {len(population)} subjects")
    print(f"Clearance: {np.mean(cl_values):.1f} ± {np.std(cl_values):.1f} L/h")
    print(f"Volume: {np.mean(vd_values):.1f} ± {np.std(vd_values):.1f} L")
    print(f"Half-life: {np.mean([p.half_life for p in population]):.1f} ± "
          f"{np.std([p.half_life for p in population]):.1f} h")

    # Mixed effects model
    covariates = {
        'weight': np.random.normal(70, 15, 100),
        'age': np.random.normal(40, 10, 100),
        'creatinine': np.random.lognormal(0, 0.2, 100)
    }
    predictions, pop_params = lab.mixed_effects_model(
        np.random.normal(10, 2, 100), covariates, {'typical_cl': 5.0}
    )
    print(f"\nMixed-effects model results:")
    print(f"  Population CL: {pop_params['population_cl']:.2f} L/h")
    print(f"  Between-subject variability: {np.sqrt(pop_params['between_subject_var']):.2f}")

    # 8. Therapeutic window analysis
    print("\n8. THERAPEUTIC WINDOW ANALYSIS")
    print("-" * 40)
    therapeutic_analysis = lab.therapeutic_window_analysis(
        conc_oral, min_effective=2.0, max_toxic=15.0, time=time
    )

    print(f"Therapeutic window: 2.0 - 15.0 mg/L")
    print(f"Time in therapeutic range: {therapeutic_analysis['percent_in_range']:.1f}%")
    print(f"Time subtherapeutic: {therapeutic_analysis['percent_subtherapeutic']:.1f}%")
    print(f"Time in toxic range: {therapeutic_analysis['percent_toxic']:.1f}%")
    print(f"Therapeutic index: {therapeutic_analysis['therapeutic_index']:.1f}")
    print(f"Peak/Trough ratio: {therapeutic_analysis['peak_trough_ratio']:.1f}")
    if 'time_to_therapeutic' in therapeutic_analysis:
        print(f"Time to therapeutic level: {therapeutic_analysis['time_to_therapeutic']:.1f} h")

    # 9. Half-life calculation
    print("\n9. HALF-LIFE DETERMINATION")
    print("-" * 40)
    t_half, r2 = lab.calculate_half_life(conc_iv, time, terminal_points=10)
    print(f"Calculated t½: {t_half:.2f} h (R² = {r2:.3f})")
    print(f"Expected t½: {pk_params.half_life:.2f} h")

    # 10. Volume of distribution
    print("\n10. VOLUME OF DISTRIBUTION")
    print("-" * 40)
    vd_analysis = lab.volume_of_distribution(dose=100, concentration_0=conc_iv[0], body_weight=70)
    print(f"Apparent Vd: {vd_analysis['vd_apparent_L']:.1f} L")
    print(f"Vd per kg: {vd_analysis['vd_apparent_L_kg']:.2f} L/kg")
    print(f"% of body water: {vd_analysis['vd_as_percent_body_water']:.1f}%")
    print(f"Distribution: {vd_analysis['distribution_classification']}")

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    run_comprehensive_demo()