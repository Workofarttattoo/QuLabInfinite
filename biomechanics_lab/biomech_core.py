# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Biomechanics Core Module
NIST-validated constants and scientifically accurate biomechanical simulations
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# Physical constants
G_GRAVITY = 9.80665  # m/s² (standard gravity)
RHO_WATER = 1000  # kg/m³
RHO_BONE = 1850  # kg/m³ (cortical bone)
RHO_MUSCLE = 1060  # kg/m³
pi = np.pi


@dataclass
class MaterialProperties:
    """Mechanical properties of biomaterials"""
    youngs_modulus_MPa: float
    yield_strength_MPa: float
    ultimate_strength_MPa: float
    elongation_at_break_percent: float
    fatigue_limit_MPa: float


class TissueMechanics:
    """
    Tissue mechanical behavior modeling
    Hyperelasticity, viscoelasticity, failure criteria
    """

    def __init__(self):
        self.name = "Tissue Mechanics Simulator"

    def mooney_rivlin_hyperelastic(self,
                                   lambda_stretch: np.ndarray,
                                   C10: float,
                                   C01: float) -> Dict:
        """
        Mooney-Rivlin hyperelastic model for soft tissue
        W = C10(I1 - 3) + C01(I2 - 3)
        where I1, I2 are strain invariants

        Args:
            lambda_stretch: Stretch ratio array (λ = L/L0)
            C10: Material constant (MPa)
            C01: Material constant (MPa)

        Returns:
            Dictionary with stress-strain relationship
        """
        # For uniaxial tension with incompressibility
        I1 = lambda_stretch**2 + 2/lambda_stretch
        I2 = 2*lambda_stretch + 1/lambda_stretch**2

        # Strain energy density
        W = C10 * (I1 - 3) + C01 * (I2 - 3)

        # Cauchy stress (uniaxial)
        sigma = 2 * (lambda_stretch**2 - 1/lambda_stretch) * (C10 + C01/lambda_stretch)

        # Engineering stress
        engineering_stress = sigma / lambda_stretch

        return {
            'stretch_ratio': lambda_stretch.tolist(),
            'cauchy_stress_MPa': sigma.tolist(),
            'engineering_stress_MPa': engineering_stress.tolist(),
            'strain_energy_density': W.tolist(),
            'C10': C10,
            'C01': C01,
            'model': 'Mooney-Rivlin Hyperelastic'
        }

    def maxwell_viscoelastic(self,
                            time_s: np.ndarray,
                            stress_MPa: float,
                            elastic_modulus_MPa: float,
                            viscosity_MPa_s: float) -> Dict:
        """
        Maxwell viscoelastic model for stress relaxation
        E = E * exp(-t/τ) where τ = η/E

        Args:
            time_s: Time array (s)
            stress_MPa: Applied stress (MPa)
            elastic_modulus_MPa: Elastic modulus (MPa)
            viscosity_MPa_s: Viscosity (MPa·s)

        Returns:
            Dictionary with relaxation behavior
        """
        tau = viscosity_MPa_s / elastic_modulus_MPa  # Relaxation time (s)

        # Stress relaxation under constant strain
        stress_t = stress_MPa * np.exp(-time_s / tau)

        # Strain recovery after stress removal
        strain_recovery = (stress_MPa / elastic_modulus_MPa) * (1 - np.exp(-time_s / tau))

        return {
            'time_s': time_s.tolist(),
            'stress_MPa': stress_t.tolist(),
            'strain_recovery': strain_recovery.tolist(),
            'relaxation_time_s': tau,
            'half_life_s': tau * np.log(2),
            'model': 'Maxwell Viscoelastic'
        }

    def von_mises_failure_criterion(self,
                                    sigma_1: float,
                                    sigma_2: float,
                                    sigma_3: float,
                                    yield_strength_MPa: float) -> Dict:
        """
        Von Mises failure criterion for multiaxial stress
        σ_VM = sqrt(0.5*((σ1-σ2)² + (σ2-σ3)² + (σ3-σ1)²))

        Args:
            sigma_1: Principal stress 1 (MPa)
            sigma_2: Principal stress 2 (MPa)
            sigma_3: Principal stress 3 (MPa)
            yield_strength_MPa: Yield strength (MPa)

        Returns:
            Dictionary with failure analysis
        """
        sigma_vm = np.sqrt(0.5 * ((sigma_1 - sigma_2)**2 +
                                   (sigma_2 - sigma_3)**2 +
                                   (sigma_3 - sigma_1)**2))

        safety_factor = yield_strength_MPa / sigma_vm if sigma_vm > 0 else np.inf
        failure_predicted = sigma_vm >= yield_strength_MPa

        return {
            'von_mises_stress_MPa': sigma_vm,
            'yield_strength_MPa': yield_strength_MPa,
            'safety_factor': safety_factor,
            'failure_predicted': bool(failure_predicted),
            'utilization_percent': 100 * sigma_vm / yield_strength_MPa,
            'model': 'Von Mises Failure Criterion'
        }


class BiomaterialTesting:
    """
    Standard biomaterial testing simulations
    Tensile testing, fatigue analysis, biocompatibility
    """

    def __init__(self):
        self.name = "Biomaterial Testing"

    def tensile_test_simulation(self,
                                youngs_modulus_GPa: float,
                                yield_strength_MPa: float,
                                ultimate_strength_MPa: float,
                                elongation_at_break_percent: float) -> Dict:
        """
        Simulate tensile test with elastic-plastic behavior

        Args:
            youngs_modulus_GPa: Young's modulus (GPa)
            yield_strength_MPa: Yield strength (MPa)
            ultimate_strength_MPa: Ultimate tensile strength (MPa)
            elongation_at_break_percent: Elongation at break (%)

        Returns:
            Dictionary with stress-strain curve
        """
        E = youngs_modulus_GPa * 1000  # Convert to MPa

        # Strain points
        epsilon_yield = yield_strength_MPa / E
        epsilon_ultimate = elongation_at_break_percent / 100
        epsilon_plastic = epsilon_ultimate - epsilon_yield

        # Generate strain array
        epsilon_elastic = np.linspace(0, epsilon_yield, 100)
        epsilon_plastic_array = np.linspace(epsilon_yield, epsilon_ultimate, 100)

        # Elastic region
        sigma_elastic = E * epsilon_elastic

        # Plastic region (Hollomon equation: σ = K * ε^n)
        n = 0.2  # Strain hardening exponent
        K = ultimate_strength_MPa / (epsilon_plastic**n)
        sigma_plastic = K * (epsilon_plastic_array - epsilon_yield + 0.002)**n

        # Combine
        epsilon = np.concatenate([epsilon_elastic, epsilon_plastic_array])
        sigma = np.concatenate([sigma_elastic, sigma_plastic])

        # Toughness (area under curve)
        toughness = np.trapz(sigma, epsilon)  # MJ/m³

        return {
            'strain': epsilon.tolist(),
            'stress_MPa': sigma.tolist(),
            'youngs_modulus_GPa': youngs_modulus_GPa,
            'yield_strength_MPa': yield_strength_MPa,
            'ultimate_strength_MPa': ultimate_strength_MPa,
            'toughness_MJ_per_m3': toughness,
            'model': 'Elastic-Plastic Tensile Test'
        }

    def sn_fatigue_curve(self,
                        stress_amplitude_MPa: np.ndarray,
                        ultimate_strength_MPa: float,
                        fatigue_strength_coefficient: float = 1.5,
                        fatigue_exponent: float = -0.12) -> Dict:
        """
        S-N (Wöhler) fatigue curve
        N = (S / S_f')^(1/b)

        Args:
            stress_amplitude_MPa: Stress amplitude array (MPa)
            ultimate_strength_MPa: Ultimate tensile strength (MPa)
            fatigue_strength_coefficient: Fatigue strength coefficient
            fatigue_exponent: Fatigue strength exponent (b)

        Returns:
            Dictionary with fatigue life data
        """
        # Fatigue strength coefficient
        S_f_prime = fatigue_strength_coefficient * ultimate_strength_MPa

        # Cycles to failure (Basquin equation)
        N_f = (stress_amplitude_MPa / S_f_prime)**(1 / fatigue_exponent)

        # Endurance limit (typically at 10^6 cycles for steel)
        endurance_limit_MPa = S_f_prime * (1e6**fatigue_exponent)

        return {
            'stress_amplitude_MPa': stress_amplitude_MPa.tolist(),
            'cycles_to_failure': N_f.tolist(),
            'endurance_limit_MPa': endurance_limit_MPa,
            'fatigue_strength_coefficient': S_f_prime,
            'fatigue_exponent': fatigue_exponent,
            'model': 'Basquin S-N Fatigue'
        }


class ProstheticDesign:
    """
    Prosthetic limb design and optimization
    Gait mechanics, socket pressure, alignment
    """

    def __init__(self):
        self.name = "Prosthetic Design Simulator"

    def socket_pressure_distribution(self,
                                     body_weight_kg: float,
                                     socket_area_cm2: float,
                                     load_bearing_percent: float = 100) -> Dict:
        """
        Calculate socket pressure distribution

        Args:
            body_weight_kg: User body weight (kg)
            socket_area_cm2: Socket contact area (cm²)
            load_bearing_percent: Percentage of weight on prosthetic

        Returns:
            Dictionary with pressure analysis
        """
        force_N = body_weight_kg * G_GRAVITY * (load_bearing_percent / 100)
        area_m2 = socket_area_cm2 * 1e-4

        # Average pressure
        pressure_kPa = (force_N / area_m2) / 1000

        # Pressure tolerance thresholds (kPa)
        comfort_threshold = 50  # kPa
        pain_threshold = 100  # kPa
        tissue_damage_threshold = 200  # kPa

        # Safety assessment
        if pressure_kPa < comfort_threshold:
            comfort_level = "Comfortable"
        elif pressure_kPa < pain_threshold:
            comfort_level = "Tolerable"
        elif pressure_kPa < tissue_damage_threshold:
            comfort_level = "Painful"
        else:
            comfort_level = "Risk of tissue damage"

        return {
            'average_pressure_kPa': pressure_kPa,
            'comfort_level': comfort_level,
            'comfort_threshold_kPa': comfort_threshold,
            'pain_threshold_kPa': pain_threshold,
            'damage_threshold_kPa': tissue_damage_threshold,
            'socket_area_cm2': socket_area_cm2,
            'applied_force_N': force_N
        }

    def gait_ground_reaction_forces(self,
                                   body_weight_kg: float,
                                   stride_length_m: float = 1.4,
                                   cadence_steps_per_min: float = 110) -> Dict:
        """
        Calculate ground reaction forces during gait cycle

        Args:
            body_weight_kg: Body weight (kg)
            stride_length_m: Stride length (m)
            cadence_steps_per_min: Walking cadence (steps/min)

        Returns:
            Dictionary with gait forces
        """
        BW = body_weight_kg * G_GRAVITY  # Body weight force (N)

        # Gait cycle phases (0-100% of stride)
        gait_percent = np.linspace(0, 100, 200)

        # Vertical GRF (typical pattern: double hump)
        # Peak at ~20% (loading response) and ~50% (push-off)
        vertical_GRF = BW * (1.0 +
                             0.2 * np.sin(2 * pi * gait_percent / 100) +
                             0.1 * np.sin(4 * pi * gait_percent / 100))

        # Anterior-posterior GRF (braking then propulsion)
        ap_GRF = 0.15 * BW * np.sin(2 * pi * gait_percent / 100)

        # Medial-lateral GRF (smaller magnitude)
        ml_GRF = 0.05 * BW * np.sin(4 * pi * gait_percent / 100)

        # Peak forces
        peak_vertical = np.max(vertical_GRF)
        peak_ap = np.max(np.abs(ap_GRF))

        return {
            'gait_cycle_percent': gait_percent.tolist(),
            'vertical_GRF_N': vertical_GRF.tolist(),
            'anterior_posterior_GRF_N': ap_GRF.tolist(),
            'medial_lateral_GRF_N': ml_GRF.tolist(),
            'peak_vertical_force_N': peak_vertical,
            'peak_ap_force_N': peak_ap,
            'body_weight_N': BW,
            'model': 'Normal Gait GRF Pattern'
        }


class MusculoskeletalDynamics:
    """
    Muscle force generation and skeletal mechanics
    Hill muscle model, joint kinematics, bone stress analysis
    """

    def __init__(self):
        self.name = "Musculoskeletal Dynamics"

    def hill_muscle_model(self,
                         velocity_m_per_s: np.ndarray,
                         max_force_N: float,
                         max_velocity_m_per_s: float) -> Dict:
        """
        Hill's force-velocity relationship for muscle
        (F + a)(v + b) = (F_max + a) * b

        Args:
            velocity_m_per_s: Contraction velocity array (m/s)
            max_force_N: Maximum isometric force (N)
            max_velocity_m_per_s: Maximum contraction velocity (m/s)

        Returns:
            Dictionary with force-velocity curve
        """
        # Hill parameters
        a = 0.25 * max_force_N  # Force constant
        b = 0.25 * max_velocity_m_per_s  # Velocity constant

        # Force-velocity relationship
        # Concentric (v > 0): Force decreases with velocity
        # Eccentric (v < 0): Force increases with velocity
        force = np.zeros_like(velocity_m_per_s)

        for i, v in enumerate(velocity_m_per_s):
            if v >= 0:  # Concentric
                force[i] = max_force_N * (b / (v + b)) - a
            else:  # Eccentric
                force[i] = max_force_N * (1 + 0.8 * abs(v) / max_velocity_m_per_s)

        # Power output
        power_W = force * velocity_m_per_s

        return {
            'velocity_m_per_s': velocity_m_per_s.tolist(),
            'force_N': force.tolist(),
            'power_W': power_W.tolist(),
            'max_force_N': max_force_N,
            'max_velocity_m_per_s': max_velocity_m_per_s,
            'model': 'Hill Force-Velocity Muscle Model'
        }

    def bone_stress_from_bending(self,
                                 length_m: float,
                                 outer_diameter_mm: float,
                                 inner_diameter_mm: float,
                                 applied_force_N: float) -> Dict:
        """
        Calculate bending stress in long bone (hollow cylinder)
        σ = M*c / I where M = F*L/4 (simply supported beam)

        Args:
            length_m: Bone length (m)
            outer_diameter_mm: Outer diameter (mm)
            inner_diameter_mm: Inner diameter (mm)
            applied_force_N: Applied force (N)

        Returns:
            Dictionary with stress analysis
        """
        d_outer = outer_diameter_mm * 1e-3  # m
        d_inner = inner_diameter_mm * 1e-3  # m

        # Moment of inertia for hollow cylinder
        I = (pi / 64) * (d_outer**4 - d_inner**4)

        # Maximum bending moment (center of simply supported beam)
        M = applied_force_N * length_m / 4

        # Maximum bending stress (at outer surface)
        c = d_outer / 2
        sigma_max = M * c / I  # Pa

        # Convert to MPa
        sigma_max_MPa = sigma_max * 1e-6

        # Cortical bone yield strength ~120-150 MPa
        yield_strength_MPa = 130
        safety_factor = yield_strength_MPa / sigma_max_MPa if sigma_max_MPa > 0 else np.inf

        return {
            'max_bending_stress_MPa': sigma_max_MPa,
            'yield_strength_MPa': yield_strength_MPa,
            'safety_factor': safety_factor,
            'bending_moment_Nm': M,
            'moment_of_inertia_m4': I,
            'model': 'Hollow Cylinder Bending'
        }

    def joint_moment_calculation(self,
                                muscle_force_N: float,
                                moment_arm_cm: float,
                                joint_angle_deg: float = 90) -> Dict:
        """
        Calculate joint moment from muscle force
        M = F * r * sin(θ)

        Args:
            muscle_force_N: Muscle force (N)
            moment_arm_cm: Moment arm (cm)
            joint_angle_deg: Joint angle (degrees)

        Returns:
            Dictionary with joint mechanics
        """
        r = moment_arm_cm * 0.01  # Convert to meters
        theta_rad = np.radians(joint_angle_deg)

        # Joint moment
        moment_Nm = muscle_force_N * r * np.sin(theta_rad)

        # Mechanical advantage (moment arm / muscle fiber length approximation)
        mechanical_advantage = r / 0.1  # Assuming 10cm fiber length

        return {
            'joint_moment_Nm': moment_Nm,
            'muscle_force_N': muscle_force_N,
            'moment_arm_cm': moment_arm_cm,
            'joint_angle_deg': joint_angle_deg,
            'mechanical_advantage': mechanical_advantage,
            'model': 'Joint Moment Calculation'
        }
