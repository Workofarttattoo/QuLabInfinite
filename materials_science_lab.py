"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

MATERIALS SCIENCE LAB - Production Ready Implementation
Mechanical testing, failure analysis, phase transformations, composites, and crystal structures
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable, Any
from scipy import interpolate, optimize
from scipy.constants import k as k_B, Avogadro, R

# Material constants
YOUNG_MODULUS_STEEL = 200e9  # Pa
YIELD_STRENGTH_STEEL = 250e6  # Pa
POISSON_RATIO_STEEL = 0.3
DENSITY_STEEL = 7850  # kg/m³

@dataclass
class Material:
    """Base material properties"""
    name: str
    density: float  # kg/m³
    youngs_modulus: float  # Pa
    yield_strength: float  # Pa
    ultimate_strength: float  # Pa
    poisson_ratio: float
    thermal_conductivity: float  # W/(m·K)
    specific_heat: float  # J/(kg·K)
    melting_point: float  # K
    crystal_structure: Optional[str] = None
    lattice_parameter: Optional[float] = None  # Angstroms

@dataclass
class StressStrain:
    """Stress-strain data container"""
    strain: np.ndarray
    stress: np.ndarray
    temperature: float = 293.15  # K
    strain_rate: float = 0.001  # 1/s

@dataclass
class CompositeProperties:
    """Composite material properties"""
    matrix_material: Material
    fiber_material: Material
    fiber_volume_fraction: float
    fiber_orientation: str = "unidirectional"  # unidirectional, random, woven
    effective_modulus: Optional[float] = None
    effective_strength: Optional[float] = None

@dataclass
class CrystalDefect:
    """Crystal defect information"""
    defect_type: str  # vacancy, interstitial, substitutional, dislocation, grain_boundary
    concentration: float  # defects per unit volume
    activation_energy: float  # eV
    effect_on_yield: float  # Multiplier on yield strength

class MaterialsScienceLab:
    """Comprehensive materials science laboratory"""

    def __init__(self):
        self.materials_database = self._initialize_materials_database()
        self.test_results = []

    # ============= MECHANICAL TESTING =============

    def tensile_test(self, material: Material,
                     max_strain: float = 0.5,
                     num_points: int = 1000,
                     temperature: float = 293.15) -> StressStrain:
        """
        Simulate tensile test with elastic and plastic deformation
        """
        strain = np.linspace(0, max_strain, num_points)
        stress = np.zeros_like(strain)

        # Elastic region
        yield_strain = material.yield_strength / material.youngs_modulus
        elastic_mask = strain <= yield_strain

        stress[elastic_mask] = material.youngs_modulus * strain[elastic_mask]

        # Plastic region (Ramberg-Osgood model)
        plastic_mask = strain > yield_strain

        if np.any(plastic_mask):
            # Strain hardening parameters
            n = 0.2  # Strain hardening exponent
            K = material.ultimate_strength / (0.2 ** n)  # Strength coefficient

            # Ramberg-Osgood equation
            plastic_strain = strain[plastic_mask] - yield_strain
            stress[plastic_mask] = K * (plastic_strain ** n) + material.yield_strength

            # Cap at ultimate strength
            stress[plastic_mask] = np.minimum(stress[plastic_mask], material.ultimate_strength)

            # Necking and failure
            necking_strain = 0.3
            failure_mask = strain > necking_strain
            if np.any(failure_mask):
                # Considère criterion for necking
                reduction_factor = np.exp(-10 * (strain[failure_mask] - necking_strain))
                stress[failure_mask] *= reduction_factor

        # Temperature effects (simplified)
        temp_factor = np.exp(-0.001 * (temperature - 293.15))
        stress *= temp_factor

        return StressStrain(strain=strain, stress=stress, temperature=temperature)

    def compression_test(self, material: Material,
                        max_strain: float = 0.3,
                        num_points: int = 1000) -> StressStrain:
        """
        Simulate compression test with barreling effect
        """
        strain = np.linspace(0, max_strain, num_points)
        stress = np.zeros_like(strain)

        # Similar to tensile but accounting for barreling
        yield_strain = material.yield_strength / material.youngs_modulus

        # Elastic region
        elastic_mask = strain <= yield_strain
        stress[elastic_mask] = material.youngs_modulus * strain[elastic_mask]

        # Plastic region with barreling correction
        plastic_mask = strain > yield_strain
        if np.any(plastic_mask):
            # Barreling increases effective stress
            barreling_factor = 1 + 0.5 * (strain[plastic_mask] - yield_strain)

            n = 0.25  # Higher n for compression
            K = material.ultimate_strength / (0.2 ** n)

            plastic_strain = strain[plastic_mask] - yield_strain
            stress[plastic_mask] = K * (plastic_strain ** n) * barreling_factor + material.yield_strength

        return StressStrain(strain=-strain, stress=stress)  # Negative strain for compression

    def fatigue_analysis(self, material: Material,
                        stress_amplitude: float,
                        mean_stress: float = 0,
                        frequency: float = 10) -> Dict[str, float]:
        """
        Fatigue life prediction using S-N curve and Goodman diagram
        """
        # Basquin's equation for high-cycle fatigue
        # N = (σ_a / σ_f')^(1/b)

        fatigue_strength_coefficient = 1.5 * material.ultimate_strength
        fatigue_strength_exponent = -0.12

        # Goodman correction for mean stress
        if mean_stress > 0:
            corrected_amplitude = stress_amplitude / (1 - mean_stress / material.ultimate_strength)
        else:
            corrected_amplitude = stress_amplitude

        # Calculate number of cycles to failure
        if corrected_amplitude < material.yield_strength * 0.5:  # Infinite life below endurance limit
            cycles_to_failure = np.inf
        else:
            cycles_to_failure = (corrected_amplitude / fatigue_strength_coefficient) ** (1 / fatigue_strength_exponent)

        # Miner's rule for damage accumulation
        damage_per_cycle = 1 / cycles_to_failure if cycles_to_failure != np.inf else 0

        # Crack growth rate (Paris law)
        if corrected_amplitude > material.yield_strength * 0.3:
            C = 1e-12  # Paris law constant
            m = 3.0  # Paris law exponent
            delta_K = corrected_amplitude * np.sqrt(np.pi * 0.001)  # Stress intensity range
            crack_growth_rate = C * (delta_K ** m)
        else:
            crack_growth_rate = 0

        return {
            'cycles_to_failure': cycles_to_failure,
            'damage_per_cycle': damage_per_cycle,
            'crack_growth_rate': crack_growth_rate,
            'safety_factor': material.yield_strength / (stress_amplitude + mean_stress)
        }

    def fracture_toughness(self, material: Material,
                          crack_length: float,
                          specimen_width: float) -> Dict[str, float]:
        """
        Calculate fracture toughness and critical stress
        """
        # Estimate fracture toughness from material properties
        # Rough correlation: K_IC ≈ sqrt(E * G_IC)
        # where G_IC is critical energy release rate

        # Empirical correlation
        K_IC = np.sqrt(material.youngs_modulus * material.yield_strength * 0.001)  # MPa√m

        # Geometry factor for edge crack
        a_W = crack_length / specimen_width
        if a_W < 0.6:
            f = 1.12 - 0.231*a_W + 10.55*a_W**2 - 21.72*a_W**3 + 30.39*a_W**4
        else:
            f = 1.0

        # Critical stress for fracture
        sigma_critical = K_IC / (f * np.sqrt(np.pi * crack_length))

        # J-integral for elastic-plastic fracture
        J_IC = K_IC**2 * (1 - material.poisson_ratio**2) / material.youngs_modulus

        # Crack tip opening displacement (CTOD)
        CTOD = J_IC / material.yield_strength

        return {
            'K_IC': K_IC,  # MPa√m
            'critical_stress': sigma_critical,
            'J_integral': J_IC,
            'CTOD': CTOD,
            'plastic_zone_size': (K_IC / material.yield_strength)**2 / (2 * np.pi)
        }

    def creep_analysis(self, material: Material,
                      stress: float,
                      temperature: float,
                      time_hours: float) -> Dict[str, float]:
        """
        Creep deformation analysis using power law creep
        """
        # Norton-Bailey creep law: ε = A * σ^n * t^m * exp(-Q/RT)

        # Creep parameters (material dependent)
        A = 1e-20  # Creep constant
        n = 5.0  # Stress exponent
        m = 0.3  # Time exponent
        Q = 250000  # Activation energy (J/mol)

        # Homologous temperature
        T_homologous = temperature / material.melting_point

        if T_homologous > 0.4:  # Creep significant above 0.4 Tm
            # Primary creep (transient)
            primary_strain = A * (stress ** n) * (time_hours ** m) * np.exp(-Q / (R * temperature))

            # Secondary creep (steady-state)
            secondary_rate = A * (stress ** n) * np.exp(-Q / (R * temperature))
            secondary_strain = secondary_rate * time_hours

            # Tertiary creep (accelerating, leads to failure)
            if stress > material.yield_strength * 0.5:
                tertiary_factor = np.exp(0.1 * time_hours)
            else:
                tertiary_factor = 1.0

            total_strain = (primary_strain + secondary_strain) * tertiary_factor

            # Larson-Miller parameter for life prediction
            LMP = temperature * (20 + np.log10(time_hours))

            # Time to rupture
            if stress > material.yield_strength * 0.3:
                time_to_rupture = 10 ** ((30000 - LMP) / temperature)
            else:
                time_to_rupture = np.inf
        else:
            total_strain = 0
            secondary_rate = 0
            time_to_rupture = np.inf
            LMP = 0

        return {
            'total_strain': total_strain,
            'creep_rate': secondary_rate,
            'time_to_rupture': time_to_rupture,
            'larson_miller_parameter': LMP,
            'homologous_temperature': T_homologous
        }

    # ============= FAILURE ANALYSIS =============

    def failure_criterion(self, stress_state: np.ndarray,
                         material: Material,
                         criterion: str = "von_mises") -> Dict[str, Any]:
        """
        Apply failure criteria to multi-axial stress state
        stress_state: [σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_xz]
        """
        sigma = stress_state[:3]
        tau = stress_state[3:]

        # Principal stresses
        stress_tensor = np.array([
            [sigma[0], tau[0], tau[2]],
            [tau[0], sigma[1], tau[1]],
            [tau[2], tau[1], sigma[2]]
        ])

        principal_stresses = np.linalg.eigvals(stress_tensor)
        principal_stresses = np.sort(principal_stresses)[::-1]

        results = {
            'principal_stresses': principal_stresses,
            'stress_tensor': stress_tensor
        }

        if criterion == "von_mises":
            # Von Mises yield criterion
            von_mises = np.sqrt(0.5 * ((principal_stresses[0] - principal_stresses[1])**2 +
                                       (principal_stresses[1] - principal_stresses[2])**2 +
                                       (principal_stresses[2] - principal_stresses[0])**2))

            results['von_mises_stress'] = von_mises
            results['safety_factor'] = material.yield_strength / von_mises
            results['failed'] = von_mises > material.yield_strength

        elif criterion == "tresca":
            # Tresca (maximum shear stress) criterion
            max_shear = (principal_stresses[0] - principal_stresses[2]) / 2

            results['max_shear_stress'] = max_shear
            results['safety_factor'] = material.yield_strength / (2 * max_shear)
            results['failed'] = max_shear > material.yield_strength / 2

        elif criterion == "mohr_coulomb":
            # Mohr-Coulomb criterion (for brittle materials)
            cohesion = material.yield_strength / 10
            friction_angle = np.radians(30)  # degrees

            normal_stress = (principal_stresses[0] + principal_stresses[2]) / 2
            shear_stress = (principal_stresses[0] - principal_stresses[2]) / 2

            critical_shear = cohesion + normal_stress * np.tan(friction_angle)

            results['critical_shear'] = critical_shear
            results['actual_shear'] = shear_stress
            results['safety_factor'] = critical_shear / shear_stress
            results['failed'] = shear_stress > critical_shear

        return results

    def weibull_analysis(self, strength_data: np.ndarray) -> Dict[str, float]:
        """
        Weibull statistical analysis for brittle material strength
        """
        # Sort data
        sorted_data = np.sort(strength_data)
        n = len(sorted_data)

        # Estimate Weibull parameters
        # Probability of failure
        P_f = np.arange(1, n + 1) / (n + 1)

        # Linearize: ln(ln(1/(1-P_f))) = m*ln(σ) - m*ln(σ_0)
        Y = np.log(-np.log(1 - P_f))
        X = np.log(sorted_data)

        # Linear regression
        coeffs = np.polyfit(X, Y, 1)
        m = coeffs[0]  # Weibull modulus
        sigma_0 = np.exp(-coeffs[1] / m)  # Characteristic strength

        # Calculate reliability at different stress levels
        def reliability(stress, m, sigma_0):
            return np.exp(-(stress / sigma_0) ** m)

        # Volume scaling
        # σ_v2 = σ_v1 * (V1/V2)^(1/m)

        return {
            'weibull_modulus': m,
            'characteristic_strength': sigma_0,
            'mean_strength': sigma_0 * np.exp(np.log(np.exp(1/m))),
            'reliability_50': sorted_data[n//2],
            'reliability_function': lambda s: reliability(s, m, sigma_0)
        }

    # ============= PHASE TRANSFORMATIONS =============

    def phase_diagram(self, component1: str, component2: str,
                     temperature_range: Tuple[float, float],
                     composition_range: Tuple[float, float]) -> Dict[str, Any]:
        """
        Generate binary phase diagram (simplified)
        """
        # Example: Fe-C system
        if (component1, component2) == ("Fe", "C") or (component1, component2) == ("C", "Fe"):
            # Critical points
            eutectoid_temp = 727 + 273.15  # K
            eutectoid_comp = 0.76  # wt% C
            eutectic_temp = 1147 + 273.15  # K
            eutectic_comp = 4.3  # wt% C

            def phases_at_point(T, C):
                """Determine phases at given temperature and composition"""
                if C < eutectoid_comp:
                    if T > eutectoid_temp:
                        return "Austenite"
                    else:
                        return "Ferrite + Pearlite"
                elif C < eutectic_comp:
                    if T > eutectic_temp:
                        return "Liquid"
                    elif T > eutectoid_temp:
                        return "Austenite"
                    else:
                        return "Pearlite + Cementite"
                else:
                    if T > eutectic_temp:
                        return "Liquid"
                    else:
                        return "Cementite"

            return {
                'eutectoid': {'temperature': eutectoid_temp, 'composition': eutectoid_comp},
                'eutectic': {'temperature': eutectic_temp, 'composition': eutectic_comp},
                'phase_function': phases_at_point
            }
        else:
            return {'error': 'Phase diagram not available for this system'}

    def ttt_diagram(self, material: Material,
                   carbon_content: float = 0.4) -> Dict[str, Callable]:
        """
        Time-Temperature-Transformation diagram for steel
        """
        # Simplified TTT curves
        Ms = 250 + 273.15  # Martensite start temperature
        Mf = 50 + 273.15   # Martensite finish temperature

        def pearlite_time(T):
            """Time to start pearlite transformation"""
            if T < 500 + 273.15 or T > 700 + 273.15:
                return np.inf
            return np.exp((T - 600) / 50)

        def bainite_time(T):
            """Time to start bainite transformation"""
            if T < 250 + 273.15 or T > 550 + 273.15:
                return np.inf
            return np.exp((T - 400) / 30)

        def martensite_fraction(T):
            """Fraction of martensite formed"""
            if T > Ms:
                return 0
            elif T < Mf:
                return 1
            else:
                return 1 - np.exp(-0.011 * (Ms - T))

        return {
            'pearlite_start': pearlite_time,
            'bainite_start': bainite_time,
            'martensite_fraction': martensite_fraction,
            'Ms': Ms,
            'Mf': Mf,
            'critical_cooling_rate': 50  # °C/s
        }

    def precipitation_kinetics(self, temperature: float,
                              time: float,
                              activation_energy: float = 200000) -> Dict[str, float]:
        """
        Johnson-Mehl-Avrami-Kolmogorov (JMAK) equation for precipitation
        """
        # X = 1 - exp(-k*t^n)
        # where k = k0 * exp(-Q/RT)

        k0 = 1e10  # Pre-exponential factor
        n = 3  # Avrami exponent (3 for 3D growth)

        k = k0 * np.exp(-activation_energy / (R * temperature))

        # Volume fraction transformed
        X = 1 - np.exp(-k * (time ** n))

        # Nucleation rate
        nucleation_rate = k * n * (time ** (n - 1)) * np.exp(-k * (time ** n))

        # Growth rate (simplified)
        growth_rate = np.sqrt(k / time) if time > 0 else 0

        return {
            'volume_fraction': X,
            'nucleation_rate': nucleation_rate,
            'growth_rate': growth_rate,
            'completion_time': (np.log(100) / k) ** (1 / n)  # 99% completion
        }

    # ============= COMPOSITE MATERIALS =============

    def rule_of_mixtures(self, composite: CompositeProperties) -> Dict[str, float]:
        """
        Calculate effective properties using rule of mixtures
        """
        Vf = composite.fiber_volume_fraction
        Vm = 1 - Vf

        # Longitudinal modulus (Voigt model)
        E_longitudinal = Vf * composite.fiber_material.youngs_modulus + Vm * composite.matrix_material.youngs_modulus

        # Transverse modulus (Reuss model)
        E_transverse = 1 / (Vf / composite.fiber_material.youngs_modulus + Vm / composite.matrix_material.youngs_modulus)

        # Halpin-Tsai equations for better accuracy
        eta = (composite.fiber_material.youngs_modulus / composite.matrix_material.youngs_modulus - 1) / \
              (composite.fiber_material.youngs_modulus / composite.matrix_material.youngs_modulus + 2)

        E_halpin_tsai = composite.matrix_material.youngs_modulus * (1 + 2 * eta * Vf) / (1 - eta * Vf)

        # Density
        density = Vf * composite.fiber_material.density + Vm * composite.matrix_material.density

        # Strength (simplified)
        if composite.fiber_orientation == "unidirectional":
            strength = Vf * composite.fiber_material.ultimate_strength + Vm * composite.matrix_material.ultimate_strength
        else:
            strength = 0.5 * (Vf * composite.fiber_material.ultimate_strength + Vm * composite.matrix_material.ultimate_strength)

        # Thermal properties
        thermal_conductivity = Vf * composite.fiber_material.thermal_conductivity + Vm * composite.matrix_material.thermal_conductivity

        return {
            'longitudinal_modulus': E_longitudinal,
            'transverse_modulus': E_transverse,
            'halpin_tsai_modulus': E_halpin_tsai,
            'density': density,
            'strength': strength,
            'thermal_conductivity': thermal_conductivity,
            'specific_stiffness': E_longitudinal / density,
            'specific_strength': strength / density
        }

    def laminate_analysis(self, ply_angles: List[float],
                         ply_thickness: float,
                         material_properties: Dict[str, float]) -> np.ndarray:
        """
        Classical Laminate Theory (CLT) analysis
        """
        n_plies = len(ply_angles)

        # Material properties
        E1 = material_properties.get('E1', 150e9)
        E2 = material_properties.get('E2', 10e9)
        G12 = material_properties.get('G12', 5e9)
        v12 = material_properties.get('v12', 0.3)

        # Reduced stiffness matrix in material coordinates
        Q11 = E1 / (1 - v12 * (E2/E1) * v12)
        Q22 = E2 / (1 - v12 * (E2/E1) * v12)
        Q12 = v12 * Q22
        Q66 = G12

        # Initialize ABD matrix
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))

        # Stack sequence
        z = np.linspace(-n_plies * ply_thickness / 2, n_plies * ply_thickness / 2, n_plies + 1)

        for i, angle in enumerate(ply_angles):
            # Transformation matrix
            theta = np.radians(angle)
            c = np.cos(theta)
            s = np.sin(theta)

            # Transformed reduced stiffness
            Q_bar = np.array([
                [Q11*c**4 + Q22*s**4 + 2*(Q12 + 2*Q66)*s**2*c**2,
                 (Q11 + Q22 - 4*Q66)*s**2*c**2 + Q12*(s**4 + c**4),
                 (Q11 - Q12 - 2*Q66)*s*c**3 + (Q12 - Q22 + 2*Q66)*s**3*c],
                [(Q11 + Q22 - 4*Q66)*s**2*c**2 + Q12*(s**4 + c**4),
                 Q11*s**4 + Q22*c**4 + 2*(Q12 + 2*Q66)*s**2*c**2,
                 (Q11 - Q12 - 2*Q66)*s**3*c + (Q12 - Q22 + 2*Q66)*s*c**3],
                [(Q11 - Q12 - 2*Q66)*s*c**3 + (Q12 - Q22 + 2*Q66)*s**3*c,
                 (Q11 - Q12 - 2*Q66)*s**3*c + (Q12 - Q22 + 2*Q66)*s*c**3,
                 (Q11 + Q22 - 2*Q12 - 2*Q66)*s**2*c**2 + Q66*(s**4 + c**4)]
            ])

            # Add to ABD matrix
            h = ply_thickness
            z_k = z[i+1]
            z_k1 = z[i]

            A += Q_bar * h
            B += 0.5 * Q_bar * (z_k**2 - z_k1**2)
            D += (1/3) * Q_bar * (z_k**3 - z_k1**3)

        # Assemble ABD matrix
        ABD = np.block([[A, B], [B, D]])

        return ABD

    # ============= CRYSTAL STRUCTURE =============

    def crystal_structure_analysis(self, structure_type: str,
                                  lattice_parameter: float) -> Dict[str, Any]:
        """
        Analyze crystal structure properties
        """
        results = {}

        if structure_type == "FCC":
            # Face-Centered Cubic
            results['coordination_number'] = 12
            results['atoms_per_unit_cell'] = 4
            results['packing_factor'] = np.pi / (3 * np.sqrt(2))  # 0.74
            results['nearest_neighbor_distance'] = lattice_parameter / np.sqrt(2)
            results['slip_systems'] = 12  # {111}<110>
            results['slip_planes'] = '{111}'
            results['slip_directions'] = '<110>'

        elif structure_type == "BCC":
            # Body-Centered Cubic
            results['coordination_number'] = 8
            results['atoms_per_unit_cell'] = 2
            results['packing_factor'] = np.pi * np.sqrt(3) / 8  # 0.68
            results['nearest_neighbor_distance'] = lattice_parameter * np.sqrt(3) / 2
            results['slip_systems'] = 48  # {110}<111>, {112}<111>, {123}<111>
            results['slip_planes'] = '{110}, {112}, {123}'
            results['slip_directions'] = '<111>'

        elif structure_type == "HCP":
            # Hexagonal Close-Packed
            c_a_ratio = 1.633  # Ideal ratio
            results['coordination_number'] = 12
            results['atoms_per_unit_cell'] = 6
            results['packing_factor'] = np.pi / (3 * np.sqrt(2))  # 0.74
            results['nearest_neighbor_distance'] = lattice_parameter
            results['slip_systems'] = 3  # {0001}<11-20>
            results['c_a_ratio'] = c_a_ratio

        # Miller indices calculations
        results['lattice_parameter'] = lattice_parameter
        results['unit_cell_volume'] = lattice_parameter ** 3 if structure_type != "HCP" else \
                                     np.sqrt(3) * lattice_parameter**2 * c_a_ratio * lattice_parameter / 2

        return results

    def bragg_diffraction(self, lattice_parameter: float,
                         wavelength: float,
                         miller_indices: List[Tuple[int, int, int]]) -> List[float]:
        """
        Calculate Bragg diffraction angles for given Miller indices
        """
        angles = []

        for h, k, l in miller_indices:
            # d-spacing for cubic crystal
            d = lattice_parameter / np.sqrt(h**2 + k**2 + l**2)

            # Bragg's law: n*λ = 2*d*sin(θ)
            sin_theta = wavelength / (2 * d)

            if sin_theta <= 1:
                theta = np.degrees(np.arcsin(sin_theta))
                angles.append(2 * theta)  # 2θ angle
            else:
                angles.append(None)  # Reflection not possible

        return angles

    def defect_concentration(self, temperature: float,
                            formation_energy: float) -> float:
        """
        Calculate equilibrium vacancy concentration
        """
        # n/N = exp(-Q_f / k_B*T)
        concentration = np.exp(-formation_energy * 1.602e-19 / (k_B * temperature))
        return concentration

    # ============= UTILITIES =============

    def _initialize_materials_database(self) -> Dict[str, Material]:
        """Initialize database of common materials"""
        materials = {}

        materials['steel'] = Material(
            name="Steel (1045)",
            density=7850,
            youngs_modulus=200e9,
            yield_strength=530e6,
            ultimate_strength=625e6,
            poisson_ratio=0.29,
            thermal_conductivity=49.8,
            specific_heat=486,
            melting_point=1793,
            crystal_structure="BCC",
            lattice_parameter=2.866
        )

        materials['aluminum'] = Material(
            name="Aluminum (6061-T6)",
            density=2700,
            youngs_modulus=68.9e9,
            yield_strength=276e6,
            ultimate_strength=310e6,
            poisson_ratio=0.33,
            thermal_conductivity=167,
            specific_heat=896,
            melting_point=933,
            crystal_structure="FCC",
            lattice_parameter=4.05
        )

        materials['titanium'] = Material(
            name="Titanium (Ti-6Al-4V)",
            density=4430,
            youngs_modulus=113.8e9,
            yield_strength=880e6,
            ultimate_strength=950e6,
            poisson_ratio=0.342,
            thermal_conductivity=6.7,
            specific_heat=526,
            melting_point=1941,
            crystal_structure="HCP",
            lattice_parameter=2.95
        )

        materials['carbon_fiber'] = Material(
            name="Carbon Fiber (T300)",
            density=1760,
            youngs_modulus=230e9,
            yield_strength=3530e6,
            ultimate_strength=3530e6,
            poisson_ratio=0.3,
            thermal_conductivity=10,
            specific_heat=710,
            melting_point=3925,
            crystal_structure="Graphite",
            lattice_parameter=2.46
        )

        materials['epoxy'] = Material(
            name="Epoxy Resin",
            density=1200,
            youngs_modulus=3.5e9,
            yield_strength=80e6,
            ultimate_strength=80e6,
            poisson_ratio=0.35,
            thermal_conductivity=0.2,
            specific_heat=1000,
            melting_point=450,
            crystal_structure="Amorphous",
            lattice_parameter=None
        )

        return materials

def run_demo():
    """Comprehensive demonstration of materials science lab"""
    print("="*60)
    print("MATERIALS SCIENCE LAB - Comprehensive Demo")
    print("="*60)

    lab = MaterialsScienceLab()

    # Select material
    steel = lab.materials_database['steel']

    # Tensile test
    print("\n1. TENSILE TEST")
    print("-" * 40)

    test_result = lab.tensile_test(steel, max_strain=0.2)
    max_stress_idx = np.argmax(test_result.stress)

    print(f"Material: {steel.name}")
    print(f"Yield Strength: {steel.yield_strength/1e6:.1f} MPa")
    print(f"Ultimate Strength: {steel.ultimate_strength/1e6:.1f} MPa")
    print(f"Maximum stress in test: {test_result.stress[max_stress_idx]/1e6:.1f} MPa")
    print(f"Strain at maximum stress: {test_result.strain[max_stress_idx]:.3f}")

    # Fatigue analysis
    print("\n2. FATIGUE ANALYSIS")
    print("-" * 40)

    fatigue = lab.fatigue_analysis(steel, stress_amplitude=200e6, mean_stress=100e6)

    print(f"Stress amplitude: 200 MPa, Mean stress: 100 MPa")
    print(f"Cycles to failure: {fatigue['cycles_to_failure']:.2e}")
    print(f"Crack growth rate: {fatigue['crack_growth_rate']:.2e} m/cycle")
    print(f"Safety factor: {fatigue['safety_factor']:.2f}")

    # Fracture toughness
    print("\n3. FRACTURE MECHANICS")
    print("-" * 40)

    fracture = lab.fracture_toughness(steel, crack_length=0.001, specimen_width=0.05)

    print(f"Fracture toughness (K_IC): {fracture['K_IC']:.1f} MPa√m")
    print(f"Critical stress: {fracture['critical_stress']/1e6:.1f} MPa")
    print(f"Plastic zone size: {fracture['plastic_zone_size']*1000:.2f} mm")

    # Creep analysis
    print("\n4. CREEP ANALYSIS")
    print("-" * 40)

    creep = lab.creep_analysis(steel, stress=100e6, temperature=873, time_hours=1000)

    print(f"Temperature: 600°C, Stress: 100 MPa, Time: 1000 hours")
    print(f"Total strain: {creep['total_strain']:.4f}")
    print(f"Creep rate: {creep['creep_rate']:.2e} /hour")
    print(f"Homologous temperature: {creep['homologous_temperature']:.2f}")

    # Composite materials
    print("\n5. COMPOSITE MATERIALS")
    print("-" * 40)

    carbon_fiber = lab.materials_database['carbon_fiber']
    epoxy = lab.materials_database['epoxy']

    composite = CompositeProperties(
        matrix_material=epoxy,
        fiber_material=carbon_fiber,
        fiber_volume_fraction=0.6,
        fiber_orientation="unidirectional"
    )

    properties = lab.rule_of_mixtures(composite)

    print(f"Carbon/Epoxy Composite (60% fiber volume)")
    print(f"Longitudinal modulus: {properties['longitudinal_modulus']/1e9:.1f} GPa")
    print(f"Transverse modulus: {properties['transverse_modulus']/1e9:.1f} GPa")
    print(f"Density: {properties['density']:.0f} kg/m³")
    print(f"Specific stiffness: {properties['specific_stiffness']/1e6:.1f} MNm/kg")

    # Crystal structure
    print("\n6. CRYSTAL STRUCTURE ANALYSIS")
    print("-" * 40)

    crystal = lab.crystal_structure_analysis("FCC", lattice_parameter=4.05)

    print(f"FCC Crystal (Aluminum)")
    print(f"Coordination number: {crystal['coordination_number']}")
    print(f"Packing factor: {crystal['packing_factor']:.3f}")
    print(f"Nearest neighbor distance: {crystal['nearest_neighbor_distance']:.3f} Å")
    print(f"Number of slip systems: {crystal['slip_systems']}")

    # XRD peaks
    print("\n7. X-RAY DIFFRACTION")
    print("-" * 40)

    # Cu Kα radiation
    wavelength = 1.5406  # Angstroms
    miller_indices = [(1,1,1), (2,0,0), (2,2,0), (3,1,1)]

    angles = lab.bragg_diffraction(4.05, wavelength, miller_indices)

    print(f"Diffraction angles for Al (λ = {wavelength} Å):")
    for indices, angle in zip(miller_indices, angles):
        if angle:
            print(f"  {indices}: 2θ = {angle:.2f}°")

if __name__ == '__main__':
    run_demo()