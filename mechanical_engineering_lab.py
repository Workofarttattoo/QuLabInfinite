"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

MECHANICAL ENGINEERING LAB
Production-ready engineering calculations for stress analysis, fatigue, vibration, heat transfer, and fluid mechanics.
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy import constants
from scipy.optimize import fsolve
from typing import Dict, Tuple, Optional, List
import warnings

@dataclass
class MechanicalEngineeringLab:
    """
    Comprehensive mechanical engineering calculations including:
    - Stress and strain analysis
    - Fatigue life prediction
    - Vibration analysis
    - Heat transfer calculations
    - Fluid mechanics
    - Finite Element Analysis basics
    """

    # Material Properties (Default: Steel)
    youngs_modulus: float = 200e9  # Pa (Young's modulus)
    poissons_ratio: float = 0.3    # Dimensionless
    yield_strength: float = 250e6   # Pa (Yield strength)
    ultimate_strength: float = 400e6  # Pa (Ultimate tensile strength)
    density: float = 7850           # kg/m^3
    thermal_conductivity: float = 50  # W/(m·K)
    specific_heat: float = 486      # J/(kg·K)
    thermal_expansion: float = 12e-6  # 1/K (Coefficient of thermal expansion)

    # Operating Conditions
    temperature: float = 293         # K (Room temperature)
    pressure: float = 101325         # Pa (Atmospheric pressure)

    def __post_init__(self):
        """Initialize derived properties"""
        self.shear_modulus = self.youngs_modulus / (2 * (1 + self.poissons_ratio))
        self.bulk_modulus = self.youngs_modulus / (3 * (1 - 2 * self.poissons_ratio))
        self.lame_first = self.youngs_modulus * self.poissons_ratio / ((1 + self.poissons_ratio) * (1 - 2 * self.poissons_ratio))
        self.lame_second = self.shear_modulus

    def calculate_stress(self, force: float, area: float) -> float:
        """
        Calculate normal stress using σ = F/A

        Args:
            force: Applied force in Newtons
            area: Cross-sectional area in m^2

        Returns:
            Stress in Pa
        """
        if area <= 0:
            raise ValueError("Area must be positive")
        return force / area

    def calculate_strain(self, stress: float) -> float:
        """
        Calculate strain using Hooke's Law: ε = σ/E

        Args:
            stress: Applied stress in Pa

        Returns:
            Strain (dimensionless)
        """
        return stress / self.youngs_modulus

    def von_mises_stress(self, sigma_x: float, sigma_y: float, sigma_z: float,
                         tau_xy: float = 0, tau_yz: float = 0, tau_xz: float = 0) -> float:
        """
        Calculate von Mises equivalent stress for 3D stress state

        σ_vm = sqrt(0.5 * [(σ_x - σ_y)² + (σ_y - σ_z)² + (σ_z - σ_x)² + 6(τ_xy² + τ_yz² + τ_xz²)])

        Args:
            sigma_x, sigma_y, sigma_z: Normal stresses in Pa
            tau_xy, tau_yz, tau_xz: Shear stresses in Pa

        Returns:
            Von Mises equivalent stress in Pa
        """
        term1 = (sigma_x - sigma_y)**2 + (sigma_y - sigma_z)**2 + (sigma_z - sigma_x)**2
        term2 = 6 * (tau_xy**2 + tau_yz**2 + tau_xz**2)
        return np.sqrt(0.5 * (term1 + term2))

    def principal_stresses(self, sigma_x: float, sigma_y: float, tau_xy: float) -> Tuple[float, float, float]:
        """
        Calculate principal stresses and angle for 2D stress state

        Args:
            sigma_x, sigma_y: Normal stresses in Pa
            tau_xy: Shear stress in Pa

        Returns:
            Tuple of (σ_1, σ_2, θ_p) where θ_p is principal angle in radians
        """
        sigma_avg = (sigma_x + sigma_y) / 2
        radius = np.sqrt(((sigma_x - sigma_y) / 2)**2 + tau_xy**2)

        sigma_1 = sigma_avg + radius  # Maximum principal stress
        sigma_2 = sigma_avg - radius  # Minimum principal stress

        # Principal angle
        if sigma_x == sigma_y:
            theta_p = np.pi / 4 if tau_xy > 0 else -np.pi / 4
        else:
            theta_p = 0.5 * np.arctan(2 * tau_xy / (sigma_x - sigma_y))

        return sigma_1, sigma_2, theta_p

    def fatigue_life_basquin(self, stress_amplitude: float,
                             fatigue_strength_coeff: Optional[float] = None,
                             fatigue_strength_exp: float = -0.085) -> int:
        """
        Calculate fatigue life using Basquin's equation: σ_a = σ_f' * (2N_f)^b

        Args:
            stress_amplitude: Stress amplitude in Pa
            fatigue_strength_coeff: Fatigue strength coefficient (default: 0.9 * ultimate_strength)
            fatigue_strength_exp: Fatigue strength exponent (typically -0.05 to -0.12)

        Returns:
            Number of cycles to failure
        """
        if fatigue_strength_coeff is None:
            fatigue_strength_coeff = 0.9 * self.ultimate_strength

        if stress_amplitude <= 0 or stress_amplitude > self.ultimate_strength:
            return 0

        # Solve for N_f: N_f = 0.5 * (σ_a / σ_f')^(1/b)
        cycles = 0.5 * (stress_amplitude / fatigue_strength_coeff)**(1 / fatigue_strength_exp)
        return int(cycles)

    def goodman_diagram(self, mean_stress: float, alternating_stress: float,
                       endurance_limit: Optional[float] = None) -> float:
        """
        Calculate safety factor using Goodman criterion for fatigue under mean stress

        Goodman equation: σ_a/S_e + σ_m/S_ut = 1/n

        Args:
            mean_stress: Mean stress in Pa
            alternating_stress: Alternating stress in Pa
            endurance_limit: Material endurance limit (default: 0.5 * ultimate_strength)

        Returns:
            Safety factor
        """
        if endurance_limit is None:
            endurance_limit = 0.5 * self.ultimate_strength

        if alternating_stress <= 0:
            return float('inf')

        # Calculate safety factor
        denominator = alternating_stress / endurance_limit + mean_stress / self.ultimate_strength

        if denominator <= 0:
            return float('inf')

        return 1.0 / denominator

    def natural_frequency_beam(self, length: float, width: float, height: float,
                              boundary: str = 'simply_supported') -> float:
        """
        Calculate natural frequency of a rectangular beam

        Args:
            length: Beam length in meters
            width: Beam width in meters
            height: Beam height in meters
            boundary: Boundary condition ('simply_supported', 'cantilever', 'fixed')

        Returns:
            Natural frequency in Hz
        """
        # Moment of inertia for rectangular cross-section
        moment_of_inertia = (width * height**3) / 12

        # Cross-sectional area
        area = width * height

        # Mass per unit length
        mass_per_length = self.density * area

        # Boundary condition coefficients (first mode)
        coeffs = {
            'simply_supported': (np.pi / length)**2,
            'cantilever': (1.875 / length)**2,
            'fixed': (4.73 / length)**2
        }

        if boundary not in coeffs:
            raise ValueError(f"Boundary condition must be one of {list(coeffs.keys())}")

        coeff = coeffs[boundary]

        # Natural frequency: ω = (coeff) * sqrt(EI/μ)
        omega = coeff * np.sqrt(self.youngs_modulus * moment_of_inertia / mass_per_length)

        return omega / (2 * np.pi)  # Convert to Hz

    def damped_vibration(self, mass: float, stiffness: float, damping: float,
                        time: np.ndarray, initial_disp: float = 0.01,
                        initial_vel: float = 0) -> np.ndarray:
        """
        Calculate response of damped single degree of freedom system

        Args:
            mass: Mass in kg
            stiffness: Spring stiffness in N/m
            damping: Damping coefficient in N·s/m
            time: Time array in seconds
            initial_disp: Initial displacement in meters
            initial_vel: Initial velocity in m/s

        Returns:
            Displacement array in meters
        """
        # Natural frequency and damping ratio
        omega_n = np.sqrt(stiffness / mass)
        zeta = damping / (2 * np.sqrt(mass * stiffness))

        if zeta < 1:  # Underdamped
            omega_d = omega_n * np.sqrt(1 - zeta**2)
            A = initial_disp
            B = (initial_vel + zeta * omega_n * initial_disp) / omega_d

            displacement = np.exp(-zeta * omega_n * time) * (
                A * np.cos(omega_d * time) + B * np.sin(omega_d * time)
            )
        elif zeta == 1:  # Critically damped
            displacement = np.exp(-omega_n * time) * (
                initial_disp + (initial_vel + omega_n * initial_disp) * time
            )
        else:  # Overdamped
            r1 = -omega_n * (zeta - np.sqrt(zeta**2 - 1))
            r2 = -omega_n * (zeta + np.sqrt(zeta**2 - 1))
            A = (initial_vel - r2 * initial_disp) / (r1 - r2)
            B = initial_disp - A

            displacement = A * np.exp(r1 * time) + B * np.exp(r2 * time)

        return displacement

    def heat_conduction_1d(self, length: float, temp_hot: float, temp_cold: float,
                          area: float, thermal_resistance: float = 0) -> Dict[str, float]:
        """
        Calculate 1D steady-state heat conduction using Fourier's law

        Args:
            length: Material thickness in meters
            temp_hot: Hot side temperature in K
            temp_cold: Cold side temperature in K
            area: Cross-sectional area in m^2
            thermal_resistance: Additional thermal resistance in K/W

        Returns:
            Dictionary with heat flux, heat rate, and thermal resistance
        """
        if length <= 0 or area <= 0:
            raise ValueError("Length and area must be positive")

        # Thermal resistance of material
        R_material = length / (self.thermal_conductivity * area)
        R_total = R_material + thermal_resistance

        # Temperature difference
        delta_T = temp_hot - temp_cold

        # Heat rate
        heat_rate = delta_T / R_total

        # Heat flux
        heat_flux = heat_rate / area

        return {
            'heat_flux': heat_flux,  # W/m^2
            'heat_rate': heat_rate,  # W
            'thermal_resistance': R_total  # K/W
        }

    def convection_heat_transfer(self, surface_temp: float, fluid_temp: float,
                                area: float, heat_transfer_coeff: float = 25) -> Dict[str, float]:
        """
        Calculate convective heat transfer using Newton's law of cooling

        Args:
            surface_temp: Surface temperature in K
            fluid_temp: Fluid temperature in K
            area: Surface area in m^2
            heat_transfer_coeff: Convection coefficient in W/(m^2·K)

        Returns:
            Dictionary with heat rate and heat flux
        """
        delta_T = abs(surface_temp - fluid_temp)
        heat_rate = heat_transfer_coeff * area * delta_T
        heat_flux = heat_transfer_coeff * delta_T

        return {
            'heat_rate': heat_rate,  # W
            'heat_flux': heat_flux,  # W/m^2
            'nusselt_number': heat_transfer_coeff * 1.0 / self.thermal_conductivity  # Approximate
        }

    def reynolds_number(self, velocity: float, characteristic_length: float,
                       viscosity: float = 1.81e-5) -> float:
        """
        Calculate Reynolds number for fluid flow

        Re = ρVL/μ

        Args:
            velocity: Flow velocity in m/s
            characteristic_length: Characteristic length in meters
            viscosity: Dynamic viscosity in Pa·s (default: air at 20°C)

        Returns:
            Reynolds number (dimensionless)
        """
        return self.density * velocity * characteristic_length / viscosity

    def pressure_drop_pipe(self, flow_rate: float, pipe_diameter: float,
                          pipe_length: float, roughness: float = 0.0015e-3,
                          viscosity: float = 1e-3) -> Dict[str, float]:
        """
        Calculate pressure drop in pipe using Darcy-Weisbach equation

        Args:
            flow_rate: Volumetric flow rate in m^3/s
            pipe_diameter: Pipe internal diameter in meters
            pipe_length: Pipe length in meters
            roughness: Pipe roughness in meters (default: commercial steel)
            viscosity: Fluid dynamic viscosity in Pa·s (default: water)

        Returns:
            Dictionary with pressure drop, velocity, Reynolds number, and friction factor
        """
        # Cross-sectional area
        area = np.pi * (pipe_diameter / 2)**2

        # Flow velocity
        velocity = flow_rate / area

        # Reynolds number
        Re = self.density * velocity * pipe_diameter / viscosity

        # Friction factor (Colebrook equation approximation)
        if Re < 2300:  # Laminar flow
            friction_factor = 64 / Re
        else:  # Turbulent flow (Swamee-Jain equation)
            relative_roughness = roughness / pipe_diameter
            A = relative_roughness / 3.7
            B = 5.74 / Re**0.9
            friction_factor = 0.25 / (np.log10(A + B))**2

        # Pressure drop
        pressure_drop = friction_factor * (pipe_length / pipe_diameter) * \
                       (0.5 * self.density * velocity**2)

        return {
            'pressure_drop': pressure_drop,  # Pa
            'velocity': velocity,  # m/s
            'reynolds_number': Re,
            'friction_factor': friction_factor,
            'flow_regime': 'laminar' if Re < 2300 else 'turbulent'
        }

    def finite_element_stiffness_matrix(self, length: float, area: float) -> np.ndarray:
        """
        Generate stiffness matrix for 1D bar element (basic FEA)

        Args:
            length: Element length in meters
            area: Cross-sectional area in m^2

        Returns:
            2x2 stiffness matrix in N/m
        """
        k = self.youngs_modulus * area / length
        return np.array([[k, -k],
                        [-k, k]])

    def beam_deflection(self, length: float, load: float,
                       width: float, height: float,
                       load_type: str = 'point_center') -> float:
        """
        Calculate maximum deflection of a beam under various loading conditions

        Args:
            length: Beam length in meters
            load: Applied load in N (point) or N/m (distributed)
            width: Beam width in meters
            height: Beam height in meters
            load_type: Type of loading ('point_center', 'point_end', 'distributed')

        Returns:
            Maximum deflection in meters
        """
        # Moment of inertia for rectangular cross-section
        I = (width * height**3) / 12

        deflections = {
            'point_center': load * length**3 / (48 * self.youngs_modulus * I),
            'point_end': load * length**3 / (3 * self.youngs_modulus * I),
            'distributed': 5 * load * length**4 / (384 * self.youngs_modulus * I)
        }

        if load_type not in deflections:
            raise ValueError(f"Load type must be one of {list(deflections.keys())}")

        return deflections[load_type]

    def thermal_stress(self, temp_change: float, constrained: bool = True) -> float:
        """
        Calculate thermal stress due to temperature change

        Args:
            temp_change: Change in temperature in K
            constrained: Whether thermal expansion is constrained

        Returns:
            Thermal stress in Pa (if constrained) or strain (if unconstrained)
        """
        thermal_strain = self.thermal_expansion * temp_change

        if constrained:
            # Stress = E * α * ΔT
            return self.youngs_modulus * thermal_strain
        else:
            return thermal_strain


def run_demo():
    """Demonstrate mechanical engineering calculations"""
    print("=" * 70)
    print("MECHANICAL ENGINEERING LAB - Production Demo")
    print("=" * 70)

    lab = MechanicalEngineeringLab()

    # Stress-Strain Analysis
    print("\n1. STRESS-STRAIN ANALYSIS")
    print("-" * 40)
    force = 10000  # N
    area = 0.001   # m^2
    stress = lab.calculate_stress(force, area)
    strain = lab.calculate_strain(stress)
    print(f"Applied Force: {force:.0f} N")
    print(f"Cross-sectional Area: {area:.4f} m²")
    print(f"Stress: {stress/1e6:.2f} MPa")
    print(f"Strain: {strain:.6f}")

    # Von Mises Stress
    print("\n2. VON MISES STRESS")
    print("-" * 40)
    sigma_x, sigma_y, sigma_z = 100e6, 50e6, 75e6  # Pa
    tau_xy = 25e6  # Pa
    vm_stress = lab.von_mises_stress(sigma_x, sigma_y, sigma_z, tau_xy)
    print(f"Normal Stresses: σx={sigma_x/1e6:.0f}, σy={sigma_y/1e6:.0f}, σz={sigma_z/1e6:.0f} MPa")
    print(f"Shear Stress: τxy={tau_xy/1e6:.0f} MPa")
    print(f"Von Mises Stress: {vm_stress/1e6:.2f} MPa")
    print(f"Safety Factor: {lab.yield_strength/vm_stress:.2f}")

    # Fatigue Analysis
    print("\n3. FATIGUE LIFE PREDICTION")
    print("-" * 40)
    stress_amplitude = 150e6  # Pa
    cycles = lab.fatigue_life_basquin(stress_amplitude)
    print(f"Stress Amplitude: {stress_amplitude/1e6:.0f} MPa")
    print(f"Predicted Life: {cycles:,} cycles")
    mean_stress = 50e6
    alt_stress = 100e6
    sf = lab.goodman_diagram(mean_stress, alt_stress)
    print(f"Goodman Safety Factor: {sf:.2f}")

    # Vibration Analysis
    print("\n4. VIBRATION ANALYSIS")
    print("-" * 40)
    length, width, height = 1.0, 0.05, 0.01  # meters
    freq_ss = lab.natural_frequency_beam(length, width, height, 'simply_supported')
    freq_cant = lab.natural_frequency_beam(length, width, height, 'cantilever')
    print(f"Beam Dimensions: {length}m × {width}m × {height}m")
    print(f"Natural Frequency (Simply Supported): {freq_ss:.2f} Hz")
    print(f"Natural Frequency (Cantilever): {freq_cant:.2f} Hz")

    # Heat Transfer
    print("\n5. HEAT TRANSFER")
    print("-" * 40)
    heat_result = lab.heat_conduction_1d(0.01, 373, 293, 0.01)
    print(f"Temperature Difference: 80 K")
    print(f"Heat Flux: {heat_result['heat_flux']:.2f} W/m²")
    print(f"Heat Rate: {heat_result['heat_rate']:.2f} W")
    print(f"Thermal Resistance: {heat_result['thermal_resistance']:.4f} K/W")

    # Fluid Mechanics
    print("\n6. FLUID MECHANICS")
    print("-" * 40)
    flow_result = lab.pressure_drop_pipe(0.01, 0.05, 10)
    print(f"Flow Rate: 0.01 m³/s")
    print(f"Pipe: 50mm diameter, 10m length")
    print(f"Pressure Drop: {flow_result['pressure_drop']/1000:.2f} kPa")
    print(f"Reynolds Number: {flow_result['reynolds_number']:.0f}")
    print(f"Flow Regime: {flow_result['flow_regime']}")

    print("\n" + "=" * 70)
    print("Demo completed successfully!")


if __name__ == '__main__':
    run_demo()