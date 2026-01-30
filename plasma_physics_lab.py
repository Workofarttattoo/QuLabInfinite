"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

PLASMA PHYSICS LAB
Advanced plasma physics simulations including Debye length, plasma frequency,
MHD equations, particle-in-cell basics, instability analysis, and fusion conditions.
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from scipy import constants, special, integrate, optimize
from typing import Tuple, Dict, List, Optional, Callable
from dataclasses import dataclass, field


# Physical constants
KB = constants.k  # Boltzmann constant (J/K)
EPSILON_0 = constants.epsilon_0  # Vacuum permittivity (F/m)
MU_0 = constants.mu_0  # Vacuum permeability (H/m)
ME = constants.m_e  # Electron mass (kg)
MP = constants.m_p  # Proton mass (kg)
E_CHARGE = constants.e  # Elementary charge (C)
C = constants.c  # Speed of light (m/s)
EV_TO_J = constants.eV  # Electron volt to Joules


@dataclass
class PlasmaSpecies:
    """Represents a plasma species (electrons, ions, etc.)."""
    name: str
    mass: float  # kg
    charge: float  # Coulombs
    density: float  # m^-3
    temperature: float  # Kelvin
    drift_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m/s


class PlasmaPhysicsLab:
    """Advanced plasma physics simulation laboratory."""

    def __init__(self):
        """Initialize the plasma physics lab."""
        self.name = "Plasma Physics Laboratory"
        self.version = "2.0.0"

    def debye_length(self, electron_density: float, electron_temp: float,
                    ion_density: Optional[float] = None,
                    ion_temp: Optional[float] = None) -> float:
        """
        Calculate Debye length for plasma shielding.

        Args:
            electron_density: Electron density (m^-3)
            electron_temp: Electron temperature (K)
            ion_density: Ion density (m^-3), defaults to electron_density
            ion_temp: Ion temperature (K), defaults to electron_temp

        Returns:
            Debye length in meters
        """
        if ion_density is None:
            ion_density = electron_density
        if ion_temp is None:
            ion_temp = electron_temp

        # Electron Debye length
        lambda_De = np.sqrt(EPSILON_0 * KB * electron_temp /
                           (electron_density * E_CHARGE**2))

        # Ion Debye length
        lambda_Di = np.sqrt(EPSILON_0 * KB * ion_temp /
                           (ion_density * E_CHARGE**2))

        # Total Debye length
        lambda_D = 1 / np.sqrt(1 / lambda_De**2 + 1 / lambda_Di**2)

        return lambda_D

    def plasma_frequency(self, density: float, mass: float,
                        charge: float = E_CHARGE) -> float:
        """
        Calculate plasma frequency.

        Args:
            density: Particle density (m^-3)
            mass: Particle mass (kg)
            charge: Particle charge (C)

        Returns:
            Plasma frequency in rad/s
        """
        omega_p = np.sqrt(density * charge**2 / (EPSILON_0 * mass))
        return omega_p

    def cyclotron_frequency(self, B: float, mass: float,
                          charge: float = E_CHARGE) -> float:
        """
        Calculate cyclotron (Larmor) frequency.

        Args:
            B: Magnetic field strength (T)
            mass: Particle mass (kg)
            charge: Particle charge (C)

        Returns:
            Cyclotron frequency in rad/s
        """
        omega_c = abs(charge) * B / mass
        return omega_c

    def thermal_velocity(self, temperature: float, mass: float) -> float:
        """
        Calculate thermal velocity of particles.

        Args:
            temperature: Temperature (K)
            mass: Particle mass (kg)

        Returns:
            Thermal velocity in m/s
        """
        v_th = np.sqrt(2 * KB * temperature / mass)
        return v_th

    def collision_frequency(self, density: float, temperature: float,
                          charge1: float = E_CHARGE, charge2: float = E_CHARGE,
                          mass1: float = ME, mass2: float = MP) -> float:
        """
        Calculate Coulomb collision frequency.

        Args:
            density: Particle density (m^-3)
            temperature: Temperature (K)
            charge1, charge2: Charges of colliding particles (C)
            mass1, mass2: Masses of colliding particles (kg)

        Returns:
            Collision frequency in Hz
        """
        # Coulomb logarithm (simplified)
        lambda_D = self.debye_length(density, temperature)
        b_min = max(abs(charge1 * charge2) / (4 * np.pi * EPSILON_0 * KB * temperature),
                   constants.hbar / np.sqrt(2 * mass1 * KB * temperature))

        ln_Lambda = np.log(lambda_D / b_min)

        # Collision frequency
        v_rel = np.sqrt(8 * KB * temperature / (np.pi * mass1))
        nu = density * (charge1 * charge2)**2 * ln_Lambda / \
             (4 * np.pi * EPSILON_0**2 * mass1 * mass2 * v_rel**3)

        return nu

    def plasma_beta(self, density: float, temperature: float, B: float) -> float:
        """
        Calculate plasma beta (ratio of thermal to magnetic pressure).

        Args:
            density: Particle density (m^-3)
            temperature: Temperature (K)
            B: Magnetic field strength (T)

        Returns:
            Plasma beta (dimensionless)
        """
        p_thermal = density * KB * temperature
        p_magnetic = B**2 / (2 * MU_0)
        beta = p_thermal / p_magnetic
        return beta

    def alfven_velocity(self, B: float, density: float, mass: float = MP) -> float:
        """
        Calculate Alfvén wave velocity.

        Args:
            B: Magnetic field strength (T)
            density: Mass density (kg/m^3) or number density with mass
            mass: Particle mass if density is number density

        Returns:
            Alfvén velocity in m/s
        """
        # If density is number density, convert to mass density
        if density < 1e10:  # Assume it's mass density if very large
            rho = density
        else:
            rho = density * mass

        v_A = B / np.sqrt(MU_0 * rho)
        return v_A

    def mhd_equations(self, rho: float, v: np.ndarray, B: np.ndarray,
                     p: float, J: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate MHD equation terms (ideal MHD).

        Args:
            rho: Mass density (kg/m^3)
            v: Velocity field (m/s)
            B: Magnetic field (T)
            p: Pressure (Pa)
            J: Current density (A/m^2)

        Returns:
            Dictionary of MHD equation terms
        """
        # Momentum equation: dv/dt = -1/rho * grad(p) + J×B/rho - v·grad(v)
        momentum_pressure = -1 / rho * np.gradient(p)
        momentum_lorentz = np.cross(J, B) / rho

        # Induction equation: dB/dt = curl(v×B)
        # Simplified for demonstration
        induction_term = np.cross(v, B)

        # Energy equation terms
        magnetic_pressure = B**2 / (2 * MU_0)

        return {
            'momentum_pressure': momentum_pressure,
            'momentum_lorentz': momentum_lorentz,
            'induction': induction_term,
            'magnetic_pressure': magnetic_pressure
        }

    def dispersion_relation_waves(self, k: float, omega: float,
                                 plasma_params: Dict) -> complex:
        """
        Calculate dispersion relation for plasma waves.

        Args:
            k: Wave number (1/m)
            omega: Wave frequency (rad/s)
            plasma_params: Dictionary with plasma parameters

        Returns:
            Dispersion relation value (should be zero for valid modes)
        """
        ne = plasma_params.get('electron_density', 1e18)
        Te = plasma_params.get('electron_temp', 1e6)
        B = plasma_params.get('B_field', 0.1)
        theta = plasma_params.get('angle', 0)  # Angle to B field

        # Plasma frequencies
        omega_pe = self.plasma_frequency(ne, ME)
        omega_ce = self.cyclotron_frequency(B, ME)

        # Thermal velocity
        v_the = self.thermal_velocity(Te, ME)

        # Parallel and perpendicular components
        k_parallel = k * np.cos(theta)
        k_perp = k * np.sin(theta)

        # Simplified cold plasma dispersion relation
        # For electromagnetic waves in magnetized plasma
        epsilon_parallel = 1 - omega_pe**2 / omega**2
        epsilon_perp = 1 - omega_pe**2 / (omega**2 - omega_ce**2)
        epsilon_cross = omega_pe**2 * omega_ce / (omega * (omega**2 - omega_ce**2))

        # Dispersion relation determinant
        n_parallel = k_parallel * C / omega
        n_perp = k_perp * C / omega
        n2 = n_parallel**2 + n_perp**2

        D = epsilon_parallel * (n2 - epsilon_perp)**2 + \
            epsilon_cross**2 * (n2 - epsilon_parallel)

        return D

    def landau_damping_rate(self, k: float, omega: float,
                           temperature: float, density: float,
                           mass: float = ME) -> float:
        """
        Calculate Landau damping rate for plasma waves.

        Args:
            k: Wave number (1/m)
            omega: Wave frequency (rad/s)
            temperature: Particle temperature (K)
            density: Particle density (m^-3)
            mass: Particle mass (kg)

        Returns:
            Landau damping rate (1/s)
        """
        v_th = self.thermal_velocity(temperature, mass)
        v_phase = omega / k

        # Plasma frequency
        omega_p = self.plasma_frequency(density, mass)

        # Landau damping rate (for Langmuir waves)
        xi = v_phase / (np.sqrt(2) * v_th)
        gamma = -np.sqrt(np.pi / 8) * (omega_p**3 / k**2 / v_th**3) * \
                np.exp(-xi**2)

        return gamma

    def two_stream_instability(self, v_drift: float, density: float,
                              temperature: float, mass: float = ME) -> Dict:
        """
        Analyze two-stream instability.

        Args:
            v_drift: Drift velocity between streams (m/s)
            density: Density of each stream (m^-3)
            temperature: Temperature (K)
            mass: Particle mass (kg)

        Returns:
            Dictionary with instability analysis
        """
        v_th = self.thermal_velocity(temperature, mass)
        omega_p = self.plasma_frequency(density, mass)

        # Critical drift velocity for instability
        v_critical = v_th * np.sqrt(3)

        # Maximum growth rate
        if v_drift > v_critical:
            gamma_max = omega_p * np.sqrt(3) / 2 / np.power(2, 1/3)
            k_max = omega_p / v_drift
            unstable = True
        else:
            gamma_max = 0
            k_max = 0
            unstable = False

        return {
            'unstable': unstable,
            'critical_velocity': v_critical,
            'max_growth_rate': gamma_max,
            'most_unstable_k': k_max,
            'drift_velocity': v_drift
        }

    def fusion_reaction_rate(self, T: float, n1: float, n2: float,
                           reaction: str = 'DT') -> float:
        """
        Calculate fusion reaction rate.

        Args:
            T: Temperature (keV)
            n1: Density of species 1 (m^-3)
            n2: Density of species 2 (m^-3)
            reaction: Type of reaction ('DT', 'DD', 'DHe3')

        Returns:
            Reaction rate (reactions/m^3/s)
        """
        # Convert temperature to keV if needed
        if T > 1000:  # Assume it's in Kelvin
            T = T * KB / (1000 * EV_TO_J)  # Convert to keV

        # Reaction rate coefficients <σv> in m^3/s
        if reaction == 'DT':
            # D-T fusion: D + T → He4 + n + 17.6 MeV
            if T < 1:
                sigma_v = 1e-24
            elif T < 10:
                sigma_v = 1.1e-24 * T**2 * np.exp(-20 / T**(1/3))
            else:
                sigma_v = 8.7e-21 / T**2

        elif reaction == 'DD':
            # D-D fusion: D + D → He3 + n + 3.27 MeV (50%)
            #            D + D → T + p + 4.03 MeV (50%)
            if T < 1:
                sigma_v = 1e-26
            elif T < 10:
                sigma_v = 2.3e-26 * T**2 * np.exp(-18.8 / T**(1/3))
            else:
                sigma_v = 1.4e-22 / T**2

        elif reaction == 'DHe3':
            # D-He3 fusion: D + He3 → He4 + p + 18.3 MeV
            if T < 10:
                sigma_v = 1e-28
            elif T < 50:
                sigma_v = 1e-26 * T**2 * np.exp(-30 / T**(1/3))
            else:
                sigma_v = 5.5e-21 / T**(2/3)
        else:
            sigma_v = 0

        # Reaction rate
        R = n1 * n2 * sigma_v

        return R

    def lawson_criterion(self, n: float, tau: float, T: float) -> float:
        """
        Calculate Lawson criterion for fusion ignition.

        Args:
            n: Plasma density (m^-3)
            tau: Energy confinement time (s)
            T: Temperature (keV)

        Returns:
            n*tau*T product (m^-3 * s * keV)
        """
        # Convert temperature if needed
        if T > 1000:
            T = T * KB / (1000 * EV_TO_J)

        return n * tau * T

    def ignition_condition(self, n: float, T: float,
                          reaction: str = 'DT') -> float:
        """
        Calculate required confinement time for ignition.

        Args:
            n: Plasma density (m^-3)
            T: Temperature (keV)
            reaction: Fusion reaction type

        Returns:
            Required confinement time (s)
        """
        # Convert temperature if needed
        if T > 1000:
            T_keV = T * KB / (1000 * EV_TO_J)
        else:
            T_keV = T
            T = T_keV * 1000 * EV_TO_J / KB  # Convert to Kelvin

        # Fusion power density
        P_fusion = self.fusion_reaction_rate(T_keV, n/2, n/2, reaction)

        if reaction == 'DT':
            E_fusion = 17.6e6 * EV_TO_J  # Energy per reaction
            E_alpha = 3.5e6 * EV_TO_J    # Alpha particle energy
        elif reaction == 'DD':
            E_fusion = 3.65e6 * EV_TO_J  # Average energy
            E_alpha = 0.82e6 * EV_TO_J   # Charged product energy
        else:
            E_fusion = 18.3e6 * EV_TO_J
            E_alpha = 18.3e6 * EV_TO_J   # All energy to charged products

        # Alpha heating power
        P_alpha = P_fusion * E_alpha / E_fusion

        # Bremsstrahlung losses (simplified)
        Z_eff = 1  # Effective charge
        P_brem = 5.35e-37 * Z_eff * n**2 * np.sqrt(T)

        # Required confinement time for ignition
        # P_alpha = P_losses = 3nkT/tau
        if P_alpha > P_brem:
            tau_E = 3 * n * KB * T / (P_alpha - P_brem)
        else:
            tau_E = np.inf  # Cannot achieve ignition

        return tau_E

    def particle_in_cell_step(self, particles: np.ndarray, velocities: np.ndarray,
                             E_field: np.ndarray, B_field: np.ndarray,
                             dt: float, charge: float = E_CHARGE,
                             mass: float = ME) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single step of particle-in-cell (PIC) simulation.

        Args:
            particles: Particle positions (N x 3 array)
            velocities: Particle velocities (N x 3 array)
            E_field: Electric field at particle positions (N x 3)
            B_field: Magnetic field at particle positions (N x 3)
            dt: Time step (s)
            charge: Particle charge (C)
            mass: Particle mass (kg)

        Returns:
            Updated positions and velocities
        """
        # Boris pusher algorithm for particle motion in E and B fields

        # Half acceleration from E field
        v_minus = velocities + (charge / mass) * E_field * dt / 2

        # Rotation from B field
        theta = charge * B_field * dt / mass
        theta_mag = np.linalg.norm(theta, axis=1, keepdims=True)

        # Avoid division by zero
        theta_mag[theta_mag == 0] = 1

        t = np.tan(theta_mag / 2) * theta / theta_mag
        s = 2 * t / (1 + np.sum(t * t, axis=1, keepdims=True))

        v_cross = v_minus + np.cross(v_minus, t)
        v_plus = v_minus + np.cross(v_cross, s)

        # Second half acceleration from E field
        new_velocities = v_plus + (charge / mass) * E_field * dt / 2

        # Update positions
        new_particles = particles + new_velocities * dt

        return new_particles, new_velocities

    def magnetic_confinement_time(self, R: float, a: float, B: float,
                                 T: float, n: float) -> float:
        """
        Estimate confinement time in tokamak (simplified scaling).

        Args:
            R: Major radius (m)
            a: Minor radius (m)
            B: Toroidal field (T)
            T: Temperature (keV)
            n: Density (10^20 m^-3)

        Returns:
            Energy confinement time (s)
        """
        # ITER98 H-mode scaling (simplified)
        # tau_E ~ R^1.97 * a^0.58 * B^0.15 * n^0.41 * P^-0.69

        # Assume some heating power
        P = 50e6  # 50 MW heating (typical)

        # Simplified scaling
        tau_E = 0.05 * R**1.97 * a**0.58 * B**0.15 * (n * 1e20)**0.41 / \
                (P / 1e6)**0.69

        return tau_E

    def plasma_instability_analysis(self, plasma_params: Dict) -> Dict:
        """
        Analyze various plasma instabilities.

        Args:
            plasma_params: Dictionary of plasma parameters

        Returns:
            Dictionary of instability analysis results
        """
        n = plasma_params.get('density', 1e19)
        T = plasma_params.get('temperature', 1e7)
        B = plasma_params.get('B_field', 1.0)
        v_drift = plasma_params.get('drift_velocity', 1e6)

        results = {}

        # Rayleigh-Taylor instability (for density gradient)
        g_eff = 1e4  # Effective gravity (centrifugal acceleration)
        L_n = plasma_params.get('density_scale_length', 0.1)

        gamma_RT = np.sqrt(abs(g_eff / L_n))
        results['rayleigh_taylor_growth'] = gamma_RT

        # Kelvin-Helmholtz instability (for velocity shear)
        L_v = plasma_params.get('velocity_scale_length', 0.1)
        if L_v > 0:
            gamma_KH = v_drift / L_v
            results['kelvin_helmholtz_growth'] = gamma_KH

        # Drift wave instability
        omega_pe = self.plasma_frequency(n, ME)
        rho_s = np.sqrt(KB * T / MP) / self.cyclotron_frequency(B, MP)

        k_perp = 1 / rho_s  # Typical perpendicular wave number
        omega_drift = k_perp * KB * T / (E_CHARGE * B * L_n)
        results['drift_frequency'] = omega_drift

        # Interchange instability (bad curvature)
        R = plasma_params.get('major_radius', 2.0)
        pressure_gradient = n * KB * T / L_n

        gamma_interchange = np.sqrt(2 * pressure_gradient / (n * MP * R * L_n))
        results['interchange_growth'] = gamma_interchange

        return results


def run_demo():
    """Demonstrate plasma physics calculations."""
    lab = PlasmaPhysicsLab()
    print(f"Initializing {lab.name} v{lab.version}")
    print("=" * 60)

    # 1. Basic plasma parameters
    print("\n1. Basic Plasma Parameters:")

    n_e = 1e19  # m^-3
    T_e = 1e7   # K (~860 eV)
    B = 2.0     # Tesla

    lambda_D = lab.debye_length(n_e, T_e)
    omega_pe = lab.plasma_frequency(n_e, ME)
    omega_ce = lab.cyclotron_frequency(B, ME)
    v_th = lab.thermal_velocity(T_e, ME)
    beta = lab.plasma_beta(n_e, T_e, B)

    print(f"   Electron density: {n_e:.2e} m^-3")
    print(f"   Temperature: {T_e:.2e} K ({T_e * KB / EV_TO_J:.1f} eV)")
    print(f"   Debye length: {lambda_D * 1e6:.2f} μm")
    print(f"   Plasma frequency: {omega_pe / (2 * np.pi) / 1e9:.2f} GHz")
    print(f"   Cyclotron frequency: {omega_ce / (2 * np.pi) / 1e9:.2f} GHz")
    print(f"   Thermal velocity: {v_th / 1e6:.2f} Mm/s")
    print(f"   Plasma beta: {beta:.3f}")

    # 2. MHD waves
    print("\n2. MHD Wave Properties:")

    n_ion = n_e
    v_A = lab.alfven_velocity(B, n_ion * MP)
    print(f"   Alfvén velocity: {v_A / 1e6:.2f} Mm/s")

    # Sound speed
    gamma = 5/3  # Adiabatic index
    c_s = np.sqrt(gamma * KB * T_e / MP)
    print(f"   Sound speed: {c_s / 1e6:.2f} Mm/s")

    # Fast magnetosonic speed
    v_fast = np.sqrt(v_A**2 + c_s**2)
    print(f"   Fast magnetosonic speed: {v_fast / 1e6:.2f} Mm/s")

    # 3. Fusion conditions
    print("\n3. Fusion Reaction Analysis:")

    T_fusion = 10  # keV
    n_fusion = 1e20  # m^-3

    # D-T reaction rate
    rate_DT = lab.fusion_reaction_rate(T_fusion, n_fusion/2, n_fusion/2, 'DT')
    print(f"   D-T reaction rate at {T_fusion} keV: {rate_DT:.2e} reactions/m³/s")

    # Power density
    E_per_reaction = 17.6e6 * EV_TO_J  # Joules
    P_fusion = rate_DT * E_per_reaction
    print(f"   Fusion power density: {P_fusion / 1e6:.2f} MW/m³")

    # Ignition condition
    tau_E_required = lab.ignition_condition(n_fusion, T_fusion, 'DT')
    print(f"   Required confinement time for ignition: {tau_E_required:.3f} s")

    # Lawson criterion
    lawson = lab.lawson_criterion(n_fusion, tau_E_required, T_fusion)
    print(f"   Lawson parameter n·τ·T: {lawson:.2e} m^-3·s·keV")

    # 4. Instabilities
    print("\n4. Plasma Instabilities:")

    # Two-stream instability
    v_drift = 5e6  # m/s
    two_stream = lab.two_stream_instability(v_drift, n_e, T_e)
    print(f"   Two-stream instability:")
    print(f"     Drift velocity: {v_drift / 1e6:.1f} Mm/s")
    print(f"     Critical velocity: {two_stream['critical_velocity'] / 1e6:.1f} Mm/s")
    print(f"     Unstable: {two_stream['unstable']}")
    if two_stream['unstable']:
        print(f"     Max growth rate: {two_stream['max_growth_rate'] / 1e9:.2f} GHz")

    # Landau damping
    k = 1e6  # 1/m
    omega = omega_pe * 1.1  # Near plasma frequency
    gamma_L = lab.landau_damping_rate(k, omega, T_e, n_e)
    print(f"\n   Landau damping:")
    print(f"     Wave frequency: {omega / (2 * np.pi) / 1e9:.2f} GHz")
    print(f"     Damping rate: {gamma_L / 1e6:.2f} MHz")

    # 5. Magnetic confinement
    print("\n5. Magnetic Confinement (Tokamak):")

    # ITER-like parameters
    R_major = 6.2  # m
    a_minor = 2.0  # m
    B_tor = 5.3    # T
    T_conf = 10    # keV
    n_conf = 1.0   # 10^20 m^-3

    tau_E = lab.magnetic_confinement_time(R_major, a_minor, B_tor, T_conf, n_conf)
    print(f"   Major radius: {R_major} m")
    print(f"   Minor radius: {a_minor} m")
    print(f"   Toroidal field: {B_tor} T")
    print(f"   Estimated confinement time: {tau_E:.2f} s")

    # Triple product
    triple_product = n_conf * tau_E * T_conf
    print(f"   Triple product n·τ·T: {triple_product:.1f} × 10^20 m^-3·s·keV")
    print(f"   Required for ignition: ~3 × 10^21 m^-3·s·keV")

    # 6. Particle-in-cell simulation step
    print("\n6. Particle-in-Cell Simulation:")

    # Initialize particles
    N_particles = 100
    particles = np.random.randn(N_particles, 3) * 1e-6  # μm scale
    velocities = np.random.randn(N_particles, 3) * v_th / 10

    # Uniform fields for demo
    E_field = np.ones((N_particles, 3)) * 1e5  # V/m
    B_field = np.ones((N_particles, 3)) * 0.1  # T
    B_field[:, 0] = 0  # B in y-z plane

    dt = 1e-12  # 1 ps time step

    # Single PIC step
    new_particles, new_velocities = lab.particle_in_cell_step(
        particles, velocities, E_field, B_field, dt
    )

    # Calculate average drift
    v_drift_calc = np.mean(new_velocities - velocities, axis=0) / dt
    v_ExB = np.cross(E_field[0], B_field[0]) / np.dot(B_field[0], B_field[0])

    print(f"   Number of particles: {N_particles}")
    print(f"   Time step: {dt * 1e12:.1f} ps")
    print(f"   E×B drift (theoretical): {v_ExB / 1e3:.1f} km/s")
    print(f"   Average particle displacement: {np.mean(np.linalg.norm(new_particles - particles, axis=1)) * 1e9:.2f} nm")

    # 7. Wave dispersion
    print("\n7. Wave Dispersion Analysis:")

    plasma_params = {
        'electron_density': n_e,
        'electron_temp': T_e,
        'B_field': B,
        'angle': np.pi / 4  # 45 degrees to B
    }

    # Find wave modes
    k_test = 1e6  # 1/m
    omega_test = omega_pe

    dispersion = lab.dispersion_relation_waves(k_test, omega_test, plasma_params)
    print(f"   Wave number: {k_test / 1e6:.1f} /Mm")
    print(f"   Test frequency: {omega_test / (2 * np.pi) / 1e9:.2f} GHz")
    print(f"   Dispersion relation |D|: {abs(dispersion):.2e}")
    print(f"   (Should be ~0 for valid wave mode)")

    # 8. Comprehensive instability analysis
    print("\n8. Instability Analysis:")

    full_params = {
        'density': n_e,
        'temperature': T_e,
        'B_field': B,
        'drift_velocity': v_drift,
        'density_scale_length': 0.1,
        'velocity_scale_length': 0.05,
        'major_radius': R_major
    }

    instabilities = lab.plasma_instability_analysis(full_params)

    print(f"   Rayleigh-Taylor growth rate: {instabilities['rayleigh_taylor_growth'] / 1e3:.1f} kHz")
    print(f"   Kelvin-Helmholtz growth rate: {instabilities['kelvin_helmholtz_growth'] / 1e6:.1f} MHz")
    print(f"   Drift wave frequency: {instabilities['drift_frequency'] / (2 * np.pi) / 1e3:.1f} kHz")
    print(f"   Interchange growth rate: {instabilities['interchange_growth'] / 1e3:.1f} kHz")

    print("\n" + "=" * 60)
    print("Plasma Physics Lab demonstration complete!")


if __name__ == "__main__":
    run_demo()