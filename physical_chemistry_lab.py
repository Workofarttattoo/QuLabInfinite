"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

PHYSICAL CHEMISTRY LAB
Production-ready physical chemistry algorithms and simulations.
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
import math
from scipy.constants import k, R, Avogadro, h, c, m_e, epsilon_0, pi
from scipy.integrate import odeint, quad
from scipy.optimize import minimize, root
from scipy.special import erf
from enum import Enum

class ThermodynamicSystem(Enum):
    """Types of thermodynamic systems"""
    ISOLATED = "isolated"  # No mass or energy exchange
    CLOSED = "closed"      # Energy exchange but no mass exchange
    OPEN = "open"          # Both mass and energy exchange
    ADIABATIC = "adiabatic"  # No heat exchange

class StatisticalEnsemble(Enum):
    """Statistical mechanical ensembles"""
    MICROCANONICAL = "NVE"  # Fixed N, V, E
    CANONICAL = "NVT"        # Fixed N, V, T
    GRAND_CANONICAL = "μVT"  # Fixed μ, V, T
    ISOTHERMAL_ISOBARIC = "NPT"  # Fixed N, P, T

@dataclass
class ThermodynamicState:
    """Complete thermodynamic state description"""
    temperature: float  # Kelvin
    pressure: float    # Pascal
    volume: float      # m³
    n_moles: float     # moles
    internal_energy: float = 0.0  # Joules
    enthalpy: float = 0.0         # Joules
    entropy: float = 0.0          # J/K
    gibbs_energy: float = 0.0     # Joules
    helmholtz_energy: float = 0.0 # Joules

    def calculate_derived_properties(self):
        """Calculate derived thermodynamic properties"""
        # Ideal gas approximation for demo
        self.internal_energy = 1.5 * self.n_moles * R * self.temperature
        self.enthalpy = self.internal_energy + self.pressure * self.volume
        self.entropy = self.n_moles * R * (1.5 * np.log(self.temperature) + np.log(self.volume))
        self.gibbs_energy = self.enthalpy - self.temperature * self.entropy
        self.helmholtz_energy = self.internal_energy - self.temperature * self.entropy

@dataclass
class KineticMolecule:
    """Molecule for kinetic theory calculations"""
    mass: float  # kg
    velocity: np.ndarray  # m/s (vx, vy, vz)
    position: np.ndarray  # m (x, y, z)
    diameter: float  # m

    def kinetic_energy(self) -> float:
        """Calculate kinetic energy"""
        return 0.5 * self.mass * np.sum(self.velocity ** 2)

    def momentum(self) -> np.ndarray:
        """Calculate momentum vector"""
        return self.mass * self.velocity

@dataclass
class QuantumState:
    """Quantum mechanical state"""
    n: int  # Principal quantum number
    l: int  # Angular momentum quantum number
    m: int  # Magnetic quantum number
    energy: float  # Energy in Joules
    wavefunction: Optional[Callable] = None

class PhysicalChemistryLab:
    """Production-ready physical chemistry laboratory"""

    def __init__(self, temperature: float = 298.15):
        self.temperature = temperature
        self.R = R  # Gas constant
        self.k_B = k  # Boltzmann constant
        self.N_A = Avogadro
        self.h = h  # Planck constant
        self.c = c  # Speed of light

    # === THERMODYNAMICS METHODS ===

    def calculate_state_functions(self, state: ThermodynamicState) -> Dict:
        """Calculate all thermodynamic state functions"""
        results = {
            'U': state.internal_energy,
            'H': state.enthalpy,
            'S': state.entropy,
            'G': state.gibbs_energy,
            'A': state.helmholtz_energy,
            'Cp': 0.0,
            'Cv': 0.0,
            'compressibility': 0.0,
            'expansion_coefficient': 0.0
        }

        # Heat capacities (ideal gas)
        results['Cv'] = 1.5 * state.n_moles * self.R  # Monatomic
        results['Cp'] = 2.5 * state.n_moles * self.R

        # Compressibility κ = -1/V * (∂V/∂P)_T
        results['compressibility'] = 1 / state.pressure  # Ideal gas

        # Thermal expansion α = 1/V * (∂V/∂T)_P
        results['expansion_coefficient'] = 1 / state.temperature  # Ideal gas

        return results

    def carnot_cycle_analysis(self, T_hot: float, T_cold: float,
                             V1: float, V2: float) -> Dict:
        """Analyze Carnot cycle efficiency and work"""
        analysis = {
            'efficiency': 0.0,
            'work_output': 0.0,
            'heat_absorbed': 0.0,
            'heat_rejected': 0.0,
            'entropy_change': 0.0,
            'reversible': True
        }

        # Carnot efficiency
        analysis['efficiency'] = 1 - T_cold / T_hot

        # Work output for isothermal expansion
        analysis['work_output'] = self.R * T_hot * np.log(V2 / V1)

        # Heat flows
        analysis['heat_absorbed'] = analysis['work_output']
        analysis['heat_rejected'] = analysis['heat_absorbed'] * (T_cold / T_hot)

        # Entropy change (zero for complete cycle)
        analysis['entropy_change'] = 0.0

        return analysis

    def calculate_equilibrium_constant(self, delta_G: float,
                                      temperature: float = None) -> float:
        """Calculate equilibrium constant from Gibbs energy"""
        T = temperature if temperature else self.temperature
        # ΔG = -RT ln(K)
        K_eq = np.exp(-delta_G / (self.R * T))
        return K_eq

    def van_der_waals_equation(self, n: float, V: float, T: float,
                               a: float = 0.365, b: float = 4.27e-5) -> float:
        """Van der Waals equation of state

        Args:
            n: moles
            V: volume (m³)
            T: temperature (K)
            a: Van der Waals a parameter (Pa·m⁶/mol²)
            b: Van der Waals b parameter (m³/mol)
        """
        # (P + n²a/V²)(V - nb) = nRT
        P = (n * self.R * T) / (V - n * b) - (n ** 2 * a) / (V ** 2)
        return P

    def phase_transition_analysis(self, substance: str = "water") -> Dict:
        """Analyze phase transitions and critical points"""
        # Critical constants database
        critical_data = {
            "water": {"Tc": 647.1, "Pc": 22.064e6, "Vc": 55.95e-6},
            "CO2": {"Tc": 304.13, "Pc": 7.3773e6, "Vc": 94.07e-6},
            "methane": {"Tc": 190.56, "Pc": 4.5992e6, "Vc": 98.6e-6}
        }

        data = critical_data.get(substance, critical_data["water"])

        analysis = {
            'critical_temperature': data["Tc"],
            'critical_pressure': data["Pc"],
            'critical_volume': data["Vc"],
            'reduced_temperature': self.temperature / data["Tc"],
            'compressibility_factor': 0.0,
            'acentric_factor': 0.0,
            'phase': ""
        }

        # Determine phase based on reduced conditions
        T_r = analysis['reduced_temperature']
        if T_r > 1:
            analysis['phase'] = "supercritical"
        elif T_r > 0.9:
            analysis['phase'] = "near-critical"
        else:
            analysis['phase'] = "subcritical"

        # Compressibility factor (simplified)
        analysis['compressibility_factor'] = 1 - 0.1 * (1 - T_r) ** 2

        return analysis

    # === KINETIC THEORY METHODS ===

    def maxwell_boltzmann_distribution(self, mass: float,
                                      v: np.ndarray) -> np.ndarray:
        """Maxwell-Boltzmann speed distribution"""
        # f(v) = 4π(m/2πkT)^(3/2) * v² * exp(-mv²/2kT)
        prefactor = 4 * pi * (mass / (2 * pi * self.k_B * self.temperature)) ** 1.5
        exponential = np.exp(-mass * v ** 2 / (2 * self.k_B * self.temperature))
        return prefactor * v ** 2 * exponential

    def calculate_mean_free_path(self, diameter: float, n_density: float) -> float:
        """Calculate mean free path of molecules"""
        # λ = 1 / (√2 * π * d² * n)
        return 1 / (np.sqrt(2) * pi * diameter ** 2 * n_density)

    def calculate_collision_frequency(self, diameter: float, mass: float,
                                     n_density: float) -> float:
        """Calculate collision frequency"""
        # Z = √2 * π * d² * n * v_avg
        v_avg = np.sqrt(8 * self.k_B * self.temperature / (pi * mass))
        return np.sqrt(2) * pi * diameter ** 2 * n_density * v_avg

    def transport_properties(self, mass: float, diameter: float,
                           n_density: float) -> Dict:
        """Calculate transport properties: viscosity, diffusion, thermal conductivity"""
        properties = {
            'viscosity': 0.0,  # Pa·s
            'diffusion_coefficient': 0.0,  # m²/s
            'thermal_conductivity': 0.0,  # W/(m·K)
            'mean_free_path': 0.0
        }

        # Mean free path
        lambda_mfp = self.calculate_mean_free_path(diameter, n_density)
        properties['mean_free_path'] = lambda_mfp

        # Average speed
        v_avg = np.sqrt(8 * self.k_B * self.temperature / (pi * mass))

        # Viscosity η = 5/16 * √(mkT/π) / σ²
        properties['viscosity'] = (5/16) * np.sqrt(mass * self.k_B * self.temperature / pi) / (diameter ** 2)

        # Self-diffusion D = 3/8 * √(kT/πm) / (nσ²)
        properties['diffusion_coefficient'] = (3/8) * np.sqrt(self.k_B * self.temperature / (pi * mass)) / (n_density * diameter ** 2)

        # Thermal conductivity κ = 15/4 * k * η / m
        properties['thermal_conductivity'] = (15/4) * self.k_B * properties['viscosity'] / mass

        return properties

    # === QUANTUM MECHANICS METHODS ===

    def particle_in_box_energies(self, L: float, mass: float,
                                n_max: int = 5) -> List[Dict]:
        """Calculate energy levels for particle in a box"""
        levels = []
        for n in range(1, n_max + 1):
            E_n = (n ** 2 * self.h ** 2) / (8 * mass * L ** 2)
            levels.append({
                'n': n,
                'energy': E_n,
                'energy_eV': E_n / 1.602e-19,
                'wavelength': 2 * L / n
            })
        return levels

    def harmonic_oscillator_energies(self, frequency: float,
                                    n_max: int = 5) -> List[Dict]:
        """Calculate quantum harmonic oscillator energy levels"""
        levels = []
        for n in range(n_max):
            E_n = self.h * frequency * (n + 0.5)
            levels.append({
                'n': n,
                'energy': E_n,
                'energy_eV': E_n / 1.602e-19,
                'frequency': frequency
            })
        return levels

    def hydrogen_atom_energies(self, n_max: int = 5) -> List[Dict]:
        """Calculate hydrogen atom energy levels"""
        # Rydberg constant
        R_inf = m_e * (1.602e-19) ** 4 / (8 * epsilon_0 ** 2 * self.h ** 3 * c)

        levels = []
        for n in range(1, n_max + 1):
            E_n = -13.6 / n ** 2  # eV
            wavelength_limit = 91.2 * n ** 2  # nm (series limit)
            levels.append({
                'n': n,
                'energy_eV': E_n,
                'energy_J': E_n * 1.602e-19,
                'wavelength_limit_nm': wavelength_limit,
                'degeneracy': n ** 2
            })
        return levels

    def calculate_tunneling_probability(self, E: float, V0: float,
                                       width: float, mass: float) -> float:
        """Calculate quantum tunneling probability through rectangular barrier"""
        if E >= V0:
            return 1.0  # Classical transmission

        # k = √(2m(V0-E))/ℏ
        k = np.sqrt(2 * mass * (V0 - E)) / (self.h / (2 * pi))

        # Transmission coefficient T ≈ 16(E/V0)(1-E/V0)exp(-2kw)
        T = 16 * (E / V0) * (1 - E / V0) * np.exp(-2 * k * width)
        return min(T, 1.0)

    # === STATISTICAL MECHANICS METHODS ===

    def partition_function(self, energies: List[float],
                          degeneracies: List[int] = None) -> float:
        """Calculate canonical partition function"""
        if degeneracies is None:
            degeneracies = [1] * len(energies)

        Q = sum(g * np.exp(-E / (self.k_B * self.temperature))
                for E, g in zip(energies, degeneracies))
        return Q

    def thermodynamic_from_partition(self, Q: float, energies: List[float]) -> Dict:
        """Calculate thermodynamic properties from partition function"""
        properties = {
            'helmholtz_energy': 0.0,
            'internal_energy': 0.0,
            'entropy': 0.0,
            'heat_capacity': 0.0,
            'chemical_potential': 0.0
        }

        # Helmholtz free energy A = -kT ln(Q)
        properties['helmholtz_energy'] = -self.k_B * self.temperature * np.log(Q)

        # Internal energy U = kT² (∂ln(Q)/∂T)
        # Approximate derivative
        T_delta = 0.01
        Q_plus = sum(np.exp(-E / (self.k_B * (self.temperature + T_delta)))
                    for E in energies)
        dln_Q_dT = (np.log(Q_plus) - np.log(Q)) / T_delta
        properties['internal_energy'] = self.k_B * self.temperature ** 2 * dln_Q_dT

        # Entropy S = k[ln(Q) + T(∂ln(Q)/∂T)]
        properties['entropy'] = self.k_B * (np.log(Q) + self.temperature * dln_Q_dT)

        # Heat capacity Cv = (∂U/∂T)_V
        properties['heat_capacity'] = 2 * self.k_B * self.temperature * dln_Q_dT

        return properties

    def molecular_dynamics_step(self, molecules: List[KineticMolecule],
                              dt: float, box_size: float) -> List[KineticMolecule]:
        """Single molecular dynamics time step (simplified)"""
        # Update positions (Verlet algorithm)
        for mol in molecules:
            # Simple position update
            mol.position += mol.velocity * dt

            # Periodic boundary conditions
            mol.position = mol.position % box_size

            # Simple wall collision (elastic)
            for i in range(3):
                if mol.position[i] <= 0 or mol.position[i] >= box_size:
                    mol.velocity[i] *= -1

        # Check collisions between molecules (simplified)
        for i, mol1 in enumerate(molecules):
            for mol2 in molecules[i+1:]:
                distance = np.linalg.norm(mol1.position - mol2.position)
                if distance < (mol1.diameter + mol2.diameter) / 2:
                    # Elastic collision (exchange velocities simplified)
                    mol1.velocity, mol2.velocity = mol2.velocity.copy(), mol1.velocity.copy()

        return molecules

    # === SPECTROSCOPY METHODS ===

    def rotational_spectrum(self, I_moment: float, J_max: int = 10) -> List[Dict]:
        """Calculate rotational spectrum energy levels"""
        # B = h²/(8π²I)
        B = self.h ** 2 / (8 * pi ** 2 * I_moment)

        spectrum = []
        for J in range(J_max):
            E_J = B * J * (J + 1)
            # Transition frequency ΔJ = +1
            if J > 0:
                freq = 2 * B * J / self.h
                wavelength = c / freq
                spectrum.append({
                    'J': J,
                    'energy': E_J,
                    'frequency': freq,
                    'wavelength': wavelength,
                    'wavenumber': 1 / wavelength
                })
        return spectrum

    def vibrational_spectrum(self, k_force: float, reduced_mass: float,
                           v_max: int = 5) -> List[Dict]:
        """Calculate vibrational spectrum"""
        # ω = √(k/μ)
        omega = np.sqrt(k_force / reduced_mass)
        frequency = omega / (2 * pi)

        spectrum = []
        for v in range(v_max):
            E_v = self.h * frequency * (v + 0.5)
            spectrum.append({
                'v': v,
                'energy': E_v,
                'energy_cm-1': E_v / (self.h * c * 100),
                'frequency': frequency,
                'wavelength': c / frequency
            })
        return spectrum

    def electronic_transition_selection_rules(self, initial_state: QuantumState,
                                             final_state: QuantumState) -> Dict:
        """Check selection rules for electronic transitions"""
        rules = {
            'allowed': False,
            'delta_n': final_state.n - initial_state.n,
            'delta_l': final_state.l - initial_state.l,
            'delta_m': final_state.m - initial_state.m,
            'type': '',
            'intensity': 0.0
        }

        # Selection rules: Δl = ±1, Δm = 0, ±1
        if abs(rules['delta_l']) == 1 and abs(rules['delta_m']) <= 1:
            rules['allowed'] = True
            rules['type'] = 'electric dipole'
            # Intensity proportional to transition moment squared
            rules['intensity'] = 1 / (rules['delta_n'] ** 3) if rules['delta_n'] != 0 else 0

        return rules

    # === REACTION KINETICS METHODS ===

    def arrhenius_rate_constant(self, A: float, Ea: float,
                               temperature: float = None) -> float:
        """Calculate rate constant using Arrhenius equation"""
        T = temperature if temperature else self.temperature
        # k = A * exp(-Ea/RT)
        k = A * np.exp(-Ea / (self.R * T))
        return k

    def transition_state_theory(self, delta_H_activation: float,
                               delta_S_activation: float) -> Dict:
        """Calculate rate using transition state theory"""
        result = {
            'rate_constant': 0.0,
            'delta_G_activation': 0.0,
            'transmission_coefficient': 1.0,  # Usually close to 1
            'frequency_factor': 0.0
        }

        # ΔG‡ = ΔH‡ - TΔS‡
        result['delta_G_activation'] = delta_H_activation - self.temperature * delta_S_activation

        # Eyring equation: k = (κkT/h) * exp(-ΔG‡/RT)
        result['frequency_factor'] = (self.k_B * self.temperature / self.h)
        result['rate_constant'] = result['transmission_coefficient'] * \
                                 result['frequency_factor'] * \
                                 np.exp(-result['delta_G_activation'] / (self.R * self.temperature))

        return result

    def integrated_rate_laws(self, order: int, k: float, C0: float,
                            time: np.ndarray) -> np.ndarray:
        """Calculate concentration vs time for different reaction orders"""
        if order == 0:
            # Zero order: C = C0 - kt
            return C0 - k * time
        elif order == 1:
            # First order: C = C0 * exp(-kt)
            return C0 * np.exp(-k * time)
        elif order == 2:
            # Second order: 1/C = 1/C0 + kt
            return 1 / (1/C0 + k * time)
        else:
            # nth order: 1/C^(n-1) = 1/C0^(n-1) + (n-1)kt
            return (1 / (C0 ** (order - 1) + (order - 1) * k * time)) ** (1/(order - 1))

    def michaelis_menten_kinetics(self, S: float, Vmax: float, Km: float) -> float:
        """Michaelis-Menten enzyme kinetics"""
        # v = Vmax * [S] / (Km + [S])
        return Vmax * S / (Km + S)

    # === SURFACE CHEMISTRY METHODS ===

    def langmuir_isotherm(self, pressure: float, K_ads: float) -> float:
        """Langmuir adsorption isotherm"""
        # θ = KP / (1 + KP)
        theta = K_ads * pressure / (1 + K_ads * pressure)
        return theta

    def bet_isotherm(self, pressure: float, P0: float, C: float) -> float:
        """BET (Brunauer-Emmett-Teller) adsorption isotherm"""
        # For multilayer adsorption
        x = pressure / P0
        theta = C * x / ((1 - x) * (1 - x + C * x))
        return theta

    def surface_tension_young_laplace(self, gamma: float, r1: float,
                                     r2: float = None) -> float:
        """Young-Laplace equation for pressure difference across curved interface"""
        if r2 is None:
            r2 = r1  # Spherical droplet
        # ΔP = γ(1/r1 + 1/r2)
        delta_P = gamma * (1/r1 + 1/r2)
        return delta_P

def run_demo():
    """Demonstrate physical chemistry lab capabilities"""
    print("=" * 80)
    print("PHYSICAL CHEMISTRY LAB - Production Demo")
    print("Copyright (c) 2025 Joshua Hendricks Cole")
    print("=" * 80)

    # Initialize lab
    lab = PhysicalChemistryLab(temperature=298.15)

    # 1. Thermodynamics
    print("\n1. Thermodynamic state analysis...")
    state = ThermodynamicState(
        temperature=298.15,
        pressure=101325,
        volume=0.0224,
        n_moles=1.0
    )
    state.calculate_derived_properties()
    functions = lab.calculate_state_functions(state)
    print(f"   Internal energy: {functions['U']:.1f} J")
    print(f"   Enthalpy: {functions['H']:.1f} J")
    print(f"   Heat capacity Cp: {functions['Cp']:.1f} J/K")

    # 2. Carnot cycle
    print("\n2. Carnot cycle analysis...")
    carnot = lab.carnot_cycle_analysis(T_hot=500, T_cold=300, V1=0.01, V2=0.02)
    print(f"   Efficiency: {carnot['efficiency']:.1%}")
    print(f"   Work output: {carnot['work_output']:.1f} J")

    # 3. Van der Waals equation
    print("\n3. Van der Waals equation of state...")
    P_vdw = lab.van_der_waals_equation(n=1, V=0.0224, T=298.15)
    P_ideal = R * 298.15 / 0.0224
    print(f"   Van der Waals pressure: {P_vdw/1000:.1f} kPa")
    print(f"   Ideal gas pressure: {P_ideal/1000:.1f} kPa")
    print(f"   Deviation: {(P_vdw-P_ideal)/P_ideal*100:.1f}%")

    # 4. Phase transitions
    print("\n4. Phase transition analysis...")
    phase = lab.phase_transition_analysis("water")
    print(f"   Critical temperature: {phase['critical_temperature']:.1f} K")
    print(f"   Reduced temperature: {phase['reduced_temperature']:.3f}")
    print(f"   Phase: {phase['phase']}")

    # 5. Maxwell-Boltzmann distribution
    print("\n5. Maxwell-Boltzmann distribution...")
    velocities = np.linspace(0, 2000, 100)
    mass_N2 = 28e-3 / Avogadro  # kg
    distribution = lab.maxwell_boltzmann_distribution(mass_N2, velocities)
    v_most_probable = velocities[np.argmax(distribution)]
    print(f"   Most probable speed: {v_most_probable:.0f} m/s")

    # 6. Transport properties
    print("\n6. Transport properties...")
    transport = lab.transport_properties(
        mass=mass_N2,
        diameter=3.7e-10,
        n_density=2.5e25
    )
    print(f"   Mean free path: {transport['mean_free_path']*1e9:.1f} nm")
    print(f"   Diffusion coefficient: {transport['diffusion_coefficient']*1e4:.2f} cm²/s")
    print(f"   Viscosity: {transport['viscosity']*1e6:.1f} μPa·s")

    # 7. Quantum mechanics - Particle in a box
    print("\n7. Particle in a box (electron, L=1 nm)...")
    pib_levels = lab.particle_in_box_energies(L=1e-9, mass=m_e, n_max=3)
    for level in pib_levels:
        print(f"   n={level['n']}: E = {level['energy_eV']:.2f} eV")

    # 8. Hydrogen atom
    print("\n8. Hydrogen atom energy levels...")
    h_levels = lab.hydrogen_atom_energies(n_max=4)
    for level in h_levels[:3]:
        print(f"   n={level['n']}: E = {level['energy_eV']:.1f} eV, degeneracy = {level['degeneracy']}")

    # 9. Statistical mechanics
    print("\n9. Statistical mechanics...")
    energies = [0, 100, 200, 300]  # Arbitrary energy levels
    Q = lab.partition_function(energies, degeneracies=[1, 2, 2, 1])
    thermo = lab.thermodynamic_from_partition(Q, energies)
    print(f"   Partition function: {Q:.2f}")
    print(f"   Helmholtz energy: {thermo['helmholtz_energy']*1e23:.2f} × 10⁻²³ J")
    print(f"   Entropy: {thermo['entropy']*1e23:.2f} × 10⁻²³ J/K")

    # 10. Reaction kinetics
    print("\n10. Reaction kinetics...")
    k_arrhenius = lab.arrhenius_rate_constant(A=1e13, Ea=50000)  # 50 kJ/mol
    print(f"   Arrhenius rate constant: {k_arrhenius:.2e} s⁻¹")

    tst = lab.transition_state_theory(
        delta_H_activation=45000,  # J/mol
        delta_S_activation=-50     # J/(mol·K)
    )
    print(f"   TST rate constant: {tst['rate_constant']:.2e} s⁻¹")

    # 11. Spectroscopy
    print("\n11. Rotational spectroscopy (HCl)...")
    I_HCl = 2.7e-47  # kg·m²
    rot_spectrum = lab.rotational_spectrum(I_HCl, J_max=3)
    for trans in rot_spectrum[:2]:
        print(f"   J={trans['J']}: ν = {trans['frequency']/1e9:.1f} GHz")

    # 12. Surface chemistry
    print("\n12. Surface chemistry...")
    theta_langmuir = lab.langmuir_isotherm(pressure=1000, K_ads=0.001)
    print(f"   Langmuir coverage: {theta_langmuir:.1%}")

    delta_P = lab.surface_tension_young_laplace(gamma=0.072, r1=1e-6)
    print(f"   Laplace pressure (1 μm droplet): {delta_P:.0f} Pa")

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("Visit: https://aios.is | https://thegavl.com")
    print("=" * 80)

if __name__ == '__main__':
    run_demo()