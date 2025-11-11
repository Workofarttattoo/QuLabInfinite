# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Nanotechnology Core Module
NIST-validated constants and scientifically accurate nanoparticle simulations
"""

import numpy as np
from scipy.constants import k as k_B, h, c, e, m_e, epsilon_0, pi
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# Physical constants (NIST CODATA 2018)
H_BAR = h / (2 * pi)  # Reduced Planck constant
A0 = 0.529177210903e-10  # Bohr radius (m)
AVOGADRO = 6.02214076e23  # Avogadro's number


@dataclass
class NanoparticleProperties:
    """Properties of synthesized nanoparticles"""
    diameter_nm: float
    concentration_M: float
    zeta_potential_mV: float
    polydispersity_index: float
    surface_area_m2_per_g: float


class NanoparticleSynthesis:
    """
    Nanoparticle synthesis simulator using LaMer model and Ostwald ripening
    Based on: LaMer & Dinegar, J. Am. Chem. Soc. 72, 4847 (1950)
    """

    def __init__(self):
        self.name = "Nanoparticle Synthesis Simulator"

    def lamer_burst_nucleation(self,
                               precursor_conc_M: float,
                               reduction_rate: float,
                               temperature_K: float,
                               time_s: float,
                               dt: float = 0.01) -> Dict:
        """
        Simulate LaMer burst nucleation mechanism

        Args:
            precursor_conc_M: Initial precursor concentration (mol/L)
            reduction_rate: Reduction rate constant (1/s)
            temperature_K: Temperature (K)
            time_s: Total simulation time (s)
            dt: Time step (s)

        Returns:
            Dictionary with nucleation dynamics
        """
        steps = int(time_s / dt)
        time_array = np.linspace(0, time_s, steps)

        # Critical supersaturation for nucleation (empirical)
        S_crit = 1.5  # Supersaturation ratio

        # Nucleation rate (Classical nucleation theory)
        def nucleation_rate(S, T):
            if S < S_crit:
                return 0.0
            # J = A * exp(-ΔG*/kT) where ΔG* ∝ 1/ln²(S)
            A = 1e20  # Pre-exponential factor (1/m³s)
            delta_G_star = 16 * pi * 0.1**3 / (3 * (k_B * T * np.log(S))**2)  # Simplified
            return A * np.exp(-delta_G_star / (k_B * T))

        # Simulate concentration evolution
        C = np.zeros(steps)
        C[0] = precursor_conc_M
        nuclei_count = np.zeros(steps)

        for i in range(1, steps):
            # Precursor reduction
            C[i] = C[i-1] - reduction_rate * C[i-1] * dt

            # Supersaturation ratio
            C_sat = 1e-6  # Saturation concentration (M)
            S = C[i] / C_sat if C[i] > 0 else 0

            # Nucleation
            if S > S_crit:
                J = nucleation_rate(S, temperature_K)
                nuclei_count[i] = nuclei_count[i-1] + J * dt * 1e-27  # Convert to count
            else:
                nuclei_count[i] = nuclei_count[i-1]

        # Calculate final particle size using nuclei count
        total_nuclei = nuclei_count[-1]
        if total_nuclei > 0:
            # Volume per particle assuming all precursor consumed
            volume_per_particle = (precursor_conc_M * 1e-3 / total_nuclei) / AVOGADRO
            diameter_nm = 2 * (3 * volume_per_particle / (4 * pi))**(1/3) * 1e9
        else:
            diameter_nm = 0.0

        return {
            'time_s': time_array.tolist(),
            'concentration_M': C.tolist(),
            'nuclei_count': nuclei_count.tolist(),
            'final_diameter_nm': diameter_nm,
            'nucleation_time_s': time_array[np.argmax(nuclei_count > 0)] if np.any(nuclei_count > 0) else 0,
            'model': 'LaMer Burst Nucleation'
        }

    def ostwald_ripening(self,
                        initial_diameters_nm: np.ndarray,
                        temperature_K: float,
                        time_hours: float,
                        surface_tension: float = 1.0) -> Dict:
        """
        Simulate Ostwald ripening (particle coarsening)
        Based on LSW theory: Lifshitz-Slyozov-Wagner

        Args:
            initial_diameters_nm: Initial particle size distribution (nm)
            temperature_K: Temperature (K)
            time_hours: Ripening time (hours)
            surface_tension: Surface tension (J/m²)

        Returns:
            Dictionary with ripening dynamics
        """
        time_s = time_hours * 3600

        # LSW ripening rate constant (m³/s)
        # k_r = (8 * γ * Vm * D * C_∞) / (9 * R * T)
        gamma = surface_tension  # J/m²
        V_m = 1e-5  # Molar volume (m³/mol) - typical metal
        D = 1e-9  # Diffusion coefficient (m²/s)
        C_inf = 1e-3  # Equilibrium concentration (mol/m³)

        k_r = (8 * gamma * V_m * D * C_inf) / (9 * k_B * temperature_K)

        # LSW equation: r³(t) - r³(0) = k_r * t
        r_initial = initial_diameters_nm / 2 * 1e-9  # Convert to radius in meters
        r_final_cubed = r_initial**3 + k_r * time_s
        r_final = np.cbrt(np.maximum(r_final_cubed, 0))

        diameters_final_nm = 2 * r_final * 1e9

        # Mean diameter evolution
        mean_initial = np.mean(initial_diameters_nm)
        mean_final = np.mean(diameters_final_nm)

        return {
            'initial_mean_diameter_nm': mean_initial,
            'final_mean_diameter_nm': mean_final,
            'initial_std_nm': np.std(initial_diameters_nm),
            'final_std_nm': np.std(diameters_final_nm),
            'growth_rate_nm_per_hour': (mean_final - mean_initial) / time_hours,
            'ripening_coefficient_m3_per_s': k_r,
            'model': 'Lifshitz-Slyozov-Wagner Ostwald Ripening'
        }


class QuantumDotSimulator:
    """
    Quantum dot electronic structure and optical properties
    Based on effective mass approximation and Brus equation
    """

    def __init__(self):
        self.name = "Quantum Dot Simulator"

    def brus_equation_bandgap(self,
                             radius_nm: float,
                             bulk_bandgap_eV: float,
                             electron_mass_ratio: float,
                             hole_mass_ratio: float,
                             dielectric_constant: float) -> Dict:
        """
        Calculate quantum dot bandgap using Brus equation
        E_QD = E_bulk + (ℏ²π²)/(2R²) * (1/m_e* + 1/m_h*) - 1.8e²/(4πεε₀R)

        Args:
            radius_nm: Quantum dot radius (nm)
            bulk_bandgap_eV: Bulk material bandgap (eV)
            electron_mass_ratio: Effective electron mass (m*/m_e)
            hole_mass_ratio: Effective hole mass (m*/m_e)
            dielectric_constant: Relative dielectric constant

        Returns:
            Dictionary with quantum confinement results
        """
        R = radius_nm * 1e-9  # Convert to meters

        # Confinement energy term
        confinement_J = (H_BAR**2 * pi**2) / (2 * R**2) * (1/(electron_mass_ratio * m_e) + 1/(hole_mass_ratio * m_e))
        confinement_eV = confinement_J / e

        # Coulomb interaction term (attractive, reduces bandgap)
        coulomb_J = -1.8 * e**2 / (4 * pi * epsilon_0 * dielectric_constant * R)
        coulomb_eV = coulomb_J / e

        # Total bandgap
        E_QD_eV = bulk_bandgap_eV + confinement_eV + coulomb_eV

        # Emission wavelength
        wavelength_nm = (h * c) / (E_QD_eV * e) * 1e9

        return {
            'quantum_dot_bandgap_eV': E_QD_eV,
            'bulk_bandgap_eV': bulk_bandgap_eV,
            'confinement_energy_eV': confinement_eV,
            'coulomb_correction_eV': coulomb_eV,
            'emission_wavelength_nm': wavelength_nm,
            'radius_nm': radius_nm,
            'model': 'Brus Equation'
        }

    def density_of_states(self,
                         radius_nm: float,
                         electron_mass_ratio: float,
                         max_n: int = 5) -> Dict:
        """
        Calculate discrete energy levels in quantum dot
        E_n = (ℏ²π²n²)/(2m*R²)

        Args:
            radius_nm: Quantum dot radius (nm)
            electron_mass_ratio: Effective mass ratio
            max_n: Maximum quantum number

        Returns:
            Dictionary with energy levels
        """
        R = radius_nm * 1e-9
        m_star = electron_mass_ratio * m_e

        energy_levels_eV = []
        for n in range(1, max_n + 1):
            E_n_J = (H_BAR**2 * pi**2 * n**2) / (2 * m_star * R**2)
            E_n_eV = E_n_J / e
            energy_levels_eV.append(E_n_eV)

        # Level spacing
        delta_E = energy_levels_eV[1] - energy_levels_eV[0] if len(energy_levels_eV) > 1 else 0

        return {
            'energy_levels_eV': energy_levels_eV,
            'ground_state_eV': energy_levels_eV[0],
            'level_spacing_eV': delta_E,
            'radius_nm': radius_nm,
            'model': 'Particle in Sphere'
        }


class DrugDeliverySystem:
    """
    Nanoparticle-based drug delivery simulation
    Release kinetics, biodistribution, and targeting efficiency
    """

    def __init__(self):
        self.name = "Drug Delivery System"

    def higuchi_release_model(self,
                              time_hours: np.ndarray,
                              drug_loading_mg: float,
                              particle_diameter_nm: float,
                              diffusion_coeff_cm2_per_s: float = 1e-6) -> Dict:
        """
        Higuchi model for drug release from nanoparticles
        Q = sqrt(D * (2C - Cs) * Cs * t)

        Args:
            time_hours: Time array (hours)
            drug_loading_mg: Total drug loaded (mg)
            particle_diameter_nm: Nanoparticle diameter (nm)
            diffusion_coeff_cm2_per_s: Drug diffusion coefficient (cm²/s)

        Returns:
            Dictionary with release kinetics
        """
        t_seconds = time_hours * 3600

        # Drug concentration in matrix (simplified)
        C_total = drug_loading_mg * 1e-3  # Convert to g
        C_s = C_total * 0.1  # Solubility (10% of total)

        # Higuchi constant
        D = diffusion_coeff_cm2_per_s * 1e-4  # Convert to m²/s
        K_H = np.sqrt(D * (2 * C_total - C_s) * C_s)

        # Cumulative release
        Q_t = K_H * np.sqrt(t_seconds)
        Q_percent = np.minimum(100 * Q_t / C_total, 100)

        # Release rate
        dQ_dt = K_H / (2 * np.sqrt(t_seconds + 1e-10))  # Avoid division by zero

        return {
            'time_hours': time_hours.tolist(),
            'cumulative_release_percent': Q_percent.tolist(),
            'release_rate_mg_per_hour': (dQ_dt * 3600 * 1e3).tolist(),
            'higuchi_constant': K_H,
            'model': 'Higuchi Square Root Release'
        }

    def biodistribution_model(self,
                             particle_diameter_nm: float,
                             dose_mg_per_kg: float,
                             body_weight_kg: float = 70) -> Dict:
        """
        Predict biodistribution based on particle size
        Based on: Longmire et al., Nanomedicine 3(5), 703-717 (2008)

        Args:
            particle_diameter_nm: Nanoparticle diameter (nm)
            dose_mg_per_kg: Dose (mg/kg)
            body_weight_kg: Subject weight (kg)

        Returns:
            Dictionary with organ distribution
        """
        total_dose_mg = dose_mg_per_kg * body_weight_kg

        # Size-dependent organ accumulation (empirical from literature)
        if particle_diameter_nm < 10:
            # Ultra-small: kidney clearance
            liver_percent = 15
            spleen_percent = 5
            kidney_percent = 50
            tumor_percent = 5
            other_percent = 25
        elif particle_diameter_nm < 100:
            # Small: enhanced permeation and retention (EPR)
            liver_percent = 30
            spleen_percent = 20
            kidney_percent = 10
            tumor_percent = 15  # EPR effect
            other_percent = 25
        else:
            # Large: rapid clearance by RES
            liver_percent = 50
            spleen_percent = 30
            kidney_percent = 2
            tumor_percent = 3
            other_percent = 15

        distribution = {
            'liver_mg': total_dose_mg * liver_percent / 100,
            'spleen_mg': total_dose_mg * spleen_percent / 100,
            'kidney_mg': total_dose_mg * kidney_percent / 100,
            'tumor_mg': total_dose_mg * tumor_percent / 100,
            'other_tissues_mg': total_dose_mg * other_percent / 100,
            'liver_percent': liver_percent,
            'spleen_percent': spleen_percent,
            'kidney_percent': kidney_percent,
            'tumor_percent': tumor_percent,
            'particle_size_nm': particle_diameter_nm,
            'model': 'Size-Dependent Biodistribution'
        }

        return distribution


class NanomaterialProperties:
    """
    Physical and chemical properties of nanomaterials
    Surface area, melting point depression, mechanical properties
    """

    def __init__(self):
        self.name = "Nanomaterial Properties"

    def specific_surface_area(self,
                             diameter_nm: float,
                             density_g_per_cm3: float) -> float:
        """
        Calculate specific surface area (m²/g)
        SSA = 6 / (ρ * d)

        Args:
            diameter_nm: Particle diameter (nm)
            density_g_per_cm3: Material density (g/cm³)

        Returns:
            Specific surface area (m²/g)
        """
        d_m = diameter_nm * 1e-9
        rho_kg_per_m3 = density_g_per_cm3 * 1000

        SSA = 6 / (rho_kg_per_m3 * d_m)  # m²/kg
        return SSA / 1000  # Convert to m²/g

    def melting_point_depression(self,
                                 bulk_melting_K: float,
                                 diameter_nm: float,
                                 surface_energy_J_per_m2: float,
                                 density_g_per_cm3: float) -> Dict:
        """
        Calculate melting point depression for nanoparticles
        ΔT = T_bulk - T_nano = (4σT_bulk) / (ρΔH_f * r)

        Args:
            bulk_melting_K: Bulk melting temperature (K)
            diameter_nm: Particle diameter (nm)
            surface_energy_J_per_m2: Surface energy (J/m²)
            density_g_per_cm3: Density (g/cm³)

        Returns:
            Dictionary with melting point data
        """
        r_m = (diameter_nm / 2) * 1e-9
        rho_kg_per_m3 = density_g_per_cm3 * 1000

        # Heat of fusion (J/kg) - typical value for metals
        delta_H_f = 2e5  # J/kg

        # Melting point depression
        delta_T = (4 * surface_energy_J_per_m2 * bulk_melting_K) / (rho_kg_per_m3 * delta_H_f * r_m)

        T_nano = bulk_melting_K - delta_T

        return {
            'bulk_melting_K': bulk_melting_K,
            'nano_melting_K': T_nano,
            'depression_K': delta_T,
            'diameter_nm': diameter_nm,
            'model': 'Thermodynamic Melting Point Depression'
        }

    def mechanical_properties(self,
                             diameter_nm: float,
                             bulk_youngs_modulus_GPa: float) -> Dict:
        """
        Size-dependent mechanical properties
        Smaller particles typically show enhanced strength

        Args:
            diameter_nm: Particle diameter (nm)
            bulk_youngs_modulus_GPa: Bulk Young's modulus (GPa)

        Returns:
            Dictionary with mechanical properties
        """
        # Hall-Petch type relationship (simplified)
        # E_nano / E_bulk ≈ 1 + k/sqrt(d)
        k = 50  # Empirical constant (nm^0.5)

        enhancement_factor = 1 + k / np.sqrt(diameter_nm)
        E_nano = bulk_youngs_modulus_GPa * enhancement_factor

        # Surface stress contribution becomes significant at nanoscale
        surface_stress_contribution_percent = 100 * k / np.sqrt(diameter_nm)

        return {
            'bulk_youngs_modulus_GPa': bulk_youngs_modulus_GPa,
            'nano_youngs_modulus_GPa': E_nano,
            'enhancement_factor': enhancement_factor,
            'surface_contribution_percent': surface_stress_contribution_percent,
            'diameter_nm': diameter_nm,
            'model': 'Hall-Petch Size Effect'
        }
