# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

from nist_constants import *

"""
Optics Core Module
NIST-validated constants and scientifically accurate optical simulations
"""

import numpy as np
from scipy.constants import c, h, k as k_B, epsilon_0, mu_0, pi, e
from scipy.special import erf
from typing import Dict, List, Tuple, Optional


# Optical constants (NIST)
N_A = 6.02214076e23  # Avogadro's number
H_BAR = h / (2 * pi)  # Reduced Planck constant
ALPHA_FINE = e**2 / (4 * pi * epsilon_0 * H_BAR * c)  # Fine structure constant


class LaserPhysics:
    """
    Laser operation and beam propagation
    Rate equations, cavity modes, Gaussian beams
    """

    def __init__(self):
        self.name = "Laser Physics Simulator"

    def rate_equations_steady_state(self,
                                    pump_rate_per_s: float,
                                    upper_lifetime_s: float,
                                    lower_lifetime_s: float,
                                    cross_section_m2: float,
                                    cavity_loss: float) -> Dict:
        """
        Solve four-level laser rate equations at steady state
        dN2/dt = R_p - N2/τ2 - σ*φ*N2 = 0

        Args:
            pump_rate_per_s: Pumping rate (1/s)
            upper_lifetime_s: Upper level lifetime (s)
            lower_lifetime_s: Lower level lifetime (s)
            cross_section_m2: Stimulated emission cross-section (m²)
            cavity_loss: Round-trip cavity loss (fractional)

        Returns:
            Dictionary with laser operating parameters
        """
        R_p = pump_rate_per_s
        tau_2 = upper_lifetime_s
        sigma = cross_section_m2
        gamma = cavity_loss

        # Threshold pump rate
        R_th = gamma / (sigma * tau_2)

        # Check if above threshold
        if R_p < R_th:
            return {
                'lasing': False,
                'pump_rate_per_s': R_p,
                'threshold_rate_per_s': R_th,
                'population_inversion': 0,
                'photon_flux': 0,
                'output_power_W': 0,
                'model': 'Four-Level Laser Rate Equations'
            }

        # Above threshold - steady state solution
        photon_flux = (R_p - R_th) / gamma  # Photons per second
        N2 = R_th / (sigma * photon_flux) if photon_flux > 0 else 0

        # Output power (assuming single photon energy)
        wavelength_m = 1064e-9  # Nd:YAG example
        photon_energy_J = h * c / wavelength_m
        output_power_W = photon_flux * photon_energy_J * gamma / 2  # Half escapes

        # Slope efficiency
        slope_efficiency = (1 - R_th / R_p) if R_p > 0 else 0

        return {
            'lasing': True,
            'pump_rate_per_s': R_p,
            'threshold_rate_per_s': R_th,
            'population_inversion': N2,
            'photon_flux_per_s': photon_flux,
            'output_power_W': output_power_W,
            'slope_efficiency': slope_efficiency,
            'model': 'Four-Level Laser Rate Equations'
        }

    def gaussian_beam_propagation(self,
                                  wavelength_nm: float,
                                  waist_radius_um: float,
                                  z_positions_mm: np.ndarray) -> Dict:
        """
        Gaussian beam propagation (paraxial wave equation solution)
        w(z) = w0 * sqrt(1 + (z/z_R)²)

        Args:
            wavelength_nm: Wavelength (nm)
            waist_radius_um: Beam waist radius (μm)
            z_positions_mm: Propagation distances (mm)

        Returns:
            Dictionary with beam parameters vs position
        """
        lambda_m = wavelength_nm * 1e-9
        w0 = waist_radius_um * 1e-6
        z = z_positions_mm * 1e-3

        # Rayleigh range
        z_R = pi * w0**2 / lambda_m

        # Beam radius
        w_z = w0 * np.sqrt(1 + (z / z_R)**2)

        # Radius of curvature
        R_z = z * (1 + (z_R / z)**2)
        R_z[z == 0] = np.inf

        # Gouy phase
        psi_z = np.arctan(z / z_R)

        # Divergence angle (far field)
        theta_div_rad = lambda_m / (pi * w0)
        theta_div_mrad = theta_div_rad * 1000

        return {
            'z_positions_mm': z_positions_mm.tolist(),
            'beam_radius_um': (w_z * 1e6).tolist(),
            'radius_of_curvature_mm': (R_z * 1e3).tolist(),
            'gouy_phase_rad': psi_z.tolist(),
            'rayleigh_range_mm': z_R * 1e3,
            'divergence_angle_mrad': theta_div_mrad,
            'waist_radius_um': waist_radius_um,
            'model': 'Gaussian Beam TEM00'
        }

    def q_switching_pulse(self,
                         energy_stored_J: float,
                         cavity_length_m: float,
                         output_coupling: float) -> Dict:
        """
        Q-switched laser pulse characteristics

        Args:
            energy_stored_J: Energy stored in gain medium (J)
            cavity_length_m: Cavity length (m)
            output_coupling: Output coupler reflectivity (fractional)

        Returns:
            Dictionary with pulse parameters
        """
        L = cavity_length_m
        R_oc = output_coupling

        # Round-trip time
        t_rt = 2 * L / c

        # Pulse buildup time (simplified)
        N_passes = -np.log(R_oc) / (1 - R_oc) if R_oc < 1 else 10
        pulse_duration_s = N_passes * t_rt

        # Peak power
        peak_power_W = energy_stored_J / pulse_duration_s

        # Pulse width (FWHM, assuming Gaussian)
        pulse_width_ns = pulse_duration_s * 1e9 / 2

        return {
            'energy_J': energy_stored_J,
            'pulse_duration_s': pulse_duration_s,
            'pulse_width_ns': pulse_width_ns,
            'peak_power_W': peak_power_W,
            'peak_power_MW': peak_power_W * 1e-6,
            'round_trip_time_ns': t_rt * 1e9,
            'cavity_passes': N_passes,
            'model': 'Q-Switched Pulse'
        }


class PhotonicsSimulator:
    """
    Photonic device simulation
    Waveguides, fiber optics, interferometry
    """

    def __init__(self):
        self.name = "Photonics Simulator"

    def waveguide_mode_solver(self,
                             core_refractive_index: float,
                             cladding_refractive_index: float,
                             core_width_um: float,
                             wavelength_nm: float) -> Dict:
        """
        Solve for guided modes in slab waveguide
        Using V-number and normalized propagation constant

        Args:
            core_refractive_index: Core refractive index
            cladding_refractive_index: Cladding refractive index
            core_width_um: Core width (μm)
            wavelength_nm: Wavelength (nm)

        Returns:
            Dictionary with mode properties
        """
        n1 = core_refractive_index
        n2 = cladding_refractive_index
        a = core_width_um * 1e-6 / 2  # Half-width
        lambda_m = wavelength_nm * 1e-9

        # Numerical aperture
        NA = np.sqrt(n1**2 - n2**2)

        # V-number (normalized frequency)
        V = (2 * pi * a / lambda_m) * NA

        # Number of guided modes
        M = int(V / (pi / 2)) + 1

        # Mode cutoff wavelengths
        cutoff_wavelengths_nm = []
        for m in range(M):
            lambda_c = (4 * a * NA) / (2 * m + 1) * 1e9
            cutoff_wavelengths_nm.append(lambda_c)

        # Effective index (approximation for fundamental mode)
        b = 1 - (1 / V)**2  # Normalized propagation constant
        n_eff = n2 + b * (n1 - n2)

        return {
            'V_number': V,
            'number_of_modes': M,
            'numerical_aperture': NA,
            'effective_index': n_eff,
            'cutoff_wavelengths_nm': cutoff_wavelengths_nm,
            'single_mode': M == 1,
            'model': 'Slab Waveguide Mode Solver'
        }

    def fiber_dispersion(self,
                        wavelength_nm: np.ndarray,
                        zero_dispersion_wavelength_nm: float = 1310) -> Dict:
        """
        Calculate chromatic dispersion in optical fiber
        D(λ) = (S0/4) * (λ - λ0⁴/λ³)

        Args:
            wavelength_nm: Wavelength array (nm)
            zero_dispersion_wavelength_nm: Zero dispersion wavelength (nm)

        Returns:
            Dictionary with dispersion characteristics
        """
        lambda_nm = wavelength_nm
        lambda_0 = zero_dispersion_wavelength_nm

        # Dispersion slope at zero (typical for SMF-28)
        S0 = 0.092  # ps/(nm²·km)

        # Dispersion parameter
        D = (S0 / 4) * (lambda_nm - lambda_0**4 / lambda_nm**3)

        # Dispersion slope
        dD_dlambda = (S0 / 4) * (1 + 3 * lambda_0**4 / lambda_nm**4)

        return {
            'wavelength_nm': lambda_nm.tolist(),
            'dispersion_ps_per_nm_km': D.tolist(),
            'dispersion_slope': dD_dlambda.tolist(),
            'zero_dispersion_wavelength_nm': zero_dispersion_wavelength_nm,
            'dispersion_slope_parameter': S0,
            'model': 'Chromatic Dispersion in Single-Mode Fiber'
        }

    def mach_zehnder_interferometer(self,
                                   path_difference_um: float,
                                   wavelength_nm: float,
                                   refractive_index: float = 1.0) -> Dict:
        """
        Mach-Zehnder interferometer transmission

        Args:
            path_difference_um: Optical path difference (μm)
            wavelength_nm: Wavelength (nm)
            refractive_index: Medium refractive index

        Returns:
            Dictionary with interference pattern
        """
        delta_L = path_difference_um * 1e-6
        lambda_m = wavelength_nm * 1e-9
        n = refractive_index

        # Phase difference
        delta_phi = 2 * pi * n * delta_L / lambda_m

        # Transmission (assuming 50/50 beam splitters)
        T = 0.5 * (1 + np.cos(delta_phi))

        # Fringe visibility
        V = 1.0  # Perfect visibility for ideal case

        # Free spectral range
        FSR_nm = lambda_m**2 / (n * delta_L) * 1e9

        return {
            'transmission': T,
            'phase_difference_rad': delta_phi,
            'phase_difference_deg': np.degrees(delta_phi),
            'fringe_visibility': V,
            'free_spectral_range_nm': FSR_nm,
            'path_difference_um': path_difference_um,
            'model': 'Mach-Zehnder Interferometer'
        }


class SpectroscopyAnalysis:
    """
    Spectroscopic analysis tools
    Line broadening, Beer-Lambert law, Raman scattering
    """

    def __init__(self):
        self.name = "Spectroscopy Analysis"

    def lorentzian_lineshape(self,
                            frequency_Hz: np.ndarray,
                            center_frequency_Hz: float,
                            linewidth_Hz: float) -> Dict:
        """
        Lorentzian (Cauchy) lineshape for homogeneous broadening
        I(ν) = (Γ/2π) / ((ν - ν0)² + (Γ/2)²)

        Args:
            frequency_Hz: Frequency array (Hz)
            center_frequency_Hz: Center frequency (Hz)
            linewidth_Hz: FWHM linewidth (Hz)

        Returns:
            Dictionary with lineshape
        """
        nu = frequency_Hz
        nu0 = center_frequency_Hz
        gamma = linewidth_Hz

        # Lorentzian profile
        I = (gamma / (2 * pi)) / ((nu - nu0)**2 + (gamma / 2)**2)

        # Normalize to unit area
        I_norm = I / np.trapz(I, nu)

        return {
            'frequency_Hz': nu.tolist(),
            'intensity_normalized': I_norm.tolist(),
            'center_frequency_Hz': nu0,
            'FWHM_Hz': gamma,
            'Q_factor': nu0 / gamma,
            'model': 'Lorentzian Homogeneous Broadening'
        }

    def gaussian_lineshape(self,
                          frequency_Hz: np.ndarray,
                          center_frequency_Hz: float,
                          doppler_width_Hz: float) -> Dict:
        """
        Gaussian lineshape for Doppler broadening
        I(ν) = (1/Δν_D*sqrt(π)) * exp(-((ν-ν0)/Δν_D)²)

        Args:
            frequency_Hz: Frequency array (Hz)
            center_frequency_Hz: Center frequency (Hz)
            doppler_width_Hz: Doppler width parameter (Hz)

        Returns:
            Dictionary with lineshape
        """
        nu = frequency_Hz
        nu0 = center_frequency_Hz
        delta_nu_D = doppler_width_Hz

        # Gaussian profile
        I = (1 / (delta_nu_D * np.sqrt(pi))) * np.exp(-((nu - nu0) / delta_nu_D)**2)

        # FWHM
        FWHM = 2 * np.sqrt(np.log(2)) * delta_nu_D

        return {
            'frequency_Hz': nu.tolist(),
            'intensity_normalized': I.tolist(),
            'center_frequency_Hz': nu0,
            'FWHM_Hz': FWHM,
            'doppler_width_Hz': delta_nu_D,
            'model': 'Gaussian Doppler Broadening'
        }

    def beer_lambert_absorption(self,
                                concentration_M: float,
                                path_length_cm: float,
                                molar_absorptivity_L_per_mol_cm: float) -> Dict:
        """
        Beer-Lambert law for absorption
        A = ε * c * l
        T = 10^(-A)

        Args:
            concentration_M: Molar concentration (mol/L)
            path_length_cm: Path length (cm)
            molar_absorptivity_L_per_mol_cm: Molar absorptivity (L/(mol·cm))

        Returns:
            Dictionary with absorption data
        """
        epsilon = molar_absorptivity_L_per_mol_cm
        c = concentration_M
        l = path_length_cm

        # Absorbance
        A = epsilon * c * l

        # Transmittance
        T = 10**(-A)

        # Percent transmission
        T_percent = T * 100

        return {
            'absorbance': A,
            'transmittance': T,
            'percent_transmission': T_percent,
            'concentration_M': c,
            'path_length_cm': l,
            'molar_absorptivity': epsilon,
            'model': 'Beer-Lambert Law'
        }


class OpticalMaterials:
    """
    Optical material properties
    Refractive index, dispersion, nonlinear effects
    """

    def __init__(self):
        self.name = "Optical Materials"

    def sellmeier_equation(self,
                          wavelength_um: np.ndarray,
                          B_coefficients: List[float],
                          C_coefficients: List[float]) -> Dict:
        """
        Sellmeier equation for refractive index dispersion
        n²(λ) - 1 = Σ(Bi * λ² / (λ² - Ci))

        Args:
            wavelength_um: Wavelength array (μm)
            B_coefficients: B coefficients [B1, B2, B3]
            C_coefficients: C coefficients [C1, C2, C3] (μm²)

        Returns:
            Dictionary with refractive index vs wavelength
        """
        lambda_sq = wavelength_um**2

        n_sq_minus_1 = 0
        for B, C in zip(B_coefficients, C_coefficients):
            n_sq_minus_1 += B * lambda_sq / (lambda_sq - C)

        n = np.sqrt(1 + n_sq_minus_1)

        # Group refractive index
        dn_dlambda = np.gradient(n, wavelength_um)
        n_g = n - wavelength_um * dn_dlambda

        return {
            'wavelength_um': wavelength_um.tolist(),
            'refractive_index': n.tolist(),
            'group_index': n_g.tolist(),
            'B_coefficients': B_coefficients,
            'C_coefficients': C_coefficients,
            'model': 'Sellmeier Dispersion Equation'
        }

    def nonlinear_refractive_index(self,
                                   intensity_W_per_cm2: float,
                                   n0: float,
                                   n2_cm2_per_W: float) -> Dict:
        """
        Intensity-dependent refractive index (Kerr effect)
        n = n0 + n2 * I

        Args:
            intensity_W_per_cm2: Optical intensity (W/cm²)
            n0: Linear refractive index
            n2_cm2_per_W: Nonlinear refractive index (cm²/W)

        Returns:
            Dictionary with nonlinear effects
        """
        I = intensity_W_per_cm2

        # Total refractive index
        n = n0 + n2_cm2_per_W * I

        # Self-focusing critical power (W)
        P_crit = 0.61 * pi * (0.532e-6)**2 / (8 * n0 * n2_cm2_per_W * 1e4)  # Example for 532nm

        # Phase shift per unit length
        delta_phi_per_cm = (2 * pi / 0.532e-6) * n2_cm2_per_W * I * 1e-2

        return {
            'intensity_W_per_cm2': I,
            'linear_refractive_index': n0,
            'total_refractive_index': n,
            'nonlinear_contribution': n2_cm2_per_W * I,
            'critical_power_W': P_crit,
            'phase_shift_per_cm': delta_phi_per_cm,
            'model': 'Optical Kerr Effect'
        }

    def fresnel_reflectance(self,
                           incident_angle_deg: np.ndarray,
                           n1: float,
                           n2: float) -> Dict:
        """
        Fresnel equations for reflection at interface

        Args:
            incident_angle_deg: Incident angle array (degrees)
            n1: Refractive index of first medium
            n2: Refractive index of second medium

        Returns:
            Dictionary with reflectance vs angle
        """
        theta_i = np.radians(incident_angle_deg)

        # Snell's law
        theta_t = np.arcsin(n1 / n2 * np.sin(theta_i))

        # Fresnel coefficients
        # s-polarization
        r_s = (n1 * np.cos(theta_i) - n2 * np.cos(theta_t)) / (n1 * np.cos(theta_i) + n2 * np.cos(theta_t))
        R_s = r_s**2

        # p-polarization
        r_p = (n2 * np.cos(theta_i) - n1 * np.cos(theta_t)) / (n2 * np.cos(theta_i) + n1 * np.cos(theta_t))
        R_p = r_p**2

        # Brewster angle
        theta_B = np.degrees(np.arctan(n2 / n1))

        return {
            'incident_angle_deg': incident_angle_deg.tolist(),
            'reflectance_s_polarization': R_s.tolist(),
            'reflectance_p_polarization': R_p.tolist(),
            'brewster_angle_deg': theta_B,
            'n1': n1,
            'n2': n2,
            'model': 'Fresnel Reflectance'
        }
