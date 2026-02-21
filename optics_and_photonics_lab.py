"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Optics and Photonics Laboratory - Production-Ready Optical Physics and Photonics
Comprehensive suite for Gaussian beams, interference, diffraction, fiber optics, and nonlinear optics
"""

import numpy as np
from scipy import integrate, special, optimize
from scipy.constants import c, h, k, e, epsilon_0, pi
from typing import Dict, List, Tuple, Optional, Callable
import warnings

class OpticsPhotonicsLab:
    """Advanced optics and photonics calculations for wave propagation and nonlinear effects"""

    def __init__(self):
        # Physical constants
        self.c = c  # Speed of light
        self.h = h  # Planck constant
        self.k_B = k  # Boltzmann constant
        self.e = e  # Elementary charge
        self.epsilon_0 = epsilon_0  # Vacuum permittivity

        # Common wavelengths (nm)
        self.wavelengths = {
            'UV': 355, 'Blue': 473, 'Green': 532, 'Red': 633,
            'NIR': 1064, 'Telecom': 1550
        }

    def gaussian_beam_propagation(self, z: np.ndarray, w0: float, wavelength: float,
                                 n: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Calculate Gaussian beam parameters during propagation

        Args:
            z: Propagation distances (m)
            w0: Beam waist radius (m)
            wavelength: Wavelength (m)
            n: Refractive index

        Returns:
            Beam parameters vs distance
        """
        # Rayleigh range
        z_R = pi * w0**2 * n / wavelength

        # Beam radius
        w_z = w0 * np.sqrt(1 + (z / z_R)**2)

        # Radius of curvature
        R_z = np.zeros_like(z)
        mask = np.abs(z) > 1e-10
        R_z[mask] = z[mask] * (1 + (z_R / z[mask])**2)
        R_z[~mask] = np.inf

        # Gouy phase
        psi_z = np.arctan(z / z_R)

        # Complex beam parameter
        q_z = z + 1j * z_R

        # Beam divergence (far-field)
        theta = wavelength / (pi * w0 * n)

        return {
            'z': z,
            'beam_radius': w_z,
            'radius_of_curvature': R_z,
            'gouy_phase': psi_z,
            'q_parameter': q_z,
            'rayleigh_range': z_R,
            'divergence': theta
        }

    def fresnel_diffraction(self, x: np.ndarray, aperture_width: float,
                          distance: float, wavelength: float) -> np.ndarray:
        """
        Calculate Fresnel diffraction pattern

        Args:
            x: Observation points (m)
            aperture_width: Slit width (m)
            distance: Propagation distance (m)
            wavelength: Wavelength (m)

        Returns:
            Intensity distribution
        """
        # Fresnel number
        F = aperture_width**2 / (wavelength * distance)

        # Fresnel parameter
        v1 = -np.sqrt(2 / (wavelength * distance)) * (x + aperture_width/2)
        v2 = -np.sqrt(2 / (wavelength * distance)) * (x - aperture_width/2)

        # Fresnel integrals
        S1, C1 = special.fresnel(v1)
        S2, C2 = special.fresnel(v2)

        # Complex amplitude
        U = (C2 - C1) + 1j * (S2 - S1)

        # Intensity
        I = np.abs(U)**2

        return I

    def fabry_perot_transmission(self, wavelengths: np.ndarray, cavity_length: float,
                                R1: float, R2: float, n: float = 1.0) -> np.ndarray:
        """
        Calculate Fabry-Perot cavity transmission spectrum

        Args:
            wavelengths: Array of wavelengths (m)
            cavity_length: Cavity length (m)
            R1, R2: Mirror reflectivities
            n: Refractive index in cavity

        Returns:
            Transmission spectrum
        """
        # Finesse
        F = 4 * np.sqrt(R1 * R2) / (1 - np.sqrt(R1 * R2))**2

        # Round-trip phase
        delta = 4 * pi * n * cavity_length / wavelengths

        # Airy function
        T = 1 / (1 + F * np.sin(delta/2)**2)

        return T

    def mach_zehnder_interference(self, phase_diff: np.ndarray, visibility: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Calculate Mach-Zehnder interferometer output

        Args:
            phase_diff: Phase difference between arms (rad)
            visibility: Fringe visibility (0-1)

        Returns:
            Output intensities at both ports
        """
        # Input intensity normalized to 1
        I_in = 1.0

        # Output port 1 (constructive)
        I_1 = I_in * (1 + visibility * np.cos(phase_diff)) / 2

        # Output port 2 (destructive)
        I_2 = I_in * (1 - visibility * np.cos(phase_diff)) / 2

        return {
            'phase': phase_diff,
            'port1': I_1,
            'port2': I_2,
            'visibility': visibility
        }

    def fiber_mode_propagation(self, wavelength: float, core_radius: float,
                             n_core: float, n_cladding: float) -> Dict[str, any]:
        """
        Calculate fiber optic mode parameters

        Args:
            wavelength: Wavelength (m)
            core_radius: Core radius (m)
            n_core: Core refractive index
            n_cladding: Cladding refractive index

        Returns:
            Fiber parameters and mode information
        """
        # Numerical aperture
        NA = np.sqrt(n_core**2 - n_cladding**2)

        # V-number (normalized frequency)
        V = 2 * pi * core_radius * NA / wavelength

        # Number of modes (step-index fiber)
        if V < 2.405:
            num_modes = 1  # Single mode
            mode_type = "Single-mode"
        else:
            num_modes = int(V**2 / 2)  # Multimode
            mode_type = "Multi-mode"

        # Mode field diameter (Gaussian approximation for single mode)
        if V < 2.405:
            MFD = 2 * core_radius * (0.65 + 1.619 * V**(-1.5) + 2.879 * V**(-6))
        else:
            MFD = 2 * core_radius  # Approximate for multimode

        # Cutoff wavelength
        lambda_c = 2 * pi * core_radius * NA / 2.405

        return {
            'V_number': V,
            'numerical_aperture': NA,
            'num_modes': num_modes,
            'mode_type': mode_type,
            'mode_field_diameter': MFD,
            'cutoff_wavelength': lambda_c
        }

    def nonlinear_spm_phase(self, power: float, length: float, n2: float,
                           wavelength: float, area_eff: float) -> float:
        """
        Calculate self-phase modulation in nonlinear medium

        Args:
            power: Optical power (W)
            length: Propagation length (m)
            n2: Nonlinear refractive index (m²/W)
            wavelength: Wavelength (m)
            area_eff: Effective mode area (m²)

        Returns:
            Nonlinear phase shift (rad)
        """
        # Nonlinear coefficient
        gamma = 2 * pi * n2 / (wavelength * area_eff)

        # SPM phase shift
        phi_NL = gamma * power * length

        return phi_NL

    def second_harmonic_generation(self, fundamental_power: float, crystal_length: float,
                                  d_eff: float, wavelength: float, beam_radius: float) -> float:
        """
        Calculate second harmonic generation efficiency

        Args:
            fundamental_power: Input power (W)
            crystal_length: Crystal length (m)
            d_eff: Effective nonlinear coefficient (m/V)
            wavelength: Fundamental wavelength (m)
            beam_radius: Beam radius (m)

        Returns:
            Second harmonic power (W)
        """
        # Boyd-Kleinman focusing parameter (optimal ~ 2.84)
        xi = crystal_length / (2 * pi * beam_radius**2 / wavelength)

        # Phase matching efficiency factor
        h = 1.068 if abs(xi - 2.84) < 0.5 else 0.5  # Simplified

        # Conversion efficiency
        eta = 8 * pi**2 * d_eff**2 * crystal_length**2 * h / \
              (epsilon_0 * self.c * wavelength**2 * pi * beam_radius**2)

        # Output power
        P_2w = eta * fundamental_power**2

        return P_2w

    def bragg_grating_reflection(self, wavelengths: np.ndarray, period: float,
                               n_eff: float, delta_n: float, length: float) -> np.ndarray:
        """
        Calculate fiber Bragg grating reflection spectrum

        Args:
            wavelengths: Array of wavelengths (m)
            period: Grating period (m)
            n_eff: Effective refractive index
            delta_n: Index modulation amplitude
            length: Grating length (m)

        Returns:
            Reflection spectrum
        """
        # Bragg wavelength
        lambda_B = 2 * n_eff * period

        # Detuning
        delta = pi * n_eff * (1/wavelengths - 1/lambda_B)

        # Coupling coefficient
        kappa = pi * delta_n / lambda_B

        # Reflection coefficient
        gamma = np.sqrt(kappa**2 - delta**2 + 0j)
        R = np.abs(kappa * np.sinh(gamma * length) /
                  (delta * np.sinh(gamma * length) + 1j * gamma * np.cosh(gamma * length)))**2

        return np.real(R)

    def optical_cavity_modes(self, L: float, R1: float, R2: float,
                           wavelength: float, n_modes: int = 5) -> Dict[str, np.ndarray]:
        """
        Calculate optical cavity resonator modes

        Args:
            L: Cavity length (m)
            R1, R2: Mirror radii of curvature (m)
            wavelength: Wavelength (m)
            n_modes: Number of modes to calculate

        Returns:
            Mode frequencies and beam parameters
        """
        # Cavity g-parameters
        g1 = 1 - L / R1 if R1 != np.inf else 1
        g2 = 1 - L / R2 if R2 != np.inf else 1

        # Stability check
        stability = g1 * g2
        is_stable = 0 <= stability <= 1

        if not is_stable:
            warnings.warn("Cavity is unstable!")

        # Free spectral range
        FSR = self.c / (2 * L)

        # Mode frequencies
        q = np.arange(1, n_modes + 1)
        freq_q = q * FSR

        # Gaussian beam parameters at mirrors
        if is_stable:
            # Beam radius at mirrors
            w1_sq = (wavelength * L / pi) * np.sqrt(g2 / (g1 * (1 - g1 * g2)))
            w2_sq = (wavelength * L / pi) * np.sqrt(g1 / (g2 * (1 - g1 * g2)))
            w1 = np.sqrt(np.abs(w1_sq))
            w2 = np.sqrt(np.abs(w2_sq))
        else:
            w1 = w2 = np.nan

        return {
            'mode_numbers': q,
            'frequencies': freq_q,
            'FSR': FSR,
            'is_stable': is_stable,
            'g1': g1,
            'g2': g2,
            'waist_at_M1': w1,
            'waist_at_M2': w2
        }

    def photonic_bandgap(self, n1: float, n2: float, d1: float, d2: float,
                        wavelengths: np.ndarray) -> np.ndarray:
        """
        Calculate 1D photonic crystal transmission

        Args:
            n1, n2: Refractive indices of layers
            d1, d2: Layer thicknesses (m)
            wavelengths: Array of wavelengths (m)

        Returns:
            Transmission spectrum
        """
        transmission = np.zeros_like(wavelengths)

        for i, lam in enumerate(wavelengths):
            # Phase in each layer
            beta1 = 2 * pi * n1 / lam
            beta2 = 2 * pi * n2 / lam

            # Transfer matrix for one period
            M11 = np.cos(beta1 * d1) * np.cos(beta2 * d2) - \
                  (n2/n1 + n1/n2) * np.sin(beta1 * d1) * np.sin(beta2 * d2) / 2
            M12 = 1j * (np.cos(beta1 * d1) * np.sin(beta2 * d2) / n2 + \
                       np.sin(beta1 * d1) * np.cos(beta2 * d2) / n1)

            # Transmission coefficient
            T = 1 / np.abs(M11)**2 if np.abs(M11) > 1 else 1

            transmission[i] = T

        return transmission

    def kerr_lens_mode_locking(self, peak_power: float, n2: float, beam_radius: float,
                              crystal_length: float) -> Dict[str, float]:
        """
        Calculate Kerr lens mode-locking parameters

        Args:
            peak_power: Peak power (W)
            n2: Nonlinear refractive index (m²/W)
            beam_radius: Beam radius (m)
            crystal_length: Crystal length (m)

        Returns:
            Mode-locking parameters
        """
        # Intensity
        I_peak = peak_power / (pi * beam_radius**2)

        # Nonlinear phase shift
        delta_n = n2 * I_peak
        phi_NL = 2 * pi * delta_n * crystal_length / (633e-9)  # Reference wavelength

        # Self-focusing power (critical power)
        lambda_ref = 633e-9
        P_cr = 3.77 * lambda_ref**2 / (8 * pi * n2)

        # Kerr lens strength
        P_ratio = peak_power / P_cr

        # Effective focal length from Kerr lensing
        if P_ratio < 0.99:
            f_kerr = beam_radius**2 * pi / (lambda_ref * P_ratio)
        else:
            f_kerr = np.inf  # Near critical power

        return {
            'nonlinear_index_change': delta_n,
            'nonlinear_phase': phi_NL,
            'critical_power': P_cr,
            'power_ratio': P_ratio,
            'kerr_focal_length': f_kerr
        }

    def soliton_propagation(self, power: float, beta2: float, gamma: float,
                          pulse_width: float) -> Dict[str, float]:
        """
        Calculate optical soliton parameters

        Args:
            power: Peak power (W)
            beta2: Group velocity dispersion (s²/m)
            gamma: Nonlinear coefficient (1/(W·m))
            pulse_width: Pulse duration FWHM (s)

        Returns:
            Soliton parameters
        """
        # Soliton order
        T0 = pulse_width / 1.763  # Convert FWHM to 1/e width
        N = np.sqrt(gamma * power * T0**2 / abs(beta2))

        # Soliton period
        z0 = pi * T0**2 / (2 * abs(beta2))

        # Dispersion length
        L_D = T0**2 / abs(beta2)

        # Nonlinear length
        L_NL = 1 / (gamma * power)

        return {
            'soliton_order': N,
            'soliton_period': z0,
            'dispersion_length': L_D,
            'nonlinear_length': L_NL,
            'is_fundamental': abs(N - 1) < 0.1
        }

    def demonstrate(self):
        """Demonstrate all optics and photonics calculations"""
        print("=" * 70)
        print("OPTICS AND PHOTONICS LABORATORY DEMONSTRATION")
        print("=" * 70)

        # 1. Gaussian beam
        print("\n1. GAUSSIAN BEAM PROPAGATION")
        z = np.linspace(-0.1, 0.1, 100)
        beam = self.gaussian_beam_propagation(z, w0=1e-3, wavelength=1064e-9)
        print(f"   Beam waist: 1 mm")
        print(f"   Rayleigh range: {beam['rayleigh_range']*1000:.2f} mm")
        print(f"   Divergence: {beam['divergence']*1000:.2f} mrad")

        # 2. Fresnel diffraction
        print("\n2. FRESNEL DIFFRACTION")
        x = np.linspace(-5e-3, 5e-3, 200)
        I_fresnel = self.fresnel_diffraction(x, 1e-3, 1.0, 633e-9)
        print(f"   Slit width: 1 mm")
        print(f"   Distance: 1 m")
        print(f"   Central maximum intensity: {np.max(I_fresnel):.3f}")

        # 3. Fabry-Perot cavity
        print("\n3. FABRY-PEROT CAVITY")
        wavelengths = np.linspace(1549e-9, 1551e-9, 1000)
        T = self.fabry_perot_transmission(wavelengths, 0.01, 0.95, 0.95)
        finesse = pi * np.sqrt(0.95) / (1 - 0.95)
        print(f"   Cavity length: 10 mm")
        print(f"   Finesse: {finesse:.1f}")
        print(f"   Max transmission: {np.max(T):.3f}")

        # 4. Mach-Zehnder interferometer
        print("\n4. MACH-ZEHNDER INTERFEROMETER")
        phase = np.linspace(0, 4*pi, 100)
        mz = self.mach_zehnder_interference(phase, visibility=0.98)
        print(f"   Visibility: 98%")
        print(f"   Port 1 max: {np.max(mz['port1']):.3f}")
        print(f"   Port 2 min: {np.min(mz['port2']):.3f}")

        # 5. Fiber optics
        print("\n5. FIBER OPTIC MODES")
        fiber = self.fiber_mode_propagation(1550e-9, 4.5e-6, 1.45, 1.44)
        print(f"   Core radius: 4.5 µm")
        print(f"   V-number: {fiber['V_number']:.3f}")
        print(f"   Mode type: {fiber['mode_type']}")
        print(f"   NA: {fiber['numerical_aperture']:.3f}")

        # 6. Nonlinear optics - SPM
        print("\n6. SELF-PHASE MODULATION")
        phi_spm = self.nonlinear_spm_phase(1.0, 0.1, 2.2e-20, 1550e-9, 80e-12)
        print(f"   Power: 1 W")
        print(f"   Length: 10 cm")
        print(f"   Nonlinear phase: {phi_spm:.2f} rad")

        # 7. Second harmonic generation
        print("\n7. SECOND HARMONIC GENERATION")
        P_shg = self.second_harmonic_generation(1e6, 0.01, 2e-12, 1064e-9, 50e-6)
        efficiency = P_shg / 1e6**2 * 100
        print(f"   Fundamental: 1 MW @ 1064 nm")
        print(f"   Crystal: 1 cm")
        print(f"   SHG power: {P_shg:.2f} W")
        print(f"   Efficiency: {efficiency:.4f}%")

        # 8. Bragg grating
        print("\n8. FIBER BRAGG GRATING")
        wavelengths = np.linspace(1549e-9, 1551e-9, 500)
        R = self.bragg_grating_reflection(wavelengths, 535e-9, 1.45, 1e-3, 0.005)
        print(f"   Bragg wavelength: 1550 nm")
        print(f"   Peak reflectivity: {np.max(R)*100:.1f}%")

        # 9. Optical cavity
        print("\n9. OPTICAL CAVITY MODES")
        cavity = self.optical_cavity_modes(0.3, 1.0, 1.0, 1064e-9, 5)
        print(f"   Length: 30 cm")
        print(f"   FSR: {cavity['FSR']/1e6:.1f} MHz")
        print(f"   Stable: {cavity['is_stable']}")

        # 10. Photonic bandgap
        print("\n10. PHOTONIC CRYSTAL BANDGAP")
        wavelengths = np.linspace(1400e-9, 1700e-9, 300)
        T_pc = self.photonic_bandgap(1.45, 2.3, 267e-9, 168e-9, wavelengths)
        gap_center = wavelengths[np.argmin(T_pc)]
        print(f"   Alternating n=1.45/2.3")
        print(f"   Bandgap center: {gap_center*1e9:.0f} nm")

        print("\n" + "=" * 70)
        print("Demonstration complete. All systems operational.")


if __name__ == "__main__":
    lab = OpticsPhotonicsLab()
    lab.demonstrate()