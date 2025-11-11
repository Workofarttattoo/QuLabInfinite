# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Optics Lab Demo
Comprehensive demonstrations of all optics simulation capabilities
"""

import numpy as np
import json
from optics_lab import (
    LaserPhysics,
    PhotonicsSimulator,
    SpectroscopyAnalysis,
    OpticalMaterials
)


def run_all_demos():
    """Run all optics demonstrations"""
    results = {
        'lab_name': 'Optics Laboratory',
        'demonstrations': {}
    }

    # 1. Laser Physics
    print("=" * 60)
    print("LASER PHYSICS SIMULATION")
    print("=" * 60)

    laser = LaserPhysics()

    # Rate equations
    rate_result = laser.rate_equations_steady_state(
        pump_rate_per_s=1e10,  # Increased pump rate
        upper_lifetime_s=230e-6,  # Nd:YAG
        lower_lifetime_s=1e-9,
        cross_section_m2=2.8e-19,  # Corrected cross-section
        cavity_loss=0.05
    )
    print(f"Nd:YAG Laser Operation:")
    print(f"Lasing: {rate_result['lasing']}")
    print(f"Output power: {rate_result['output_power_W']:.3f} W")
    if rate_result['lasing']:
        print(f"Slope efficiency: {rate_result['slope_efficiency']:.3f}")

    results['demonstrations']['laser_rate_equations'] = rate_result

    # Gaussian beam
    z_positions = np.linspace(-10, 10, 100)
    gaussian_result = laser.gaussian_beam_propagation(
        wavelength_nm=1064,
        waist_radius_um=50,
        z_positions_mm=z_positions
    )
    print(f"\nGaussian Beam Propagation:")
    print(f"Waist radius: {gaussian_result['waist_radius_um']:.1f} μm")
    print(f"Rayleigh range: {gaussian_result['rayleigh_range_mm']:.2f} mm")
    print(f"Divergence angle: {gaussian_result['divergence_angle_mrad']:.2f} mrad")

    results['demonstrations']['gaussian_beam'] = gaussian_result

    # Q-switching
    qswitch_result = laser.q_switching_pulse(
        energy_stored_J=0.1,
        cavity_length_m=0.5,
        output_coupling=0.5
    )
    print(f"\nQ-Switched Pulse:")
    print(f"Energy: {qswitch_result['energy_J']:.3f} J")
    print(f"Pulse width: {qswitch_result['pulse_width_ns']:.1f} ns")
    print(f"Peak power: {qswitch_result['peak_power_MW']:.2f} MW")

    results['demonstrations']['qswitched_pulse'] = qswitch_result

    # 2. Photonics
    print("\n" + "=" * 60)
    print("PHOTONICS SIMULATION")
    print("=" * 60)

    photonics = PhotonicsSimulator()

    # Waveguide modes
    waveguide_result = photonics.waveguide_mode_solver(
        core_refractive_index=1.47,
        cladding_refractive_index=1.46,
        core_width_um=8,
        wavelength_nm=1550
    )
    print(f"Optical Waveguide:")
    print(f"V-number: {waveguide_result['V_number']:.2f}")
    print(f"Number of modes: {waveguide_result['number_of_modes']}")
    print(f"Single-mode: {waveguide_result['single_mode']}")
    print(f"Effective index: {waveguide_result['effective_index']:.4f}")

    results['demonstrations']['waveguide_modes'] = waveguide_result

    # Fiber dispersion
    wavelengths = np.linspace(1260, 1625, 50)
    dispersion_result = photonics.fiber_dispersion(
        wavelength_nm=wavelengths,
        zero_dispersion_wavelength_nm=1310
    )
    print(f"\nFiber Chromatic Dispersion:")
    print(f"Zero dispersion: {dispersion_result['zero_dispersion_wavelength_nm']:.0f} nm")

    results['demonstrations']['fiber_dispersion'] = dispersion_result

    # Mach-Zehnder
    mzi_result = photonics.mach_zehnder_interferometer(
        path_difference_um=10,
        wavelength_nm=1550,
        refractive_index=1.47
    )
    print(f"\nMach-Zehnder Interferometer:")
    print(f"Transmission: {mzi_result['transmission']:.3f}")
    print(f"Phase difference: {mzi_result['phase_difference_deg']:.1f}°")
    print(f"FSR: {mzi_result['free_spectral_range_nm']:.2f} nm")

    results['demonstrations']['mach_zehnder'] = mzi_result

    # 3. Spectroscopy
    print("\n" + "=" * 60)
    print("SPECTROSCOPY ANALYSIS")
    print("=" * 60)

    spectro = SpectroscopyAnalysis()

    # Lorentzian lineshape
    freq_Hz = np.linspace(-1e9, 1e9, 200) + 5e14
    lorentz_result = spectro.lorentzian_lineshape(
        frequency_Hz=freq_Hz,
        center_frequency_Hz=5e14,
        linewidth_Hz=1e8
    )
    print(f"Lorentzian Broadening:")
    print(f"Center frequency: {lorentz_result['center_frequency_Hz']:.2e} Hz")
    print(f"FWHM: {lorentz_result['FWHM_Hz']:.2e} Hz")
    print(f"Q-factor: {lorentz_result['Q_factor']:.2e}")

    results['demonstrations']['lorentzian_lineshape'] = lorentz_result

    # Gaussian lineshape
    gaussian_result = spectro.gaussian_lineshape(
        frequency_Hz=freq_Hz,
        center_frequency_Hz=5e14,
        doppler_width_Hz=5e7
    )
    print(f"\nGaussian Doppler Broadening:")
    print(f"FWHM: {gaussian_result['FWHM_Hz']:.2e} Hz")

    results['demonstrations']['gaussian_lineshape'] = gaussian_result

    # Beer-Lambert
    beer_result = spectro.beer_lambert_absorption(
        concentration_M=0.001,
        path_length_cm=1.0,
        molar_absorptivity_L_per_mol_cm=1000
    )
    print(f"\nBeer-Lambert Absorption:")
    print(f"Absorbance: {beer_result['absorbance']:.2f}")
    print(f"Transmittance: {beer_result['transmittance']:.4f}")
    print(f"Percent transmission: {beer_result['percent_transmission']:.2f}%")

    results['demonstrations']['beer_lambert'] = beer_result

    # 4. Optical Materials
    print("\n" + "=" * 60)
    print("OPTICAL MATERIALS")
    print("=" * 60)

    materials = OpticalMaterials()

    # Sellmeier equation (BK7 glass)
    wavelength_um = np.linspace(0.4, 2.0, 50)
    B_coeff = [1.03961212, 0.231792344, 1.01046945]
    C_coeff = [0.00600069867, 0.0200179144, 103.560653]
    sellmeier_result = materials.sellmeier_equation(
        wavelength_um=wavelength_um,
        B_coefficients=B_coeff,
        C_coefficients=C_coeff
    )
    print(f"BK7 Glass Refractive Index:")
    print(f"At 587.6 nm: {sellmeier_result['refractive_index'][20]:.5f}")
    print(f"At 1550 nm: {sellmeier_result['refractive_index'][-10]:.5f}")

    results['demonstrations']['sellmeier_dispersion'] = sellmeier_result

    # Nonlinear refractive index
    nonlinear_result = materials.nonlinear_refractive_index(
        intensity_W_per_cm2=1e10,
        n0=1.5,
        n2_cm2_per_W=3e-16
    )
    print(f"\nOptical Kerr Effect:")
    print(f"Linear index: {nonlinear_result['linear_refractive_index']:.4f}")
    print(f"Total index: {nonlinear_result['total_refractive_index']:.6f}")
    print(f"Critical power: {nonlinear_result['critical_power_W']:.2e} W")

    results['demonstrations']['kerr_effect'] = nonlinear_result

    # Fresnel reflectance
    angles = np.linspace(0, 89, 50)
    fresnel_result = materials.fresnel_reflectance(
        incident_angle_deg=angles,
        n1=1.0,  # Air
        n2=1.5   # Glass
    )
    print(f"\nFresnel Reflectance:")
    print(f"Brewster angle: {fresnel_result['brewster_angle_deg']:.2f}°")

    results['demonstrations']['fresnel_reflectance'] = fresnel_result

    print("\n" + "=" * 60)
    print("OPTICS LAB DEMO COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = run_all_demos()

    # Save results to JSON
    with open('/Users/noone/QuLabInfinite/optics_lab_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to: /Users/noone/QuLabInfinite/optics_lab_results.json")
