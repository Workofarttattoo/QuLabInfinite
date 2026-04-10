# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Semiconductor Lab Demo
Comprehensive demonstrations of all semiconductor physics simulations
"""

import numpy as np
import json
from semiconductor_lab import (
    TransistorPhysics,
    BandStructure,
    DopingAnalysis,
    DeviceSimulation
)


def run_all_demos():
    """Run all semiconductor demonstrations"""
    results = {
        'lab_name': 'Semiconductor Physics Laboratory',
        'demonstrations': {}
    }

    # 1. Transistor Physics
    print("=" * 60)
    print("TRANSISTOR PHYSICS SIMULATION")
    print("=" * 60)

    transistor = TransistorPhysics()

    # MOSFET I-V characteristics
    V_gs_array = np.array([0, 1, 2, 3, 4])
    V_ds_array = np.linspace(0, 5, 50)
    mosfet_result = transistor.mosfet_iv_characteristic(
        V_gs_array=V_gs_array,
        V_ds_array=V_ds_array,
        V_th=1.0,
        mu_n=500,  # cm²/V·s
        C_ox=3.45e-7,  # F/cm²
        W_L_ratio=10
    )
    print(f"MOSFET Characteristics:")
    print(f"Threshold voltage: {mosfet_result['V_th']:.1f} V")
    print(f"Max transconductance: {max(mosfet_result['transconductance_S']):.4f} S")

    results['demonstrations']['mosfet_iv'] = mosfet_result

    # Threshold voltage calculation
    vth_result = transistor.threshold_voltage_calculation(
        oxide_thickness_nm=2.0,
        substrate_doping_cm3=1e17,
        oxide_charge_cm2=1e10,
        temperature_K=300
    )
    print(f"\nThreshold Voltage Analysis:")
    print(f"V_th: {vth_result['threshold_voltage_V']:.3f} V")
    print(f"Flatband voltage: {vth_result['flatband_voltage_V']:.3f} V")
    print(f"Fermi potential: {vth_result['fermi_potential_V']:.3f} V")

    results['demonstrations']['threshold_voltage'] = vth_result

    # BJT Ebers-Moll
    bjt_result = transistor.bjt_ebers_moll(
        V_be=0.7,
        V_bc=-2.0,
        I_s=1e-15,
        beta_f=100,
        temperature_K=300
    )
    print(f"\nBJT Operation:")
    print(f"Operating region: {bjt_result['operating_region']}")
    print(f"Collector current: {bjt_result['collector_current_A']:.2e} A")
    print(f"Current gain β: {bjt_result['current_gain_beta']:.1f}")

    results['demonstrations']['bjt_ebers_moll'] = bjt_result

    # 2. Band Structure
    print("\n" + "=" * 60)
    print("BAND STRUCTURE CALCULATIONS")
    print("=" * 60)

    band = BandStructure()

    # Intrinsic carrier concentration
    temp_array = np.linspace(250, 450, 50)
    carrier_result = band.intrinsic_carrier_concentration(
        temperature_K=temp_array,
        bandgap_eV=1.12,  # Silicon
        m_e_star=1.08,
        m_h_star=0.81
    )
    print(f"Silicon Intrinsic Carriers:")
    print(f"At 300K: {carrier_result['intrinsic_carrier_concentration_per_cm3'][20]:.2e} cm⁻³")
    print(f"At 400K: {carrier_result['intrinsic_carrier_concentration_per_cm3'][-10]:.2e} cm⁻³")

    results['demonstrations']['intrinsic_carriers'] = carrier_result

    # p-n junction
    junction_result = band.pn_junction_built_in_potential(
        N_a_cm3=1e17,
        N_d_cm3=1e16,
        temperature_K=300
    )
    print(f"\np-n Junction:")
    print(f"Built-in potential: {junction_result['built_in_potential_V']:.3f} V")
    print(f"Depletion width: {junction_result['depletion_width_um']:.3f} μm")
    print(f"Junction capacitance: {junction_result['junction_capacitance_F_per_m2']:.2e} F/m²")

    results['demonstrations']['pn_junction'] = junction_result

    # Quantum well
    qw_result = band.quantum_well_energy_levels(
        well_width_nm=10,
        barrier_height_eV=0.3,
        effective_mass_ratio=0.067
    )
    print(f"\nGaAs Quantum Well (10 nm):")
    print(f"Ground state: {qw_result['ground_state_eV']:.3f} eV")
    print(f"Level spacing: {qw_result['level_spacing_eV']:.3f} eV")
    print(f"Energy levels: {[f'{E:.3f}' for E in qw_result['energy_levels_eV']]}")

    results['demonstrations']['quantum_well'] = qw_result

    # 3. Doping Analysis
    print("\n" + "=" * 60)
    print("DOPING ANALYSIS")
    print("=" * 60)

    doping = DopingAnalysis()

    # Ion implantation
    depth_nm = np.linspace(0, 500, 100)
    implant_result = doping.gaussian_implantation_profile(
        depth_nm=depth_nm,
        dose_cm2=1e15,
        projected_range_nm=100,
        straggle_nm=30
    )
    print(f"Ion Implantation Profile:")
    print(f"Peak concentration: {implant_result['peak_concentration_cm3']:.2e} cm⁻³")
    print(f"Junction depth: {implant_result['junction_depth_nm']:.1f} nm")

    results['demonstrations']['ion_implantation'] = implant_result

    # Diffusion profile
    depth_um = np.linspace(0, 5, 100)
    diffusion_result = doping.diffusion_profile(
        depth_um=depth_um,
        surface_concentration_cm3=1e20,
        diffusion_time_hours=4,
        diffusion_coefficient_cm2_per_s=1e-13,
        temperature_K=1273
    )
    print(f"\nDiffusion Profile:")
    print(f"Surface concentration: {diffusion_result['surface_concentration_cm3']:.2e} cm⁻³")
    print(f"Diffusion length: {diffusion_result['diffusion_length_um']:.2f} μm")
    print(f"Junction depth: {diffusion_result['junction_depth_um']:.2f} μm")

    results['demonstrations']['diffusion'] = diffusion_result

    # 4. Device Simulation
    print("\n" + "=" * 60)
    print("DEVICE SIMULATION")
    print("=" * 60)

    device = DeviceSimulation()

    # Small-signal model
    small_signal_result = device.mosfet_small_signal_model(
        I_D=1e-3,
        V_gs=2.0,
        V_th=1.0,
        mu_n=500,
        C_ox=3.45e-7,
        W_L_ratio=10,
        C_gs=1e-15,
        C_gd=0.3e-15
    )
    print(f"MOSFET Small-Signal Parameters:")
    print(f"Transconductance: {small_signal_result['transconductance_S']:.4f} S")
    print(f"Transit frequency: {small_signal_result['transit_frequency_Hz']:.2e} Hz")
    print(f"Voltage gain: {small_signal_result['voltage_gain']:.1f}")

    results['demonstrations']['small_signal'] = small_signal_result

    # Power dissipation
    power_result = device.power_dissipation(
        supply_voltage_V=1.2,
        switching_frequency_Hz=1e9,
        load_capacitance_pF=10,
        leakage_current_nA=100
    )
    print(f"\nCMOS Power Dissipation:")
    print(f"Total power: {power_result['total_power_W']:.6f} W")
    print(f"Dynamic power: {power_result['dynamic_power_W']:.6f} W")
    print(f"Static power: {power_result['static_power_W']:.9f} W")
    print(f"Power density: {power_result['power_density_W_per_MHz']:.6f} W/MHz")

    results['demonstrations']['power_dissipation'] = power_result

    print("\n" + "=" * 60)
    print("SEMICONDUCTOR LAB DEMO COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = run_all_demos()

    # Save results to JSON
    with open('/Users/noone/QuLabInfinite/semiconductor_lab_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to: /Users/noone/QuLabInfinite/semiconductor_lab_results.json")
