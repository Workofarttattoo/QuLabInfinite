# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Nanotechnology Lab Demo
Comprehensive demonstrations of all nanotech simulation capabilities
"""

import numpy as np
import json
from nanotechnology_lab import (
    NanoparticleSynthesis,
    QuantumDotSimulator,
    DrugDeliverySystem,
    NanomaterialProperties
)


def run_all_demos():
    """Run all nanotechnology demonstrations"""
    results = {
        'lab_name': 'Nanotechnology Laboratory',
        'demonstrations': {}
    }

    # 1. Nanoparticle Synthesis
    print("=" * 60)
    print("NANOPARTICLE SYNTHESIS SIMULATION")
    print("=" * 60)

    synth = NanoparticleSynthesis()

    # LaMer burst nucleation
    nucleation_result = synth.lamer_burst_nucleation(
        precursor_conc_M=0.01,
        reduction_rate=0.5,
        temperature_K=373,
        time_s=10.0
    )
    print(f"Final particle diameter: {nucleation_result['final_diameter_nm']:.2f} nm")
    print(f"Nucleation time: {nucleation_result['nucleation_time_s']:.4f} s")

    results['demonstrations']['lamer_nucleation'] = nucleation_result

    # Ostwald ripening
    initial_sizes = np.random.normal(5.0, 0.5, 1000)  # 5nm ± 0.5nm distribution
    ripening_result = synth.ostwald_ripening(
        initial_diameters_nm=initial_sizes,
        temperature_K=373,
        time_hours=24
    )
    print(f"\nOstwald Ripening (24 hours):")
    print(f"Initial mean: {ripening_result['initial_mean_diameter_nm']:.2f} nm")
    print(f"Final mean: {ripening_result['final_mean_diameter_nm']:.2f} nm")
    print(f"Growth rate: {ripening_result['growth_rate_nm_per_hour']:.4f} nm/hour")

    results['demonstrations']['ostwald_ripening'] = ripening_result

    # 2. Quantum Dots
    print("\n" + "=" * 60)
    print("QUANTUM DOT SIMULATION")
    print("=" * 60)

    qd_sim = QuantumDotSimulator()

    # CdSe quantum dot (common material)
    brus_result = qd_sim.brus_equation_bandgap(
        radius_nm=2.5,
        bulk_bandgap_eV=1.74,
        electron_mass_ratio=0.13,
        hole_mass_ratio=0.45,
        dielectric_constant=9.5
    )
    print(f"CdSe Quantum Dot (R = 2.5 nm):")
    print(f"Quantum dot bandgap: {brus_result['quantum_dot_bandgap_eV']:.3f} eV")
    print(f"Emission wavelength: {brus_result['emission_wavelength_nm']:.1f} nm")
    print(f"Confinement energy: {brus_result['confinement_energy_eV']:.3f} eV")

    results['demonstrations']['quantum_dot_brus'] = brus_result

    # Density of states
    dos_result = qd_sim.density_of_states(
        radius_nm=2.5,
        electron_mass_ratio=0.13,
        max_n=5
    )
    print(f"\nEnergy levels: {[f'{E:.3f}' for E in dos_result['energy_levels_eV']]} eV")
    print(f"Level spacing: {dos_result['level_spacing_eV']:.3f} eV")

    results['demonstrations']['quantum_dot_dos'] = dos_result

    # 3. Drug Delivery
    print("\n" + "=" * 60)
    print("DRUG DELIVERY SYSTEM")
    print("=" * 60)

    drug_delivery = DrugDeliverySystem()

    # Higuchi release model
    time_hours = np.linspace(0, 48, 100)
    release_result = drug_delivery.higuchi_release_model(
        time_hours=time_hours,
        drug_loading_mg=10.0,
        particle_diameter_nm=100
    )
    print(f"Drug Release Kinetics:")
    print(f"Release at 24h: {release_result['cumulative_release_percent'][50]:.1f}%")
    print(f"Release at 48h: {release_result['cumulative_release_percent'][-1]:.1f}%")

    results['demonstrations']['drug_release'] = release_result

    # Biodistribution
    biodist_result = drug_delivery.biodistribution_model(
        particle_diameter_nm=50,
        dose_mg_per_kg=10.0,
        body_weight_kg=70
    )
    print(f"\nBiodistribution (50 nm particles):")
    print(f"Tumor accumulation: {biodist_result['tumor_percent']:.1f}%")
    print(f"Liver accumulation: {biodist_result['liver_percent']:.1f}%")
    print(f"Kidney clearance: {biodist_result['kidney_percent']:.1f}%")

    results['demonstrations']['biodistribution'] = biodist_result

    # 4. Nanomaterial Properties
    print("\n" + "=" * 60)
    print("NANOMATERIAL PROPERTIES")
    print("=" * 60)

    nano_props = NanomaterialProperties()

    # Surface area
    ssa = nano_props.specific_surface_area(
        diameter_nm=10,
        density_g_per_cm3=19.3  # Gold
    )
    print(f"Gold nanoparticle (10 nm):")
    print(f"Specific surface area: {ssa:.1f} m²/g")

    # Melting point depression
    melting_result = nano_props.melting_point_depression(
        bulk_melting_K=1337,  # Gold
        diameter_nm=5,
        surface_energy_J_per_m2=1.5,
        density_g_per_cm3=19.3
    )
    print(f"\nMelting point depression:")
    print(f"Bulk: {melting_result['bulk_melting_K']:.0f} K")
    print(f"Nano: {melting_result['nano_melting_K']:.0f} K")
    print(f"Depression: {melting_result['depression_K']:.0f} K")

    results['demonstrations']['melting_point'] = melting_result

    # Mechanical properties
    mech_result = nano_props.mechanical_properties(
        diameter_nm=10,
        bulk_youngs_modulus_GPa=200  # Steel
    )
    print(f"\nMechanical properties enhancement:")
    print(f"Bulk modulus: {mech_result['bulk_youngs_modulus_GPa']:.0f} GPa")
    print(f"Nano modulus: {mech_result['nano_youngs_modulus_GPa']:.0f} GPa")
    print(f"Enhancement: {mech_result['enhancement_factor']:.2f}x")

    results['demonstrations']['mechanical_properties'] = mech_result
    results['demonstrations']['surface_area'] = {'specific_surface_area_m2_per_g': ssa}

    print("\n" + "=" * 60)
    print("NANOTECHNOLOGY LAB DEMO COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = run_all_demos()

    # Save results to JSON
    with open('/Users/noone/QuLabInfinite/nanotechnology_lab_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to: /Users/noone/QuLabInfinite/nanotechnology_lab_results.json")
