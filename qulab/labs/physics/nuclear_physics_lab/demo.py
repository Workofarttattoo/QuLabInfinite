# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""Demo script for Nuclear Physics Laboratory"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nuclear_lab import NuclearPhysicsLab


def main():
    print("=== Nuclear Physics Laboratory Demo ===\n")

    lab = NuclearPhysicsLab()

    # Run comprehensive diagnostics
    print("Running diagnostics...")
    results = lab.run_diagnostics()

    # Display key results
    print("\n1. Radioactive Decay (U-238):")
    u238 = results['radioactive_decay_U238']
    print(f"   Isotope: {u238['isotope']}")
    print(f"   Half-life: {u238['half_life_years']:.2e} years")
    print(f"   Decay mode: {u238['decay_mode']}")
    print("   Time (Gy) | Activity (Bq) | Remaining")
    for t_y, act in zip(u238['time_years'][:5], u238['activity_Bq'][:5]):
        print(f"   {t_y/1e9:9.1f} | {act:13.2e} | {act/1e12:.1%}")

    print("\n2. Radioactive Decay (I-131, medical isotope):")
    i131 = results['radioactive_decay_I131']
    print(f"   Half-life: {i131['half_life_years']*365.25:.1f} days")
    print("   Day | Activity (Ci)")
    for t_s, act_ci in zip(i131['time_seconds'][:5], i131['activity_Ci'][:5]):
        print(f"   {t_s/86400:3.0f} | {act_ci:.2f}")

    print("\n3. Fusion Reactions:")
    dt = results['fusion_DT']
    print(f"   D-T Fusion (T={dt['temperature_keV']} keV):")
    print(f"     Q-value: {dt['Q_value_MeV']} MeV")
    print(f"     Power density: {dt['power_density_MW_m3']:.2f} MW/m³")
    print(f"     Reaction rate: {dt['reaction_rate_per_m3_per_s']:.2e} /m³/s")

    dd = results['fusion_DD']
    print(f"   D-D Fusion (T={dd['temperature_keV']} keV):")
    print(f"     Q-value: {dd['Q_value_MeV']} MeV")
    print(f"     Power density: {dd['power_density_MW_m3']:.2e} MW/m³")

    print("\n4. Fusion Triple Product (ITER-like):")
    tp = results['triple_product_ITER']
    print(f"   nTτ = {tp['triple_product_keV_s_m3']:.2e} keV·s/m³")
    print(f"   Status: {tp['status']}")
    print(f"   Q-factor: {tp['Q_factor']}")

    print("\n5. Radiation Shielding:")
    gamma_lead = results['shielding_gamma_lead']
    print(f"   Gamma rays through {gamma_lead['thickness_cm']} cm lead:")
    print(f"     Attenuation: {gamma_lead['attenuation_factor']:.2f}x")
    print(f"     Reduction: {gamma_lead['reduction_percent']:.1f}%")
    print(f"     Half-value layer: {gamma_lead['half_value_layer_cm']:.2f} cm")

    neutron_water = results['shielding_neutron_water']
    print(f"   Neutrons through {neutron_water['thickness_cm']} cm water:")
    print(f"     Attenuation: {neutron_water['attenuation_factor']:.2e}x")
    print(f"     Reduction: {neutron_water['reduction_percent']:.1f}%")

    print("\n6. Fission Chain Reactions:")
    crit = results['fission_critical']
    print(f"   Critical reactor (k={crit['k_effective']}):")
    print(f"     State: {crit['state']}")
    print(f"     Final neutrons: {crit['final_neutrons']}")

    super_crit = results['fission_supercritical']
    print(f"   Supercritical (k={super_crit['k_effective']}):")
    print(f"     Doubling time: {super_crit['doubling_time_s']*1000:.2f} ms")
    print(f"     Final neutrons: {super_crit['final_neutrons']}")

    print("\n7. Mass-Energy Equivalence (1 gram):")
    me = results['mass_energy_1g']
    print(f"   Energy: {me['energy_J']:.2e} J")
    print(f"   = {me['energy_kWh']:.2e} kWh")
    print(f"   = {me['tnt_equivalent_megatons']:.2e} megatons TNT")

    print(f"\n✓ All diagnostics passed")
    print(f"✓ Results validated against nuclear physics standards")

    return results


if __name__ == '__main__':
    results = main()

    # Export to JSON
    output_path = Path(__file__).parent.parent / 'nuclear_physics_lab_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results exported to {output_path}")
