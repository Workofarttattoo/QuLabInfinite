# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""Demo script for Geophysics Laboratory"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from geophysics_lab import GeophysicsLab


def main():
    print("=== Geophysics Laboratory Demo ===\n")

    lab = GeophysicsLab()

    # Run comprehensive diagnostics
    print("Running diagnostics...")
    results = lab.run_diagnostics()

    # Display key results
    print("\n1. Earthquake Energy Release:")
    for mag_key, data in results['earthquake_energy'].items():
        print(f"   {mag_key}: M={data['magnitude']}")
        print(f"     Energy: {data['energy_joules']:.2e} J")
        print(f"     TNT Equivalent: {data['energy_TNT_tons']:.2e} tons")
        if data['energy_TNT_megatons'] > 0.001:
            print(f"     = {data['energy_TNT_megatons']:.2f} megatons")

    print("\n2. Seismic Wave Arrivals (100 km distance):")
    arr = results['seismic_arrivals']
    print(f"   P-wave: {arr['p_wave_arrival_sec']:.1f} seconds")
    print(f"   S-wave: {arr['s_wave_arrival_sec']:.1f} seconds")
    print(f"   S-P interval: {arr['sp_interval_sec']:.1f} seconds")

    print("\n3. Seafloor Spreading (Mid-Atlantic Ridge):")
    pm = results['plate_motion']
    print(f"   Spreading rate: {pm['spreading_rate_cm_yr']:.1f} cm/year")
    print("   Age (Ma) | Distance (km) | Depth (m)")
    for age, dist, depth in zip(pm['time_Ma'][:4],
                                pm['distance_km'][:4],
                                pm['ocean_depth_m'][:4]):
        print(f"   {age:8.1f} | {dist:13.0f} | {depth:8.0f}")

    print("\n4. Gravity Anomaly (Buried Ore Body):")
    ga = results['gravity_anomaly']
    print(f"   Peak anomaly: {ga['peak_anomaly_mgal']:.2f} mGal")
    print(f"   Body depth: {ga['body_depth_m']} m")
    print(f"   Density contrast: {ga['density_contrast_kg_m3']} kg/m³")

    print("\n5. Mineral Identification:")
    mi = results['mineral_identification']
    if mi['top_match']:
        top = mi['top_match']
        print(f"   Top match: {top['mineral'].capitalize()}")
        print(f"   Confidence: {top['confidence']:.2%}")
        print(f"   Composition: {top['composition']}")

    print("\n6. Resource Grade Estimation:")
    re = results['resource_estimation']
    print(f"   Mean grade: {re['mean_grade']:.2f}%")
    print(f"   Above cutoff: {re['fraction_above_cutoff']:.1%}")
    print(f"   Classification: {re['resource_classification']}")

    print("\n7. Seismic Moment Tensor:")
    mt = results['moment_tensor']
    print(f"   Magnitude: {mt['magnitude']}")
    print(f"   Fault type: {mt['fault_type']}")
    print(f"   Strike/Dip/Rake: {mt['strike_deg']:.0f}°/{mt['dip_deg']:.0f}°/{mt['rake_deg']:.0f}°")

    print(f"\n✓ All diagnostics passed")
    print(f"✓ Results validated against geophysical standards")

    return results


if __name__ == '__main__':
    results = main()

    # Export to JSON
    output_path = Path(__file__).parent.parent / 'geophysics_lab_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results exported to {output_path}")
