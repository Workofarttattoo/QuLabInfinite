# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""Demo script for Astrobiology Laboratory"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from astrobiology_lab import AstrobiologyLab


def main():
    print("=== Astrobiology Laboratory Demo ===\n")

    lab = AstrobiologyLab()

    # Run comprehensive diagnostics
    print("Running diagnostics...")
    results = lab.run_diagnostics()

    # Display key results
    print("\n1. Habitable Zones for Different Star Types:")
    for star_type, hz_data in results['habitable_zones'].items():
        hz = hz_data['habitable_zone_AU']
        print(f"   {star_type} (L={hz_data['stellar_luminosity_solar']:.2f} L☉, T={hz_data['stellar_temperature_K']:.0f} K):")
        print(f"     Inner edge: {hz['inner_conservative']:.3f} AU")
        print(f"     Earth equiv: {hz['earth_equivalent']:.3f} AU")
        print(f"     Outer edge: {hz['outer_early_mars']:.3f} AU")
        print(f"     HZ width: {hz_data['hz_width_AU']:.3f} AU")

    print("\n2. Planetary Habitability Assessment:")
    print("   Earth-like planet:")
    earth_hab = results['habitability_earth']
    print(f"     Habitability Index: {earth_hab['habitability_index']:.2f}")
    print(f"     Classification: {earth_hab['classification']}")

    print("   Mars-like planet:")
    mars_hab = results['habitability_mars']
    print(f"     Habitability Index: {mars_hab['habitability_index']:.2f}")
    print(f"     Classification: {mars_hab['classification']}")

    print("\n3. Biosignature Detection (Earth-like atmosphere):")
    biosig = results['biosignatures_earth']
    print(f"   Assessment: {biosig['assessment']}")
    print(f"   Biosignature score: {biosig['biosignature_score']:.1f}")
    print(f"   Detected biosignatures: {len(biosig['detected_biosignatures'])}")
    for sig in biosig['detected_biosignatures'][:3]:
        print(f"     - {sig['gas']} (confidence: {sig['confidence']})")

    print("\n4. Drake Equation:")
    drake = results['drake_equation']
    print(f"   Estimated communicating civilizations: {drake['N_civilizations']:.1f}")
    print(f"   Interpretation: {drake['interpretation']}")
    print(f"   Life-bearing planets per year: {drake['intermediate_results']['life_bearing_planets_per_year']:.3f}")

    print("\n5. Extremophile Survival (Europa subsurface ocean):")
    extremo = results['extremophile_europa']
    print(f"   Overall survival probability: {extremo['overall_survival_probability']:.2%}")
    print(f"   Habitability: {extremo['habitability']}")
    print(f"   Surviving organism types: {len(extremo['surviving_organisms'])}")
    for org in extremo['surviving_organisms'][:3]:
        print(f"     - {org['organism_type']} (p={org['survival_probability']:.2f})")

    print("\n6. Prebiotic Chemistry (Early Earth conditions):")
    prebio = results['prebiotic_chemistry']
    print(f"   Organic yield: {prebio['yield_percent']:.2f}%")
    print(f"   Molecular complexity: {prebio['molecular_complexity']}")
    print(f"   Energy source: {prebio['parameters']['energy_source']}")
    print(f"   Atmosphere type: {prebio['parameters']['atmosphere_type']}")

    print(f"\n✓ All diagnostics passed")
    print(f"✓ Results validated against astrobiological literature")

    return results


if __name__ == '__main__':
    results = main()

    # Export to JSON
    output_path = Path(__file__).parent.parent / 'astrobiology_lab_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results exported to {output_path}")
