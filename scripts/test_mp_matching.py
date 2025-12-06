#!/usr/bin/env python3
"""
Quick test of MP matching without needing the full database
Tests formula extraction and matching logic
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from materials_lab.materials_database import MaterialProperties
from scripts.match_materials_to_mp import MaterialMatcher

def test_formula_extraction():
    """Test formula extraction from material names"""
    print("Testing Formula Extraction")
    print("=" * 70)

    test_cases = [
        ("Silicon Carbide", "SiC"),
        ("Alumina 99.5%", "Al2O3"),
        ("Al 6061-T6", "Al"),
        ("Titanium Nitride", "TiN"),
        ("SS 304", None),  # Alloy, no simple formula
        ("PEEK", None),  # Polymer, no formula
        ("Graphene", "C"),
        ("Zirconia 3Y-TZP", "ZrO2"),
        ("Boron Nitride", "BN"),
        ("Silicon Nitride", "Si3N4"),
    ]

    matcher = MaterialMatcher(api_key="dummy")  # Don't need real key for extraction

    passed = 0
    for name, expected in test_cases:
        result = matcher.extract_formula(name)
        status = "✓" if result == expected else "✗"
        passed += (result == expected)
        print(f"{status} {name:30s} -> {result or 'None':10s} (expected: {expected or 'None'})")

    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_manual_mappings():
    """Test manual mappings for common materials"""
    print("\n\nTesting Manual Mappings")
    print("=" * 70)

    matcher = MaterialMatcher(api_key="dummy")

    print("\nManual mappings configured:")
    for name, mp_id in sorted(matcher.manual_mappings.items())[:10]:
        print(f"  {name:30s} -> {mp_id}")

    print(f"\nTotal manual mappings: {len(matcher.manual_mappings)}")


def test_with_sample_materials():
    """Test matching logic with sample materials (no API calls)"""
    print("\n\nTesting Material Matching Logic")
    print("=" * 70)

    matcher = MaterialMatcher(api_key="dummy")

    # Create sample materials
    samples = [
        MaterialProperties(
            name="Silicon Carbide",
            category="ceramic",
            subcategory="carbide",
            density=3210000,  # kg/m³
            youngs_modulus=410,
            band_gap_ev=3.26,
        ),
        MaterialProperties(
            name="Al 6061-T6",
            category="metal",
            subcategory="aluminum_alloy",
            density=2700000,
        ),
        MaterialProperties(
            name="SS 304",
            category="metal",
            subcategory="stainless_steel",
            density=8000000,
        ),
    ]

    for material in samples:
        formula = matcher.extract_formula(material.name)
        is_manual = material.name in matcher.manual_mappings

        print(f"\n{material.name}:")
        print(f"  Formula extracted: {formula or 'None'}")
        if is_manual:
            print(f"  Manual mapping: {matcher.manual_mappings[material.name]}")
        print(f"  Strategy: {'Manual mapping' if is_manual else 'Formula search' if formula else 'No match'}")


if __name__ == '__main__':
    print("Materials Project Matching - Test Suite")
    print("=" * 70)
    print("\nThis test verifies the matching logic without requiring API access.\n")

    success = True
    success &= test_formula_extraction()
    test_manual_mappings()
    test_with_sample_materials()

    print("\n" + "=" * 70)
    if success:
        print("✓ All tests passed!")
        print("\nTo run with actual MP API:")
        print("  1. Set your API key: export MP_API_KEY='your-key'")
        print("  2. Run: python3 scripts/match_materials_to_mp.py --limit 10 --dry-run")
    else:
        print("✗ Some tests failed - check formula extraction logic")

    sys.exit(0 if success else 1)
