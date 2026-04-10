#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Materials Database Demo
Demonstrates all capabilities of the QuLabInfinite Materials Database
"""

import json
from materials_database import MaterialsDatabase
import numpy as np


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def demo_basic_lookup():
    """Demo: Basic material lookup."""
    print_section("1. Basic Material Lookup")

    db = MaterialsDatabase(db_path='data/comprehensive_materials.json')

    # Look up a material
    steel = db.get_material('304 Stainless Steel')
    if steel:
        print(f"Material: {steel.name}")
        print(f"  Category: {steel.category}")
        print(f"  Density: {steel.density:,.0f} kg/m³")
        print(f"  Tensile Strength: {steel.tensile_strength:,.0f} MPa")
        print(f"  Yield Strength: {steel.yield_strength:,.0f} MPa")
        print(f"  Young's Modulus: {steel.youngs_modulus:,.1f} GPa")
        print(f"  Thermal Conductivity: {steel.thermal_conductivity:,.1f} W/(m·K)")
        print(f"  Cost: ${steel.cost_per_kg:.2f}/kg")


def demo_category_search():
    """Demo: Search by category."""
    print_section("2. Search by Category")

    db = MaterialsDatabase(db_path='data/comprehensive_materials.json')

    # Get all metals
    metals = {name: mat for name, mat in db.materials.items() if mat.category == 'metal'}
    print(f"Found {len(metals)} metal materials:")
    for name in sorted(list(metals.keys())[:10]):
        mat = metals[name]
        strength_str = f"{mat.tensile_strength:.0f} MPa" if mat.tensile_strength > 0 else "N/A"
        print(f"  • {name} - {strength_str} tensile strength")


def demo_property_search():
    """Demo: Search by property range."""
    print_section("3. Search by Property Range")

    db = MaterialsDatabase(db_path='data/comprehensive_materials.json')

    # Find materials with high strength
    print("High-strength materials (tensile > 1000 MPa):")
    count = 0
    for name, mat in db.materials.items():
        if mat.tensile_strength > 1000:
            print(f"  • {name}: {mat.tensile_strength:,.0f} MPa")
            count += 1
    print(f"\nTotal: {count} materials")

    # Find lightweight materials
    print("\nLightweight materials (density < 2000 kg/m³):")
    count = 0
    for name, mat in db.materials.items():
        if mat.density > 0 and mat.density < 2000:
            if mat.tensile_strength > 0:  # Only show materials with strength data
                print(f"  • {name}: {mat.density:,.0f} kg/m³, {mat.tensile_strength:.0f} MPa")
                count += 1
            if count >= 10:
                break


def demo_comparison():
    """Demo: Compare materials."""
    print_section("4. Material Comparison")

    db = MaterialsDatabase(db_path='data/comprehensive_materials.json')

    materials_to_compare = [
        '304 Stainless Steel',
        '6061-T6 Aluminum Alloy',
        'Ti-6Al-4V Titanium Alloy',
        'T300 Carbon Fiber Composite'
    ]

    print(f"{'Material':<35} {'Density':>10} {'Strength':>12} {'Strength/Weight':>15}")
    print(f"{'':35} {'(kg/m³)':>10} {'(MPa)':>12} {'(MPa·m³/kg)':>15}")
    print("-" * 75)

    for mat_name in materials_to_compare:
        mat = db.get_material(mat_name)
        if mat and mat.tensile_strength > 0:
            strength_to_weight = mat.tensile_strength / (mat.density / 1000)  # MPa per (Mg/m³)
            print(f"{mat.name:<35} {mat.density:10,.0f} {mat.tensile_strength:12,.0f} {strength_to_weight:15,.1f}")


def demo_best_for_application():
    """Demo: Find best material for application."""
    print_section("5. Best Material for Application")

    db = MaterialsDatabase(db_path='data/comprehensive_materials.json')

    print("Application: Aerospace structure (high strength-to-weight ratio)\n")

    # Calculate strength-to-weight ratio for all materials
    candidates = []
    for name, mat in db.materials.items():
        if mat.tensile_strength > 500 and mat.density > 0:  # Must have decent strength
            strength_to_weight = mat.tensile_strength / (mat.density / 1000)
            candidates.append((name, strength_to_weight, mat))

    # Sort by strength-to-weight ratio
    candidates.sort(key=lambda x: x[1], reverse=True)

    print(f"Top 10 materials by strength-to-weight ratio:")
    print(f"\n{'Rank':<6} {'Material':<40} {'Ratio':>12} {'Cost':>10}")
    print(f"{'':6} {'':40} {'(kPa·m³/kg)':>12} {'($/kg)':>10}")
    print("-" * 70)

    for i, (name, ratio, mat) in enumerate(candidates[:10], 1):
        cost_str = f"${mat.cost_per_kg:.2f}" if mat.cost_per_kg > 0 else "N/A"
        print(f"{i:<6} {name:<40} {ratio:12,.1f} {cost_str:>10}")


def demo_database_stats():
    """Demo: Database statistics."""
    print_section("6. Database Statistics")

    db = MaterialsDatabase(db_path='data/comprehensive_materials.json')

    print(f"Total materials: {len(db.materials)}")

    # Count by category
    categories = {}
    for name, mat in db.materials.items():
        cat = mat.category
        categories[cat] = categories.get(cat, 0) + 1

    print("\nMaterials by category:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat:20s}: {count:3d}")

    # Availability statistics
    availability = {}
    for name, mat in db.materials.items():
        avail = mat.availability
        availability[avail] = availability.get(avail, 0) + 1

    print("\nMaterials by availability:")
    for avail, count in sorted(availability.items(), key=lambda x: x[1], reverse=True):
        print(f"  {avail:20s}: {count:3d}")

    # Data completeness
    print("\nData completeness:")
    with_strength = sum(1 for m in db.materials.values() if m.tensile_strength > 0)
    with_thermal = sum(1 for m in db.materials.values() if m.thermal_conductivity > 0)
    with_cost = sum(1 for m in db.materials.values() if m.cost_per_kg > 0)

    print(f"  With tensile strength data: {with_strength}/{len(db.materials)} ({100*with_strength/len(db.materials):.1f}%)")
    print(f"  With thermal data: {with_thermal}/{len(db.materials)} ({100*with_thermal/len(db.materials):.1f}%)")
    print(f"  With cost data: {with_cost}/{len(db.materials)} ({100*with_cost/len(db.materials):.1f}%)")


def demo_material_details():
    """Demo: Detailed material information."""
    print_section("7. Detailed Material Information")

    db = MaterialsDatabase(db_path='data/comprehensive_materials.json')

    # Show details for graphene
    mat = db.get_material('Graphene (Single Layer)')
    if mat:
        print(f"Material: {mat.name}")
        print(f"Category: {mat.category} / {mat.subcategory}")
        print(f"\nMechanical Properties:")
        print(f"  Density: {mat.density:,.0f} kg/m³")
        print(f"  Young's Modulus: {mat.youngs_modulus:,.0f} GPa")
        print(f"  Tensile Strength: {mat.tensile_strength:,.0f} MPa")
        print(f"  Fracture Toughness: {mat.fracture_toughness:.1f} MPa·m^0.5")

        print(f"\nThermal Properties:")
        print(f"  Thermal Conductivity: {mat.thermal_conductivity:,.0f} W/(m·K)")
        print(f"  Specific Heat: {mat.specific_heat:,.0f} J/(kg·K)")
        print(f"  Thermal Expansion: {mat.thermal_expansion*1e6:.1f} μm/(m·K)")

        print(f"\nElectrical Properties:")
        print(f"  Electrical Resistivity: {mat.electrical_resistivity:.2e} Ω·m")

        print(f"\nOptical Properties:")
        print(f"  Refractive Index: {mat.refractive_index:.2f}")

        print(f"\nEconomics:")
        print(f"  Cost: ${mat.cost_per_kg:,.0f}/kg")
        print(f"  Availability: {mat.availability}")
        print(f"  Data Source: {mat.data_source}")
        print(f"  Confidence: {mat.confidence*100:.1f}%")

        if mat.notes:
            print(f"\nNotes: {mat.notes}")


def demo_cost_analysis():
    """Demo: Cost analysis."""
    print_section("8. Cost-Performance Analysis")

    db = MaterialsDatabase(db_path='data/comprehensive_materials.json')

    print("Best value materials (strength per dollar):\n")

    # Calculate strength per dollar
    candidates = []
    for name, mat in db.materials.items():
        if mat.tensile_strength > 100 and mat.cost_per_kg > 0:
            strength_per_dollar = mat.tensile_strength / mat.cost_per_kg
            candidates.append((name, strength_per_dollar, mat))

    # Sort by value
    candidates.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Material':<40} {'Strength':>12} {'Cost':>10} {'Value':>15}")
    print(f"{'':40} {'(MPa)':>12} {'($/kg)':>10} {'(MPa/$/kg)':>15}")
    print("-" * 80)

    for i, (name, value, mat) in enumerate(candidates[:10], 1):
        print(f"{name:<40} {mat.tensile_strength:12,.0f} {mat.cost_per_kg:10,.2f} {value:15,.1f}")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("  QULAB INFINITE MATERIALS DATABASE DEMO")
    print("  145 Materials with Comprehensive Real-World Properties")
    print("="*70)

    try:
        demo_basic_lookup()
        demo_category_search()
        demo_property_search()
        demo_comparison()
        demo_best_for_application()
        demo_database_stats()
        demo_material_details()
        demo_cost_analysis()

        print_section("Demo Complete!")
        print("The materials database is fully operational with 145 materials.")
        print("Use MaterialsDatabase API in your code to access all properties.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
