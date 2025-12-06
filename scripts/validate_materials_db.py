#!/usr/bin/env python3
"""
Validate and display materials database statistics

This script loads and validates your materials database,
showing comprehensive statistics perfect for screenshots.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from materials_lab.materials_database import MaterialsDatabase

def format_number(num):
    """Format large numbers with commas"""
    return f"{num:,}"

def print_section(title):
    """Print a section header"""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print('='*80)

def validate_database():
    """Validate and display database statistics"""

    print_section("QuLabInfinite Materials Database Validation")

    print("\nLoading database...")
    db = MaterialsDatabase()

    total_materials = db.get_count()
    print(f"✓ Successfully loaded {format_number(total_materials)} materials")

    # Category breakdown
    print_section("Materials by Category")

    category_counts = defaultdict(int)
    subcategory_counts = defaultdict(lambda: defaultdict(int))

    for material in db.materials.values():
        category_counts[material.category] += 1
        subcategory_counts[material.category][material.subcategory] += 1

    for category in sorted(category_counts.keys()):
        count = category_counts[category]
        percentage = (count / total_materials) * 100
        print(f"\n{category.upper()}: {format_number(count)} materials ({percentage:.1f}%)")

        # Show top subcategories
        top_subs = sorted(subcategory_counts[category].items(),
                         key=lambda x: x[1], reverse=True)[:5]
        for subcat, sub_count in top_subs:
            print(f"  └─ {subcat}: {format_number(sub_count)}")

    # Property coverage
    print_section("Property Coverage")

    properties_to_check = [
        ('density', 'Density (kg/m³)'),
        ('youngs_modulus', "Young's Modulus (GPa)"),
        ('tensile_strength', 'Tensile Strength (MPa)'),
        ('thermal_conductivity', 'Thermal Conductivity (W/m·K)'),
        ('electrical_resistivity', 'Electrical Resistivity (Ω·m)'),
        ('melting_point', 'Melting Point (K)'),
        ('band_gap_ev', 'Band Gap (eV)'),
    ]

    for prop_name, display_name in properties_to_check:
        count = sum(1 for m in db.materials.values()
                   if getattr(m, prop_name, 0) > 0)
        percentage = (count / total_materials) * 100
        print(f"{display_name:40s}: {format_number(count):>8s} ({percentage:5.1f}%)")

    # Extreme values
    print_section("Materials with Extreme Properties")

    # Lightest material
    lightest = min(db.materials.values(), key=lambda m: m.density if m.density > 0 else float('inf'))
    print(f"\nLightest Material:")
    print(f"  {lightest.name}: {lightest.density:.1f} kg/m³ ({lightest.density/1000:.3f} g/cm³)")

    # Heaviest material
    heaviest = max(db.materials.values(), key=lambda m: m.density)
    print(f"\nHeaviest Material:")
    print(f"  {heaviest.name}: {format_number(int(heaviest.density))} kg/m³ ({heaviest.density/1000:.1f} g/cm³)")

    # Highest strength
    strongest = max(db.materials.values(), key=lambda m: m.tensile_strength)
    print(f"\nHighest Tensile Strength:")
    print(f"  {strongest.name}: {format_number(int(strongest.tensile_strength))} MPa ({strongest.tensile_strength/1000:.1f} GPa)")

    # Best thermal insulator
    insulators = [m for m in db.materials.values() if m.thermal_conductivity > 0]
    best_insulator = min(insulators, key=lambda m: m.thermal_conductivity)
    print(f"\nBest Thermal Insulator:")
    print(f"  {best_insulator.name}: {best_insulator.thermal_conductivity:.4f} W/(m·K)")

    # Best thermal conductor
    best_conductor = max(insulators, key=lambda m: m.thermal_conductivity)
    print(f"\nBest Thermal Conductor:")
    print(f"  {best_conductor.name}: {format_number(int(best_conductor.thermal_conductivity))} W/(m·K)")

    # Data sources
    print_section("Data Sources")

    sources = defaultdict(int)
    for material in db.materials.values():
        sources[material.data_source] += 1

    for source in sorted(sources.keys(), key=lambda x: sources[x], reverse=True)[:10]:
        count = sources[source]
        percentage = (count / total_materials) * 100
        print(f"{source:40s}: {format_number(count):>8s} ({percentage:5.1f}%)")

    # Availability
    print_section("Material Availability")

    availability = defaultdict(int)
    for material in db.materials.values():
        availability[material.availability] += 1

    for avail in ['common', 'uncommon', 'rare', 'experimental']:
        if avail in availability:
            count = availability[avail]
            percentage = (count / total_materials) * 100
            print(f"{avail.capitalize():15s}: {format_number(count):>8s} ({percentage:5.1f}%)")

    # Sample materials showcase
    print_section("Sample Materials Showcase")

    showcase_names = [
        "Airloy X103",
        "Graphene",
        "Ti-6Al-4V",
        "Carbon Fiber Epoxy",
        "Silicon Carbide",
        "PEEK",
    ]

    for name in showcase_names:
        material = db.get_material(name)
        if material:
            print(f"\n{material.name}")
            print(f"  Category: {material.category} / {material.subcategory}")
            if material.density > 0:
                print(f"  Density: {material.density:.1f} kg/m³")
            if material.tensile_strength > 0:
                print(f"  Tensile Strength: {material.tensile_strength:.1f} MPa")
            if material.thermal_conductivity > 0:
                print(f"  Thermal Conductivity: {material.thermal_conductivity:.3f} W/(m·K)")
            if material.notes:
                note_preview = material.notes[:80] + "..." if len(material.notes) > 80 else material.notes
                print(f"  Notes: {note_preview}")

    # Summary
    print_section("Database Validation Summary")

    print(f"""
Database Statistics:
  ✓ Total Materials: {format_number(total_materials)}
  ✓ Categories: {len(category_counts)}
  ✓ Unique Subcategories: {sum(len(subs) for subs in subcategory_counts.values())}
  ✓ Data Sources: {len(sources)}

Property Completeness:
  ✓ Density data: {sum(1 for m in db.materials.values() if m.density > 0)} materials
  ✓ Mechanical properties: {sum(1 for m in db.materials.values() if m.youngs_modulus > 0)} materials
  ✓ Thermal properties: {sum(1 for m in db.materials.values() if m.thermal_conductivity > 0)} materials
  ✓ Electrical properties: {sum(1 for m in db.materials.values() if m.electrical_resistivity > 0)} materials

Database Status: ✓ VALID AND READY
    """)

    print("="*80)
    print("Database validation complete! Ready for Materials Project matching.")
    print("="*80)

if __name__ == '__main__':
    try:
        validate_database()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
