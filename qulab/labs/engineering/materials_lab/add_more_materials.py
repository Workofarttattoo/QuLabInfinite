#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Add more common engineering materials to the catalog.
"""

import json
import os
from materials_database import MaterialProperties

# Additional common materials
ADDITIONAL_MATERIALS = {
    "Brass-C26000": {
        "name": "C26000 Cartridge Brass (70% Cu, 30% Zn)",
        "category": "metal",
        "subcategory": "copper_alloy",
        "density": 8530.0,
        "youngs_modulus": 110.0,
        "shear_modulus": 40.0,
        "poissons_ratio": 0.34,
        "tensile_strength": 345.0,
        "yield_strength": 124.0,
        "elongation_at_break": 53.0,
        "melting_point": 1208.0,
        "thermal_conductivity": 120.0,
        "specific_heat": 380.0,
        "thermal_expansion": 20.0e-6,
        "electrical_resistivity": 6.2e-8,
        "corrosion_resistance": "good",
        "cost_per_kg": 7.5,
        "availability": "common",
        "data_source": "ASM Metals Handbook",
        "confidence": 0.97
    },
    "Bronze-C90700": {
        "name": "C90700 Tin Bronze (89% Cu, 10% Sn)",
        "category": "metal",
        "subcategory": "copper_alloy",
        "density": 8800.0,
        "youngs_modulus": 103.0,
        "poissons_ratio": 0.34,
        "tensile_strength": 241.0,
        "yield_strength": 103.0,
        "elongation_at_break": 30.0,
        "hardness_rockwell": "HRB 55",
        "melting_point": 1273.0,
        "thermal_conductivity": 59.0,
        "specific_heat": 377.0,
        "thermal_expansion": 18.0e-6,
        "electrical_resistivity": 1.5e-7,
        "corrosion_resistance": "excellent",
        "cost_per_kg": 12.0,
        "availability": "common",
        "data_source": "ASM Metals Handbook",
        "confidence": 0.97
    },
    "Magnesium-AZ31B": {
        "name": "AZ31B Magnesium Alloy",
        "category": "metal",
        "subcategory": "magnesium_alloy",
        "density": 1770.0,
        "youngs_modulus": 45.0,
        "shear_modulus": 17.0,
        "poissons_ratio": 0.35,
        "tensile_strength": 260.0,
        "yield_strength": 200.0,
        "elongation_at_break": 15.0,
        "hardness_rockwell": "HRB 49",
        "melting_point": 868.0,
        "thermal_conductivity": 96.0,
        "specific_heat": 1020.0,
        "thermal_expansion": 26.0e-6,
        "electrical_resistivity": 9.5e-8,
        "corrosion_resistance": "moderate",
        "cost_per_kg": 4.5,
        "availability": "common",
        "data_source": "ASTM B90/B90M",
        "confidence": 0.96
    },
    "Nickel-200": {
        "name": "Nickel 200 (99% Ni)",
        "category": "metal",
        "subcategory": "nickel_alloy",
        "density": 8890.0,
        "youngs_modulus": 204.0,
        "shear_modulus": 76.0,
        "poissons_ratio": 0.31,
        "tensile_strength": 462.0,
        "yield_strength": 148.0,
        "elongation_at_break": 47.0,
        "hardness_rockwell": "HRB 60",
        "melting_point": 1728.0,
        "boiling_point": 3186.0,
        "thermal_conductivity": 90.7,
        "specific_heat": 444.0,
        "thermal_expansion": 13.3e-6,
        "electrical_resistivity": 9.5e-8,
        "corrosion_resistance": "excellent",
        "cost_per_kg": 18.0,
        "availability": "common",
        "data_source": "ASTM B162",
        "confidence": 0.98
    },
    "Inconel-718": {
        "name": "Inconel 718 Nickel Superalloy",
        "category": "metal",
        "subcategory": "superalloy",
        "density": 8190.0,
        "youngs_modulus": 200.0,
        "shear_modulus": 77.2,
        "poissons_ratio": 0.29,
        "tensile_strength": 1375.0,
        "yield_strength": 1100.0,
        "elongation_at_break": 12.0,
        "hardness_rockwell": "HRC 38",
        "melting_point": 1609.0,
        "thermal_conductivity": 11.4,
        "specific_heat": 435.0,
        "thermal_expansion": 13.0e-6,
        "max_service_temp": 973.0,
        "electrical_resistivity": 1.25e-6,
        "corrosion_resistance": "excellent",
        "cost_per_kg": 45.0,
        "availability": "common",
        "data_source": "Special Metals Corp",
        "confidence": 0.98,
        "notes": "Used in jet engines and gas turbines"
    },
    "Tungsten-Carbide": {
        "name": "Tungsten Carbide (WC-Co)",
        "category": "ceramic",
        "subcategory": "carbide",
        "cas_number": "12070-12-1",
        "density": 14800.0,
        "youngs_modulus": 650.0,
        "shear_modulus": 274.0,
        "poissons_ratio": 0.22,
        "compressive_strength": 6000.0,
        "fracture_toughness": 14.0,
        "hardness_vickers": 1800.0,
        "melting_point": 3058.0,
        "thermal_conductivity": 110.0,
        "specific_heat": 200.0,
        "thermal_expansion": 5.5e-6,
        "electrical_resistivity": 2.0e-7,
        "corrosion_resistance": "excellent",
        "cost_per_kg": 65.0,
        "availability": "common",
        "data_source": "Kennametal, MatWeb",
        "confidence": 0.96,
        "notes": "Extremely hard, used for cutting tools"
    },
    "Sapphire": {
        "name": "Sapphire (Al2O3 Single Crystal)",
        "category": "ceramic",
        "subcategory": "oxide_ceramic",
        "cas_number": "1344-28-1",
        "density": 3980.0,
        "youngs_modulus": 345.0,
        "shear_modulus": 145.0,
        "poissons_ratio": 0.29,
        "tensile_strength": 400.0,
        "compressive_strength": 2000.0,
        "fracture_toughness": 2.3,
        "hardness_vickers": 2300.0,
        "melting_point": 2323.0,
        "thermal_conductivity": 35.0,
        "specific_heat": 750.0,
        "thermal_expansion": 5.3e-6,
        "max_service_temp": 2073.0,
        "electrical_resistivity": 1e16,
        "dielectric_constant": 9.4,
        "refractive_index": 1.77,
        "transmittance": 85.0,
        "corrosion_resistance": "excellent",
        "cost_per_kg": 200.0,
        "availability": "uncommon",
        "data_source": "Kyocera, MatWeb",
        "confidence": 0.96,
        "notes": "Optically transparent, extremely hard"
    },
    "Borosilicate-Glass": {
        "name": "Borosilicate Glass (Pyrex)",
        "category": "ceramic",
        "subcategory": "glass",
        "density": 2230.0,
        "youngs_modulus": 64.0,
        "shear_modulus": 27.0,
        "poissons_ratio": 0.2,
        "tensile_strength": 40.0,
        "compressive_strength": 50.0,
        "melting_point": 1093.0,
        "glass_transition_temp": 820.0,
        "thermal_conductivity": 1.14,
        "specific_heat": 830.0,
        "thermal_expansion": 3.3e-6,
        "max_service_temp": 763.0,
        "electrical_resistivity": 1e13,
        "dielectric_constant": 4.6,
        "refractive_index": 1.47,
        "transmittance": 92.0,
        "corrosion_resistance": "excellent",
        "cost_per_kg": 1.2,
        "availability": "common",
        "data_source": "Corning Inc",
        "confidence": 0.96,
        "notes": "Low thermal expansion, used in lab equipment"
    },
    "PEEK": {
        "name": "PEEK (Polyetheretherketone)",
        "category": "polymer",
        "subcategory": "thermoplastic",
        "density": 1320.0,
        "youngs_modulus": 3.6,
        "poissons_ratio": 0.4,
        "tensile_strength": 100.0,
        "yield_strength": 92.0,
        "elongation_at_break": 50.0,
        "melting_point": 616.0,
        "glass_transition_temp": 416.0,
        "thermal_conductivity": 0.25,
        "specific_heat": 1340.0,
        "thermal_expansion": 47.0e-6,
        "max_service_temp": 533.0,
        "electrical_resistivity": 1e16,
        "dielectric_constant": 3.2,
        "cost_per_kg": 75.0,
        "availability": "common",
        "data_source": "Victrex",
        "confidence": 0.95,
        "notes": "High performance thermoplastic, biocompatible"
    },
    "PTFE": {
        "name": "PTFE (Teflon, Polytetrafluoroethylene)",
        "category": "polymer",
        "subcategory": "fluoropolymer",
        "cas_number": "9002-84-0",
        "density": 2200.0,
        "youngs_modulus": 0.5,
        "poissons_ratio": 0.46,
        "tensile_strength": 23.0,
        "yield_strength": 14.0,
        "elongation_at_break": 300.0,
        "melting_point": 600.0,
        "thermal_conductivity": 0.25,
        "specific_heat": 1000.0,
        "thermal_expansion": 135.0e-6,
        "max_service_temp": 533.0,
        "electrical_resistivity": 1e18,
        "dielectric_constant": 2.1,
        "corrosion_resistance": "excellent",
        "chemical_stability": "highly_reactive",
        "cost_per_kg": 15.0,
        "availability": "common",
        "data_source": "DuPont",
        "confidence": 0.96,
        "notes": "Extremely low friction, chemically inert"
    },
    "Epoxy-Resin": {
        "name": "Epoxy Resin (Typical)",
        "category": "polymer",
        "subcategory": "thermoset",
        "density": 1200.0,
        "youngs_modulus": 3.0,
        "poissons_ratio": 0.38,
        "tensile_strength": 55.0,
        "elongation_at_break": 5.0,
        "glass_transition_temp": 423.0,
        "thermal_conductivity": 0.19,
        "specific_heat": 1600.0,
        "thermal_expansion": 55.0e-6,
        "max_service_temp": 393.0,
        "electrical_resistivity": 1e14,
        "dielectric_constant": 3.6,
        "cost_per_kg": 5.0,
        "availability": "common",
        "data_source": "Hexion, Huntsman",
        "confidence": 0.93,
        "notes": "Used for adhesives and composites"
    },
    "Nylon-6": {
        "name": "Nylon 6 (Polyamide 6)",
        "category": "polymer",
        "subcategory": "polyamide",
        "cas_number": "25038-54-4",
        "density": 1140.0,
        "youngs_modulus": 2.7,
        "poissons_ratio": 0.39,
        "tensile_strength": 75.0,
        "yield_strength": 50.0,
        "elongation_at_break": 90.0,
        "melting_point": 493.0,
        "glass_transition_temp": 320.0,
        "thermal_conductivity": 0.25,
        "specific_heat": 1700.0,
        "thermal_expansion": 80.0e-6,
        "water_absorption": 1.5,
        "electrical_resistivity": 1e13,
        "cost_per_kg": 3.5,
        "availability": "common",
        "data_source": "BASF, DuPont",
        "confidence": 0.94,
        "notes": "Widely used engineering plastic"
    },
    "Carbon-Nanotube-SWCNT": {
        "name": "Single-Walled Carbon Nanotube (SWCNT)",
        "category": "nanomaterial",
        "subcategory": "carbon_nanotube",
        "cas_number": "308068-56-6",
        "density": 1300.0,
        "youngs_modulus": 1000.0,
        "tensile_strength": 50000.0,
        "thermal_conductivity": 3500.0,
        "specific_heat": 700.0,
        "thermal_expansion": 0.0,
        "electrical_resistivity": 1e-5,
        "cost_per_kg": 500000.0,
        "availability": "rare",
        "data_source": "Science, Nature Nanotechnology",
        "confidence": 0.85,
        "notes": "Exceptional properties, research-grade material"
    },
    "Aerogel-Silica": {
        "name": "Silica Aerogel",
        "category": "nanomaterial",
        "subcategory": "aerogel",
        "cas_number": "7631-86-9",
        "density": 100.0,
        "youngs_modulus": 0.001,
        "poissons_ratio": 0.2,
        "compressive_strength": 0.01,
        "melting_point": 1473.0,
        "thermal_conductivity": 0.013,
        "specific_heat": 1000.0,
        "dielectric_constant": 1.02,
        "refractive_index": 1.007,
        "transmittance": 95.0,
        "cost_per_kg": 3000.0,
        "availability": "rare",
        "data_source": "Aspen Aerogels, Cabot",
        "confidence": 0.90,
        "notes": "Lowest density solid, best insulator"
    },
    "Wood-Oak": {
        "name": "Oak Wood (White Oak)",
        "category": "natural",
        "subcategory": "hardwood",
        "density": 750.0,
        "youngs_modulus": 12.0,
        "shear_modulus": 1.0,
        "compressive_strength": 52.0,
        "tensile_strength": 90.0,
        "thermal_conductivity": 0.17,
        "specific_heat": 2400.0,
        "water_absorption": 12.0,
        "cost_per_kg": 2.5,
        "availability": "common",
        "data_source": "Wood Handbook (USDA)",
        "confidence": 0.90,
        "notes": "Natural material, properties vary by specimen"
    },
    "Bamboo": {
        "name": "Bamboo (Moso Bamboo)",
        "category": "natural",
        "subcategory": "grass",
        "density": 700.0,
        "youngs_modulus": 15.0,
        "tensile_strength": 160.0,
        "compressive_strength": 80.0,
        "thermal_conductivity": 0.3,
        "specific_heat": 1600.0,
        "water_absorption": 8.0,
        "cost_per_kg": 1.5,
        "availability": "common",
        "data_source": "INBAR, Materials Science",
        "confidence": 0.88,
        "notes": "Sustainable, high strength-to-weight"
    }
}


def expand_catalog():
    """Add more materials to the comprehensive catalog."""

    # Load existing catalog
    catalog_path = os.path.join(os.path.dirname(__file__), "data", "comprehensive_materials.json")
    with open(catalog_path, 'r') as f:
        catalog = json.load(f)

    print(f"Loaded existing catalog with {len(catalog)} materials")

    # Add new materials
    added_count = 0
    for key, mat_data in ADDITIONAL_MATERIALS.items():
        props = MaterialProperties(**mat_data)
        if props.name not in catalog:
            catalog[props.name] = props.to_dict()
            added_count += 1

    # Save updated catalog
    with open(catalog_path, 'w') as f:
        json.dump(catalog, f, indent=2, sort_keys=True)

    print(f"\n✅ Catalog expanded successfully!")
    print(f"   Total materials: {len(catalog)}")
    print(f"   Added: {added_count} new materials")

    # Print statistics
    categories = {}
    for mat_name, mat_data in catalog.items():
        cat = mat_data.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\n📊 Materials by category:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"   {cat}: {count}")

    # Print subcategories
    subcategories = {}
    for mat_name, mat_data in catalog.items():
        subcat = mat_data.get('subcategory', 'unknown')
        subcategories[subcat] = subcategories.get(subcat, 0) + 1

    print(f"\n📊 Top 10 subcategories:")
    for subcat, count in sorted(subcategories.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {subcat}: {count}")


if __name__ == "__main__":
    expand_catalog()
