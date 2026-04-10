#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Build a comprehensive materials catalog from multiple sources:
- Chemical elements (mendeleev)
- Common engineering materials (hardcoded database)
- arXiv materials data (optional)
"""

import json
import os
from materials_database import MaterialProperties

# Common engineering materials with real-world properties
ENGINEERING_MATERIALS = {
    "Steel-304-Stainless": {
        "name": "304 Stainless Steel",
        "category": "metal",
        "subcategory": "stainless_steel",
        "cas_number": "12597-68-1",
        "density": 8000.0,  # kg/m³
        "youngs_modulus": 193.0,  # GPa
        "shear_modulus": 77.0,  # GPa
        "poissons_ratio": 0.29,
        "tensile_strength": 515.0,  # MPa
        "yield_strength": 205.0,  # MPa
        "elongation_at_break": 40.0,  # %
        "hardness_rockwell": "HRB 92",
        "melting_point": 1673.0,  # K
        "thermal_conductivity": 16.2,  # W/(m·K)
        "specific_heat": 500.0,  # J/(kg·K)
        "thermal_expansion": 17.3e-6,  # 1/K
        "electrical_resistivity": 7.2e-7,  # Ω·m
        "corrosion_resistance": "excellent",
        "cost_per_kg": 3.5,  # USD/kg
        "availability": "common",
        "data_source": "ASTM A240, MatWeb",
        "confidence": 0.98
    },
    "Aluminum-6061-T6": {
        "name": "6061-T6 Aluminum Alloy",
        "category": "metal",
        "subcategory": "aluminum_alloy",
        "density": 2700.0,  # kg/m³
        "youngs_modulus": 68.9,  # GPa
        "shear_modulus": 26.0,  # GPa
        "poissons_ratio": 0.33,
        "tensile_strength": 310.0,  # MPa
        "yield_strength": 276.0,  # MPa
        "elongation_at_break": 12.0,  # %
        "hardness_rockwell": "HRB 60",
        "melting_point": 855.0,  # K
        "boiling_point": 2743.0,  # K
        "thermal_conductivity": 167.0,  # W/(m·K)
        "specific_heat": 896.0,  # J/(kg·K)
        "thermal_expansion": 23.6e-6,  # 1/K
        "electrical_resistivity": 3.7e-8,  # Ω·m
        "corrosion_resistance": "good",
        "cost_per_kg": 2.8,  # USD/kg
        "availability": "common",
        "data_source": "ASM Metals Handbook",
        "confidence": 0.98
    },
    "Titanium-Ti6Al4V": {
        "name": "Ti-6Al-4V Titanium Alloy",
        "category": "metal",
        "subcategory": "titanium_alloy",
        "density": 4430.0,  # kg/m³
        "youngs_modulus": 113.8,  # GPa
        "shear_modulus": 44.0,  # GPa
        "poissons_ratio": 0.342,
        "tensile_strength": 950.0,  # MPa
        "yield_strength": 880.0,  # MPa
        "elongation_at_break": 14.0,  # %
        "hardness_rockwell": "HRC 36",
        "melting_point": 1878.0,  # K
        "thermal_conductivity": 6.7,  # W/(m·K)
        "specific_heat": 526.3,  # J/(kg·K)
        "thermal_expansion": 8.6e-6,  # 1/K
        "electrical_resistivity": 1.7e-6,  # Ω·m
        "corrosion_resistance": "excellent",
        "cost_per_kg": 35.0,  # USD/kg
        "availability": "common",
        "data_source": "ASTM B265, MatWeb",
        "confidence": 0.98
    },
    "Copper-C11000": {
        "name": "C11000 Electrolytic Tough Pitch Copper",
        "category": "metal",
        "subcategory": "copper",
        "density": 8940.0,  # kg/m³
        "youngs_modulus": 117.0,  # GPa
        "shear_modulus": 45.0,  # GPa
        "poissons_ratio": 0.34,
        "tensile_strength": 220.0,  # MPa
        "yield_strength": 69.0,  # MPa
        "elongation_at_break": 45.0,  # %
        "hardness_rockwell": "HRB 40",
        "melting_point": 1358.0,  # K
        "boiling_point": 2835.0,  # K
        "thermal_conductivity": 391.0,  # W/(m·K)
        "specific_heat": 385.0,  # J/(kg·K)
        "thermal_expansion": 16.5e-6,  # 1/K
        "electrical_resistivity": 1.7e-8,  # Ω·m
        "corrosion_resistance": "good",
        "cost_per_kg": 9.5,  # USD/kg
        "availability": "common",
        "data_source": "ASM Metals Handbook",
        "confidence": 0.98
    },
    "Concrete-Standard": {
        "name": "Standard Portland Cement Concrete",
        "category": "ceramic",
        "subcategory": "concrete",
        "density": 2400.0,  # kg/m³
        "youngs_modulus": 30.0,  # GPa
        "poissons_ratio": 0.2,
        "compressive_strength": 30.0,  # MPa (typical)
        "tensile_strength": 3.0,  # MPa
        "thermal_conductivity": 1.4,  # W/(m·K)
        "specific_heat": 880.0,  # J/(kg·K)
        "thermal_expansion": 10.0e-6,  # 1/K
        "corrosion_resistance": "moderate",
        "cost_per_kg": 0.12,  # USD/kg
        "availability": "common",
        "data_source": "ACI 318",
        "confidence": 0.95
    },
    "Glass-Soda-Lime": {
        "name": "Soda-Lime Glass",
        "category": "ceramic",
        "subcategory": "glass",
        "density": 2500.0,  # kg/m³
        "youngs_modulus": 69.0,  # GPa
        "shear_modulus": 28.0,  # GPa
        "poissons_ratio": 0.23,
        "tensile_strength": 50.0,  # MPa
        "compressive_strength": 1000.0,  # MPa
        "hardness_vickers": 550.0,  # HV
        "melting_point": 1673.0,  # K
        "glass_transition_temp": 846.0,  # K
        "thermal_conductivity": 1.05,  # W/(m·K)
        "specific_heat": 840.0,  # J/(kg·K)
        "thermal_expansion": 9.0e-6,  # 1/K
        "electrical_resistivity": 1e10,  # Ω·m
        "refractive_index": 1.52,
        "transmittance": 90.0,  # %
        "corrosion_resistance": "excellent",
        "cost_per_kg": 0.5,  # USD/kg
        "availability": "common",
        "data_source": "MatWeb, Glass Properties",
        "confidence": 0.95
    },
    "Silicon-Nitride": {
        "name": "Silicon Nitride (Si3N4)",
        "category": "ceramic",
        "subcategory": "advanced_ceramic",
        "cas_number": "12033-89-5",
        "density": 3200.0,  # kg/m³
        "youngs_modulus": 310.0,  # GPa
        "shear_modulus": 125.0,  # GPa
        "poissons_ratio": 0.27,
        "tensile_strength": 700.0,  # MPa
        "compressive_strength": 3000.0,  # MPa
        "fracture_toughness": 6.0,  # MPa·m^0.5
        "hardness_vickers": 1600.0,  # HV
        "melting_point": 2173.0,  # K
        "thermal_conductivity": 30.0,  # W/(m·K)
        "specific_heat": 700.0,  # J/(kg·K)
        "thermal_expansion": 3.2e-6,  # 1/K
        "max_service_temp": 1473.0,  # K
        "electrical_resistivity": 1e14,  # Ω·m
        "corrosion_resistance": "excellent",
        "cost_per_kg": 50.0,  # USD/kg
        "availability": "uncommon",
        "data_source": "CeramTec, MatWeb",
        "confidence": 0.96
    },
    "Graphene": {
        "name": "Graphene (Single Layer)",
        "category": "nanomaterial",
        "subcategory": "2d_material",
        "cas_number": "7782-42-5",
        "density": 2267.0,  # kg/m³ (bulk graphite, monolayer ~0.77 mg/m²)
        "youngs_modulus": 1000.0,  # GPa (in-plane)
        "tensile_strength": 130000.0,  # MPa
        "fracture_toughness": 4.0,  # MPa·m^0.5
        "thermal_conductivity": 5000.0,  # W/(m·K)
        "specific_heat": 700.0,  # J/(kg·K)
        "thermal_expansion": -7.0e-6,  # 1/K (negative!)
        "electrical_resistivity": 1e-8,  # Ω·m
        "refractive_index": 2.6,
        "cost_per_kg": 100000.0,  # USD/kg (research grade)
        "availability": "rare",
        "data_source": "Nature Materials, 2D Materials",
        "confidence": 0.90,
        "notes": "Exceptional 2D material with highest known strength and thermal conductivity"
    },
    "Carbon-Fiber-T300": {
        "name": "T300 Carbon Fiber Composite",
        "category": "composite",
        "subcategory": "carbon_fiber",
        "density": 1760.0,  # kg/m³
        "youngs_modulus": 230.0,  # GPa (fiber direction)
        "tensile_strength": 3530.0,  # MPa
        "elongation_at_break": 1.5,  # %
        "thermal_conductivity": 7.0,  # W/(m·K)
        "specific_heat": 710.0,  # J/(kg·K)
        "thermal_expansion": -0.7e-6,  # 1/K (fiber direction)
        "electrical_resistivity": 1.6e-5,  # Ω·m
        "cost_per_kg": 25.0,  # USD/kg
        "availability": "common",
        "data_source": "Toray Industries",
        "confidence": 0.96
    },
    "Kevlar-49": {
        "name": "Kevlar 49 Aramid Fiber",
        "category": "polymer",
        "subcategory": "aramid",
        "density": 1440.0,  # kg/m³
        "youngs_modulus": 112.4,  # GPa
        "tensile_strength": 3620.0,  # MPa
        "elongation_at_break": 2.8,  # %
        "melting_point": 773.0,  # K (decomposes)
        "thermal_conductivity": 0.04,  # W/(m·K)
        "specific_heat": 1420.0,  # J/(kg·K)
        "thermal_expansion": -2.0e-6,  # 1/K
        "cost_per_kg": 30.0,  # USD/kg
        "availability": "common",
        "data_source": "DuPont",
        "confidence": 0.96,
        "notes": "High strength-to-weight ratio, used in body armor and aerospace"
    }
}


def build_catalog():
    """Build comprehensive materials catalog from multiple sources."""

    catalog = {}

    # Load elements from previously generated file
    elements_path = os.path.join(os.path.dirname(__file__), "data", "elements.json")
    if os.path.exists(elements_path):
        with open(elements_path, 'r') as f:
            elements = json.load(f)
        catalog.update(elements)
        print(f"Loaded {len(elements)} elements from {elements_path}")

    # Add engineering materials
    for key, mat_data in ENGINEERING_MATERIALS.items():
        props = MaterialProperties(**mat_data)
        catalog[props.name] = props.to_dict()

    print(f"Added {len(ENGINEERING_MATERIALS)} engineering materials")

    # Write combined catalog
    output_path = os.path.join(os.path.dirname(__file__), "data", "comprehensive_materials.json")
    with open(output_path, 'w') as f:
        json.dump(catalog, f, indent=2, sort_keys=True)

    print(f"\n✅ Comprehensive materials catalog created!")
    print(f"   Total materials: {len(catalog)}")
    print(f"   Elements: {len(catalog) - len(ENGINEERING_MATERIALS)}")
    print(f"   Engineering materials: {len(ENGINEERING_MATERIALS)}")
    print(f"   Output: {output_path}")

    # Print statistics
    categories = {}
    for mat_name, mat_data in catalog.items():
        cat = mat_data.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\n📊 Materials by category:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"   {cat}: {count}")


if __name__ == "__main__":
    build_catalog()
