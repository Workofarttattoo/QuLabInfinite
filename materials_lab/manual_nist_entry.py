#!/usr/bin/env python3
"""
Interactive tool to manually add high-precision NIST data to the database.
"""

import sys
import os
# Adjust path to allow imports if run as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from materials_lab.materials_database import MaterialsDatabase, MaterialProperties

def manual_entry():
    """Prompt user for material properties."""
    print("Manual NIST Data Entry Mode")
    print("---------------------------")

    name = input("Material Name: ").strip()
    if not name: return

    category = input("Category (metal, ceramic, polymer, etc.): ").strip()
    subcategory = input("Subcategory: ").strip()

    try:
        density = float(input("Density (kg/m³): "))
    except ValueError:
        print("Invalid density.")
        return

    try:
        thermal_k = float(input("Thermal Conductivity (W/m·K): "))
    except ValueError:
        thermal_k = 0.0

    try:
        cp = float(input("Specific Heat (J/kg·K): "))
    except ValueError:
        cp = 0.0

    notes = input("Notes/Source (e.g., NIST WebBook): ").strip()

    # Create object with minimal required fields + user inputs
    # MaterialProperties has defaults for most fields
    props = MaterialProperties(
        name=name,
        category=category,
        subcategory=subcategory,
        density=density,
        thermal_conductivity=thermal_k,
        specific_heat=cp,
        notes=notes,
        data_source="manual_nist_entry"
    )

    db = MaterialsDatabase()
    if name in db.materials:
        overwrite = input(f"Material '{name}' exists. Overwrite? (y/n): ").lower()
        if overwrite != 'y':
            print("Aborted.")
            return

    db.materials[name] = props
    db.save()
    print(f"Added {name} to database.")

if __name__ == "__main__":
    manual_entry()
