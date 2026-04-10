#!/usr/bin/env python3
"""
Integrate thermochemical data from NIST into the materials catalog.
"""

import json
import os
from materials_database import MaterialProperties

def integrate_nist_data():
    """Load NIST data, transform it, and add it to the comprehensive catalog."""

    # Path to the new NIST data and the existing catalog
    nist_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "materials", "water.jsonl")
    catalog_path = os.path.join(os.path.dirname(__file__), "data", "comprehensive_materials.json")

    # Load existing catalog
    with open(catalog_path, 'r') as f:
        catalog = json.load(f)
    print(f"Loaded existing catalog with {len(catalog)} materials")

    # Process the NIST data
    if os.path.exists(nist_data_path):
        with open(nist_data_path, 'r') as f:
            water_properties = {
                "name": "Water",
                "category": "natural",
                "subcategory": "liquid",
                "cas_number": "7732-18-5",
                "density": 997,  # at 25°C
                "melting_point": 273.15,
                "boiling_point": 373.15,
                "data_source": "NIST Chemistry WebBook",
                "confidence": 0.98,
                "notes": "Thermochemical data for H2O."
            }
            for line in f:
                record = json.loads(line)
                if "enthalpy_of_formation" in record.get("tags", []) and record.get("enthalpy_j_per_mol") is not None:
                    # Storing as a custom property for now
                    water_properties["enthalpy_of_formation_j_per_mol"] = record["enthalpy_j_per_mol"]
                if "standard_entropy" in record.get("tags", []) and record.get("entropy_j_per_mol_k") is not None:
                    water_properties["standard_entropy_j_per_mol_k"] = record["entropy_j_per_mol_k"]

            # Use the MaterialProperties class for validation and structure
            props = MaterialProperties(**water_properties)

            if props.name not in catalog:
                catalog[props.name] = props.to_dict()
                print(f"Added '{props.name}' to the catalog.")
            else:
                print(f"'{props.name}' already exists in the catalog. Updating.")
                catalog[props.name].update(props.to_dict())


    # Save updated catalog
    with open(catalog_path, 'w') as f:
        json.dump(catalog, f, indent=2, sort_keys=True)

    print(f"\n✅ Catalog updated successfully!")
    print(f"   Total materials: {len(catalog)}")

if __name__ == "__main__":
    integrate_nist_data()
