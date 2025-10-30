
import json
import os
from mendeleev import get_all_elements
from materials_database import MaterialProperties

def create_elemental_database():
    """
    Fetches data for all chemical elements using the mendeleev library,
    transforms it into the MaterialProperties format, and saves it to a JSON file.
    """
    elements_data = {}
    all_elements = get_all_elements()

    print(f"Found {len(all_elements)} elements. Processing...")

    for el in all_elements:
        # Skip elements with insufficient data
        if el.density is None or el.melting_point is None:
            print(f"Skipping {el.name} due to missing density or melting point.")
            continue

        # Map mendeleev fields to our MaterialProperties dataclass
        props = MaterialProperties(
            name=el.name,
            category="element",
            subcategory=el.block + "-block",
            cas_number=el.cas,

            # Mechanical Properties
            density=el.density * 1000,  # g/cm³ to kg/m³
            youngs_modulus=el.modulus_young / 1000 if el.modulus_young else 0.0, # MPa to GPa
            shear_modulus=el.modulus_shear / 1000 if el.modulus_shear else 0.0, # MPa to GPa
            bulk_modulus=el.modulus_bulk / 1000 if el.modulus_bulk else 0.0, # MPa to GPa
            poissons_ratio=el.poisson_ratio if el.poisson_ratio else 0.0,
            hardness_vickers=el.hardness_vickers if el.hardness_vickers else 0.0,

            # Thermal Properties
            melting_point=el.melting_point if el.melting_point else 0.0,
            boiling_point=el.boiling_point if el.boiling_point else 0.0,
            thermal_conductivity=el.thermal_conductivity if el.thermal_conductivity else 0.0,
            specific_heat=el.specific_heat if el.specific_heat else 0.0,

            # Electrical Properties
            electrical_resistivity=el.resistivity if el.resistivity else 0.0,
            
            # Additional metadata
            notes=f"Chemical element, atomic number {el.atomic_number}, symbol {el.symbol}.",
            data_source="mendeleev Python library",
            confidence=0.95
        )
        elements_data[el.name] = props.to_dict()

    # Define the output path
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "elements.json")

    # Write to JSON file
    with open(output_path, 'w') as f:
        json.dump(elements_data, f, indent=2, sort_keys=True)

    print(f"\nSuccessfully processed {len(elements_data)} elements.")
    print(f"Saved elemental database to: {output_path}")


if __name__ == "__main__":
    # Note: You'll need to install mendeleev first:
    # pip install mendeleev
    create_elemental_database()
