import logging
#!/usr/bin/env python3
"""
ECH0 Test Materials Digital Twin Characterization
Test version for the first 3 discovered materials
"""

import json
from ech0_digital_twin_characterizer import ECH0_DigitalTwinCharacterizer

def main():
    """Test digital twin characterization on first 3 materials"""

    logging.info("🤖 ECH0 TEST MATERIALS DIGITAL TWIN CHARACTERIZATION")
    logging.info("=" * 55)

    characterizer = ECH0_DigitalTwinCharacterizer()

    # Test with first 3 materials only
    test_materials = [
        {
            'name': 'QCA-2026-DT',
            'category': 'electromagnetic',
            'unit_cell_design': {
                'template': 'quantum_dot_structure',
                'dimensions': {'periodicity': 'nanoscale'},
                'features': ['quantum_effects', 'high_conductivity']
            },
            'material_composition': {
                'graphene_oxide': 0.4,
                'quantum_dots_cdse': 0.1,
                'thiol_polymers': 0.2,
                'cross_linker': 0.06,
                'potassium_hydroxide': 0.24
            },
            'fabrication_method': 'pyrolysis_synthesis'
        },
        {
            'name': 'Ti₃C₂Tₓ-BIO-DT',
            'category': 'mechanical',
            'unit_cell_design': {
                'template': 'composite_matrix',
                'dimensions': {'particle_size': '200-500 nm'},
                'features': ['biocompatible', 'magnetic']
            },
            'material_composition': {
                'titanium': 0.3,
                'aluminum': 0.15,
                'carbon': 0.12,
                'iron_oxide_nanoparticles': 0.06,
                'vancomycin': 0.02,
                'peg_biotin': 0.1
            },
            'fabrication_method': 'etching_functionalization'
        },
        {
            'name': 'PCQD-26-DT',
            'category': 'electromagnetic',
            'unit_cell_design': {
                'template': 'photonic_crystal',
                'dimensions': {'crystal_structure': 'FCC'},
                'features': ['photoluminescent', 'tunable_emission']
            },
            'material_composition': {
                'cadmium_selenide': 0.4,
                'titania': 0.4,
                'oleic_acid': 0.1,
                'phosphonic_acid': 0.1
            },
            'fabrication_method': 'self_assembly_calcination'
        }
    ]

    logging.info(f"Testing {len(test_materials)} materials under standard lab conditions...")

    # Run with just standard lab conditions for testing
    campaign_results = characterizer.run_digital_twin_campaign(
        metamaterial_designs=test_materials,
        conditions=['standard_lab']  # Just one condition for testing
    )

    # Save results
    with open('ech0_test_materials_results.json', 'w') as f:
        json.dump({
            'timestamp': '2026-03-05T00:10:00',
            'materials_tested': len(test_materials),
            'conditions_tested': ['standard_lab'],
            'results': campaign_results
        }, f, indent=2, default=str)

    logging.info("✅ Test completed successfully!")
    logging.info("Results saved to ech0_test_materials_results.json")

if __name__ == "__main__":
    main()