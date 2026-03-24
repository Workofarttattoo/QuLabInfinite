import logging
#!/usr/bin/env python3
"""
ECH0 Batch Materials Digital Twin Characterization
Process all 20 discovered materials in batches to avoid output truncation
"""

import json
import os
from datetime import datetime
from ech0_digital_twin_characterizer import ECH0_DigitalTwinCharacterizer

def get_all_material_designs():
    """Get all 20 discovered materials as digital twin designs"""

    return [
        {
            'name': 'QCA-2026-DT',
            'category': 'electromagnetic',
            'unit_cell_design': {
                'template': 'quantum_dot_structure',
                'dimensions': {'periodicity': 'nanoscale', 'layer_thickness': '1-10 nm'},
                'features': ['quantum_effects', 'high_conductivity', 'tunable_properties']
            },
            'material_composition': {
                'graphene_oxide': 0.4,
                'quantum_dots_cdse': 0.1,
                'thiol_polymers': 0.2,
                'cross_linker': 0.06,
                'potassium_hydroxide': 0.24
            },
            'fabrication_method': 'pyrolysis_synthesis',
            'field': 'Energy Storage & Quantum Computing'
        },
        {
            'name': 'Ti₃C₂Tₓ-BIO-DT',
            'category': 'mechanical',
            'unit_cell_design': {
                'template': 'composite_matrix',
                'dimensions': {'particle_size': '200-500 nm', 'matrix_strength': 'high'},
                'features': ['biocompatible', 'magnetic', 'self_healing']
            },
            'material_composition': {
                'titanium': 0.3,
                'aluminum': 0.15,
                'carbon': 0.12,
                'iron_oxide_nanoparticles': 0.06,
                'vancomycin': 0.02,
                'peg_biotin': 0.1
            },
            'fabrication_method': 'etching_functionalization',
            'field': 'Biomedical Engineering'
        },
        {
            'name': 'PCQD-26-DT',
            'category': 'electromagnetic',
            'unit_cell_design': {
                'template': 'photonic_crystal',
                'dimensions': {'crystal_structure': 'FCC', 'refractive_index': '2.1-2.4'},
                'features': ['photoluminescent', 'tunable_emission', 'high_efficiency']
            },
            'material_composition': {
                'cadmium_selenide': 0.4,
                'titania': 0.4,
                'oleic_acid': 0.1,
                'phosphonic_acid': 0.1
            },
            'fabrication_method': 'self_assembly_calcination',
            'field': 'Optoelectronics'
        },
        {
            'name': 'SCGH-26-DT',
            'category': 'electromagnetic',
            'unit_cell_design': {
                'template': 'layered_composite',
                'dimensions': {'thickness': '100 nm', 'critical_field': '50 T'},
                'features': ['high_tc', 'zero_resistance', 'magnetic_field_tolerance']
            },
            'material_composition': {
                'graphene': 0.2,
                'yttrium': 0.15,
                'barium': 0.25,
                'copper_oxide': 0.3,
                'oxygen': 0.1
            },
            'fabrication_method': 'cvd_deposition_annealing',
            'field': 'Superconductivity'
        },
        {
            'name': 'SHCMC-26-DT',
            'category': 'mechanical',
            'unit_cell_design': {
                'template': 'composite_matrix',
                'dimensions': {'reinforcement_ratio': '20%', 'healing_efficiency': '85%'},
                'features': ['self_healing', 'high_toughness', 'thermal_stability']
            },
            'material_composition': {
                'alumina': 0.7,
                'silicon_carbide': 0.2,
                'healing_agent': 0.05,
                'boron_glass': 0.05
            },
            'fabrication_method': 'hot_press_synthesis',
            'field': 'Structural Materials'
        },
        {
            'name': 'NPM-26-DT',
            'category': 'electromagnetic',
            'unit_cell_design': {
                'template': 'organic_electronic',
                'dimensions': {'thickness': '50 nm', 'switching_voltage': '0.5-2.0 V'},
                'features': ['memristive', 'low_power', 'high_endurance']
            },
            'material_composition': {
                'polythiophene': 0.6,
                'titanium_dioxide': 0.3,
                'gold_electrodes': 0.1
            },
            'fabrication_method': 'spin_coating_annealing',
            'field': 'Neuromorphic Computing'
        },
        {
            'name': 'CCMOF-26-DT',
            'category': 'chemical',
            'unit_cell_design': {
                'template': 'porous_framework',
                'dimensions': {'pore_volume': '1.2 cm³/g', 'surface_area': '3500 m²/g'},
                'features': ['high_selectivity', 'regenerable', 'scalable']
            },
            'material_composition': {
                'zinc': 0.15,
                'organic_linker': 0.6,
                'amine_functionality': 0.15,
                'solvent': 0.1
            },
            'fabrication_method': 'solvothermal_synthesis',
            'field': 'Environmental Science'
        },
        {
            'name': 'QDSCE-26-DT',
            'category': 'electromagnetic',
            'unit_cell_design': {
                'template': 'quantum_dot_solar',
                'dimensions': {'bandgap': '1.1-1.4 eV', 'efficiency': '35%'},
                'features': ['broad_spectrum', 'high_efficiency', 'stable']
            },
            'material_composition': {
                'lead_sulfide': 0.3,
                'perovskite': 0.4,
                'hole_transport': 0.2,
                'electron_transport': 0.1
            },
            'fabrication_method': 'layer_by_layer_deposition',
            'field': 'Renewable Energy'
        },
        {
            'name': 'NDDV-26-DT',
            'category': 'chemical',
            'unit_cell_design': {
                'template': 'functional_nanoparticle',
                'dimensions': {'size_distribution': '50-100 nm', 'drug_loading': '90%'},
                'features': ['targeted_delivery', 'biocompatible', 'controlled_release']
            },
            'material_composition': {
                'mesoporous_silica': 0.5,
                'targeting_ligand': 0.1,
                'doxorubicin': 0.15,
                'ph_responsive_polymer': 0.15,
                'peg': 0.1
            },
            'fabrication_method': 'nanoparticle_synthesis',
            'field': 'Nanomedicine'
        },
        {
            'name': 'HTSC-26-DT',
            'category': 'electromagnetic',
            'unit_cell_design': {
                'template': 'ceramic_superconductor',
                'dimensions': {'grain_size': '5 μm', 'current_density': '10^8 A/cm²'},
                'features': ['high_tc', 'high_current', 'magnetic_field_tolerance']
            },
            'material_composition': {
                'yttrium': 0.12,
                'barium': 0.24,
                'copper': 0.32,
                'oxygen': 0.22,
                'doping_elements': 0.1
            },
            'fabrication_method': 'ceramic_sintering',
            'field': 'Energy Transmission'
        },
        {
            'name': 'SASC-26-DT',
            'category': 'mechanical',
            'unit_cell_design': {
                'template': 'smart_concrete',
                'dimensions': {'conductivity': '1 S/m', 'compressive_strength': '150 MPa'},
                'features': ['self_healing', 'sensing', 'high_strength']
            },
            'material_composition': {
                'portland_cement': 0.7,
                'carbon_nanotubes': 0.05,
                'bacterial_spores': 0.01,
                'conductive_polymer': 0.04,
                'aggregates': 0.2
            },
            'fabrication_method': 'concrete_mixing_curing',
            'field': 'Civil Engineering'
        },
        {
            'name': 'QSC-26-DT',
            'category': 'electromagnetic',
            'unit_cell_design': {
                'template': 'diamond_crystal',
                'dimensions': {'nv_concentration': '10^15 cm^-3', 'coherence_time': '1 second'},
                'features': ['quantum_sensing', 'high_sensitivity', 'room_temperature']
            },
            'material_composition': {
                'carbon': 0.99999,
                'nitrogen': 0.000005,
                'vacancy_sites': 0.000005
            },
            'fabrication_method': 'cvd_crystal_growth',
            'field': 'Quantum Sensing'
        },
        {
            'name': 'BDE-26-DT',
            'category': 'electromagnetic',
            'unit_cell_design': {
                'template': 'biodegradable_electronic',
                'dimensions': {'degradation_time': '2-4 weeks', 'transparency': '85%'},
                'features': ['transient', 'biocompatible', 'environmentally_friendly']
            },
            'material_composition': {
                'polylactic_acid': 0.6,
                'magnesium': 0.15,
                'conductive_polymer': 0.15,
                'biodegradable_encapsulant': 0.1
            },
            'fabrication_method': 'biodegradable_printing',
            'field': 'Transient Electronics'
        },
        {
            'name': 'APC-26-DT',
            'category': 'chemical',
            'unit_cell_design': {
                'template': 'catalytic_nanostructure',
                'dimensions': {'surface_area': '150 m²/g', 'efficiency': '15%'},
                'features': ['artificial_photosynthesis', 'hydrogen_production', 'stable']
            },
            'material_composition': {
                'titanium_dioxide': 0.6,
                'cobalt_oxide': 0.15,
                'nickel_hydroxide': 0.15,
                'carbon_support': 0.1
            },
            'fabrication_method': 'hydrothermal_deposition',
            'field': 'Renewable Energy'
        },
        {
            'name': 'NIH-26-DT',
            'category': 'chemical',
            'unit_cell_design': {
                'template': 'hydrogel_network',
                'dimensions': {'elastic_modulus': '1 kPa', 'electrical_conductivity': '10 S/m'},
                'features': ['neural_interface', 'biocompatible', 'conductive']
            },
            'material_composition': {
                'hydrogel_matrix': 0.7,
                'carbon_nanotubes': 0.1,
                'neurotrophic_factors': 0.05,
                'cross_linker': 0.15
            },
            'fabrication_method': 'hydrogel_synthesis',
            'field': 'Neural Engineering'
        },
        {
            'name': 'SRS-26-DT',
            'category': 'mechanical',
            'unit_cell_design': {
                'template': 'aerogel_composite',
                'dimensions': {'density': '0.05 g/cm³', 'porosity': '95%'},
                'features': ['radiation_shielding', 'lightweight', 'thermal_insulation']
            },
            'material_composition': {
                'silica': 0.8,
                'boron_carbide': 0.1,
                'polyethylene_fibers': 0.05,
                'hydrophobic_agent': 0.05
            },
            'fabrication_method': 'aerogel_synthesis',
            'field': 'Space Materials'
        },
        {
            'name': 'SACS-26-DT',
            'category': 'chemical',
            'unit_cell_design': {
                'template': 'single_atom_catalyst',
                'dimensions': {'metal_loading': '0.5 wt%', 'dispersion': '100%'},
                'features': ['single_atom_sites', 'high_efficiency', 'precious_metal_saving']
            },
            'material_composition': {
                'graphene': 0.995,
                'platinum': 0.0005,
                'nitrogen_doping': 0.004,
                'oxygen_functionality': 0.0005
            },
            'fabrication_method': 'atomic_layer_deposition',
            'field': 'Catalysis'
        },
        {
            'name': 'PDON-26-DT',
            'category': 'chemical',
            'unit_cell_design': {
                'template': 'dna_nanostructure',
                'dimensions': {'size': '50 nm', 'yield': '85%'},
                'features': ['programmable', 'targeted_delivery', 'self_assembly']
            },
            'material_composition': {
                'scaffold_dna': 0.3,
                'staple_dna': 0.6,
                'targeting_aptamer': 0.05,
                'therapeutic_payload': 0.05
            },
            'fabrication_method': 'dna_self_assembly',
            'field': 'Nanomedicine'
        },
        {
            'name': 'TISL-26-DT',
            'category': 'electromagnetic',
            'unit_cell_design': {
                'template': 'topological_superlattice',
                'dimensions': {'bulk_bandgap': '0.3 eV', 'layer_thickness': '10 nm'},
                'features': ['topological_insulator', 'quantum_hall', 'spin_momentum_locking']
            },
            'material_composition': {
                'bismuth': 0.3,
                'selenium': 0.25,
                'antimony': 0.25,
                'tellurium': 0.2
            },
            'fabrication_method': 'molecular_beam_epitaxy',
            'field': 'Quantum Materials'
        },
        {
            'name': 'SRPN-26-DT',
            'category': 'chemical',
            'unit_cell_design': {
                'template': 'self_replicating_polymer',
                'dimensions': {'replication_rate': '24 hours', 'adaptability': 'environmental'},
                'features': ['self_replicating', 'adaptive', 'living_material']
            },
            'material_composition': {
                'catalytic_polymer': 0.5,
                'monomer_reservoir': 0.3,
                'environmental_sensors': 0.1,
                'energy_source': 0.1
            },
            'fabrication_method': 'self_assembly_growth',
            'field': 'Smart Materials'
        }
    ]

def process_batch(characterizer, materials_batch, batch_num, total_batches):
    """Process a batch of materials through digital twin characterization"""

    logging.info(f"\n🔬 PROCESSING BATCH {batch_num}/{total_batches}")
    logging.info(f"   Materials in batch: {len(materials_batch)}")
    logging.info(f"   Materials: {[m['name'] for m in materials_batch]}")

    # Run characterization with all 27 stress factors
    batch_results = characterizer.run_digital_twin_campaign(
        metamaterial_designs=materials_batch,
        conditions=None  # Use all 27 stress factors
    )

    # Save batch results
    batch_filename = f'ech0_materials_batch_{batch_num}_results.json'
    with open(batch_filename, 'w') as f:
        json.dump({
            'batch_number': batch_num,
            'timestamp': datetime.now().isoformat(),
            'materials_processed': len(materials_batch),
            'material_names': [m['name'] for m in materials_batch],
            'results': batch_results
        }, f, indent=2, default=str)

    logging.info(f"✅ Batch {batch_num} completed! Results saved to {batch_filename}")

    return batch_results

def combine_batch_results():
    """Combine all batch results into final comprehensive report"""

    logging.info("\n🔗 COMBINING BATCH RESULTS...")

    all_results = []
    batch_files = [f for f in os.listdir('.') if f.startswith('ech0_materials_batch_') and f.endswith('_results.json')]

    for batch_file in sorted(batch_files):
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
            all_results.append(batch_data)

    # Create comprehensive final report
    final_report = {
        'campaign_type': 'Complete 20 Materials Digital Twin Characterization',
        'timestamp': datetime.now().isoformat(),
        'total_materials_processed': sum(len(batch['material_names']) for batch in all_results),
        'total_batches': len(all_results),
        'stress_factors_applied': 27,
        'batch_summaries': [],
        'consolidated_findings': {}
    }

    # Summarize each batch
    all_twin_results = []
    for batch in all_results:
        batch_summary = {
            'batch_number': batch['batch_number'],
            'materials_count': batch['materials_processed'],
            'material_names': batch['material_names'],
            'average_confidence': batch['results']['campaign_summary']['average_confidence'],
            'top_performer': batch['results']['top_performers'][0]['original_design'] if batch['results']['top_performers'] else 'N/A'
        }
        final_report['batch_summaries'].append(batch_summary)

        # Collect all twin results
        all_twin_results.extend(batch['results']['digital_twins_created'])

    # Consolidated findings
    final_report['consolidated_findings'] = {
        'total_digital_twins': len(all_twin_results),
        'overall_average_confidence': sum(t['confidence_level'] for t in all_twin_results) / len(all_twin_results),
        'top_performing_materials': sorted(all_twin_results, key=lambda x: x['confidence_level'], reverse=True)[:5],
        'field_distribution': {},
        'confidence_distribution': {
            'high_confidence': len([t for t in all_twin_results if t['confidence_level'] >= 0.9]),
            'medium_confidence': len([t for t in all_twin_results if 0.7 <= t['confidence_level'] < 0.9]),
            'low_confidence': len([t for t in all_twin_results if t['confidence_level'] < 0.7])
        }
    }

    # Calculate field distribution
    field_counts = {}
    for twin in all_twin_results:
        # Extract field from material name pattern
        name = twin['original_design']
        if 'QCA' in name:
            field = 'Energy Storage & Quantum Computing'
        elif 'Ti₃C₂Tₓ' in name:
            field = 'Biomedical Engineering'
        elif 'PCQD' in name:
            field = 'Optoelectronics'
        elif 'SCGH' in name:
            field = 'Superconductivity'
        elif 'SHCMC' in name:
            field = 'Structural Materials'
        elif 'NPM' in name:
            field = 'Neuromorphic Computing'
        elif 'CCMOF' in name:
            field = 'Environmental Science'
        elif 'QDSCE' in name:
            field = 'Renewable Energy'
        elif 'NDDV' in name:
            field = 'Nanomedicine'
        elif 'HTSC' in name:
            field = 'Energy Transmission'
        elif 'SASC' in name:
            field = 'Civil Engineering'
        elif 'QSC' in name:
            field = 'Quantum Sensing'
        elif 'BDE' in name:
            field = 'Transient Electronics'
        elif 'APC' in name:
            field = 'Renewable Energy'
        elif 'NIH' in name:
            field = 'Neural Engineering'
        elif 'SRS' in name:
            field = 'Space Materials'
        elif 'SACS' in name:
            field = 'Catalysis'
        elif 'PDON' in name:
            field = 'Nanomedicine'
        elif 'TISL' in name:
            field = 'Quantum Materials'
        elif 'SRPN' in name:
            field = 'Smart Materials'
        else:
            field = 'Unknown'

        field_counts[field] = field_counts.get(field, 0) + 1

    final_report['consolidated_findings']['field_distribution'] = field_counts

    # Save final comprehensive report
    with open('ech0_complete_20_materials_digital_twin_results.json', 'w') as f:
        json.dump(final_report, f, indent=2, default=str)

    logging.info("✅ Final comprehensive report created!")
    logging.info("📊 CAMPAIGN SUMMARY:")
    logging.info(f"   • Total materials processed: {final_report['total_materials_processed']}")
    logging.info(f"   • Total digital twins created: {final_report['consolidated_findings']['total_digital_twins']}")
    logging.info(f"   • Overall average confidence: {final_report['consolidated_findings']['overall_average_confidence']:.3f}")
    logging.info(f"   • High confidence twins: {final_report['consolidated_findings']['confidence_distribution']['high_confidence']}")
    logging.info(f"   • Top performer: {final_report['consolidated_findings']['top_performing_materials'][0]['original_design']}")

    return final_report

def main():
    """Run batch processing of all 20 discovered materials"""

    logging.info("🤖 ECH0 COMPLETE 20 MATERIALS DIGITAL TWIN CHARACTERIZATION")
    logging.info("=" * 65)
    logging.info("Processing all 20 revolutionary materials in batches")
    logging.info("Each material tested under 27 comprehensive stress factors")

    # Get all material designs
    all_materials = get_all_material_designs()
    logging.info(f"✅ Loaded {len(all_materials)} material designs")

    # Process in batches of 4 materials each
    batch_size = 4
    total_batches = (len(all_materials) + batch_size - 1) // batch_size  # Ceiling division

    characterizer = ECH0_DigitalTwinCharacterizer()

    for batch_num in range(1, total_batches + 1):
        start_idx = (batch_num - 1) * batch_size
        end_idx = min(start_idx + batch_size, len(all_materials))
        materials_batch = all_materials[start_idx:end_idx]

        process_batch(characterizer, materials_batch, batch_num, total_batches)

    # Combine all batch results
    final_report = combine_batch_results()

    logging.info("\n🎊 COMPLETE 20 MATERIALS DIGITAL TWIN CHARACTERIZATION FINISHED!")
    logging.info("All revolutionary materials tested through 27 stress factors")
    logging.info("Results saved to ech0_complete_20_materials_digital_twin_results.json")

if __name__ == "__main__":
    main()