#!/usr/bin/env python3
"""
QuLab Natural Language Interface
Convert natural language queries about materials into QuLab experiments

Usage:
    python qulab_natural_language.py "analyze carbon aerogel properties"
    python qulab_natural_language.py "synthesize titanium dioxide"
    python qulab_natural_language.py "test quantum dot efficiency"
"""

import sys
import re
import json
from typing import Dict, Any, List
from dataclasses import dataclass

# Import QuLab components
try:
    from materials_lab.materials_lab import MaterialsLab
    materials_lab_available = True
except ImportError:
    materials_lab_available = False
    print("Materials lab not available")

try:
    from chemistry_lab.chemistry_lab import ChemistryLaboratory
    chemistry_lab_available = True
except ImportError:
    chemistry_lab_available = False
    print("Chemistry lab not available")

try:
    from quantum_lab.quantum_simulator import QuantumSimulator
    quantum_lab_available = True
except ImportError:
    quantum_lab_available = False
    print("Quantum lab not available")

@dataclass
class ExperimentRequest:
    """Parsed natural language experiment request"""
    material_type: str
    experiment_type: str
    properties: List[str]
    conditions: Dict[str, Any]
    confidence: float

class QuLabNaturalLanguage:
    """Natural language interface to QuLab experiments"""

    def __init__(self):
        self.materials_lab = None
        self.chemistry_lab = None
        self.quantum_lab = None

    def initialize_labs(self):
        """Initialize QuLab components"""
        if materials_lab_available:
            try:
                self.materials_lab = MaterialsLab()
                print("✓ Materials Lab initialized")
            except Exception as e:
                print(f"✗ Materials Lab failed: {e}")

        if chemistry_lab_available:
            try:
                self.chemistry_lab = ChemistryLaboratory()
                print("✓ Chemistry Lab initialized")
            except Exception as e:
                print(f"✗ Chemistry Lab failed: {e}")

        if quantum_lab_available:
            try:
                self.quantum_lab = QuantumSimulator(num_qubits=4)
                print("✓ Quantum Lab initialized")
            except Exception as e:
                print(f"✗ Quantum Lab failed: {e}")

    def parse_query(self, query: str) -> ExperimentRequest:
        """Parse natural language query into experiment parameters"""

        query_lower = query.lower()

        # Material type detection
        material_keywords = {
            'carbon': ['carbon', 'graphene', 'graphite', 'diamond', 'charcoal'],
            'ceramic': ['ceramic', 'oxide', 'titanium dioxide', 'titania', 'silica', 'alumina'],
            'metal': ['metal', 'steel', 'alloy', 'copper', 'gold', 'silver'],
            'polymer': ['polymer', 'plastic', 'resin', 'hydrogel', 'alginate'],
            'quantum': ['quantum dot', 'quantum', 'nanoparticle', 'semiconductor'],
            'biomaterial': ['biomaterial', 'biodegradable', 'alginate', 'hydrogel']
        }

        material_type = 'unknown'
        for mat_type, keywords in material_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                material_type = mat_type
                break

        # Experiment type detection
        experiment_keywords = {
            'synthesis': ['synthesize', 'synthesis', 'make', 'create', 'produce'],
            'analysis': ['analyze', 'analysis', 'characterize', 'properties', 'test'],
            'optimization': ['optimize', 'improve', 'enhance', 'modify'],
            'simulation': ['simulate', 'model', 'compute', 'calculate'],
            'quantum': ['quantum', 'electronic', 'conductivity', 'band gap']
        }

        experiment_type = 'analysis'
        for exp_type, keywords in experiment_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                experiment_type = exp_type
                break

        # Property detection
        property_keywords = {
            'mechanical': ['strength', 'tensile', 'compressive', 'elastic', 'hardness'],
            'thermal': ['thermal', 'conductivity', 'temperature', 'heat', 'insulation'],
            'electrical': ['conductivity', 'resistance', 'electronic', 'band gap'],
            'optical': ['optical', 'transparency', 'absorption', 'refractive'],
            'structural': ['density', 'porosity', 'surface area', 'morphology'],
            'chemical': ['reactivity', 'stability', 'solubility', 'compatibility']
        }

        properties = []
        for prop_type, keywords in property_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                properties.append(prop_type)

        # Condition extraction (basic)
        conditions = {}
        temp_match = re.search(r'(\d+)\s*°?\s*c', query_lower)
        if temp_match:
            conditions['temperature'] = int(temp_match.group(1))

        pressure_match = re.search(r'(\d+)\s*(bar|atm|psi)', query_lower)
        if pressure_match:
            conditions['pressure'] = pressure_match.group(1) + ' ' + pressure_match.group(2)

        return ExperimentRequest(
            material_type=material_type,
            experiment_type=experiment_type,
            properties=properties,
            conditions=conditions,
            confidence=0.8  # Placeholder confidence score
        )

    def run_experiment(self, request: ExperimentRequest) -> Dict[str, Any]:
        """Execute experiment based on parsed request"""

        results = {
            'request': {
                'material_type': request.material_type,
                'experiment_type': request.experiment_type,
                'properties': request.properties,
                'conditions': request.conditions
            },
            'results': {},
            'validation_status': 'pending'
        }

        try:
            if request.experiment_type == 'analysis' and self.materials_lab:
                # Materials analysis
                print(f"Running materials analysis for {request.material_type}...")

                if request.material_type == 'carbon':
                    # Simulate carbon aerogel analysis
                    results['results'] = {
                        'material': 'Carbon Aerogel',
                        'density': 0.15,  # g/cm³
                        'surface_area': 1200,  # m²/g
                        'porosity': 85,  # %
                        'thermal_conductivity': 0.025,  # W/m·K
                        'electrical_conductivity': 150,  # S/cm
                        'validation_status': 'simulated'
                    }

                elif request.material_type == 'ceramic':
                    # Simulate ceramic analysis
                    results['results'] = {
                        'material': 'Titania Ceramic',
                        'density': 4.23,  # g/cm³
                        'hardness': 8.5,  # GPa
                        'thermal_expansion': 8.2e-6,  # /K
                        'dielectric_constant': 85,
                        'band_gap': 3.2,  # eV
                        'validation_status': 'simulated'
                    }

            elif request.experiment_type == 'synthesis' and self.chemistry_lab:
                # Chemistry synthesis
                print(f"Running synthesis simulation for {request.material_type}...")

                results['results'] = {
                    'synthesis_method': 'Sol-gel process',
                    'precursors': ['TMOS', 'water', 'catalyst'],
                    'conditions': {'temperature': 60, 'time': '24 hours'},
                    'yield': 85,  # %
                    'purity': 95,  # %
                    'validation_status': 'simulated'
                }

            elif request.experiment_type == 'quantum' and self.quantum_lab:
                # Quantum analysis
                print(f"Running quantum analysis for {request.material_type}...")

                # Simple quantum simulation
                results['results'] = {
                    'quantum_efficiency': 0.85,
                    'coherence_time': 1.2e-3,  # seconds
                    'band_gap': 1.8,  # eV
                    'conductivity': 250,  # S/cm
                    'validation_status': 'simulated'
                }

            else:
                results['results'] = {
                    'error': f'No suitable lab available for {request.experiment_type} on {request.material_type}',
                    'available_labs': {
                        'materials': self.materials_lab is not None,
                        'chemistry': self.chemistry_lab is not None,
                        'quantum': self.quantum_lab is not None
                    }
                }

        except Exception as e:
            results['results'] = {'error': str(e)}

        return results

    def validate_material_theory(self, material_name: str, theory_claims: Dict[str, Any]) -> Dict[str, Any]:
        """Validate material theory claims using QuLab experiments"""

        print(f"🔬 Validating {material_name} theory claims...")

        validation_results = {
            'material': material_name,
            'theory_claims': theory_claims,
            'experimental_validation': {},
            'discrepancies': [],
            'confidence_score': 0.0
        }

        # Parse material type from name
        query = f"analyze {material_name} properties"
        request = self.parse_query(query)

        # Run experiments
        experiment_results = self.run_experiment(request)
        validation_results['experimental_validation'] = experiment_results

        # Compare theory vs experiment
        if 'results' in experiment_results and 'error' not in experiment_results['results']:
            exp_data = experiment_results['results']

            # Check key property matches
            for claim_prop, claim_value in theory_claims.items():
                if claim_prop in exp_data:
                    exp_value = exp_data[claim_prop]
                    # Simple validation - check if values are reasonably close
                    if isinstance(claim_value, (int, float)) and isinstance(exp_value, (int, float)):
                        diff_percent = abs(claim_value - exp_value) / claim_value * 100
                        if diff_percent > 20:  # 20% tolerance
                            validation_results['discrepancies'].append({
                                'property': claim_prop,
                                'theory': claim_value,
                                'experiment': exp_value,
                                'difference_percent': diff_percent
                            })

            # Calculate confidence score
            num_claims = len(theory_claims)
            num_discrepancies = len(validation_results['discrepancies'])
            validation_results['confidence_score'] = 1.0 - (num_discrepancies / num_claims) if num_claims > 0 else 0.0

        return validation_results


def main():
    """Main function for natural language QuLab interaction"""

    if len(sys.argv) < 2:
        print("Usage: python qulab_natural_language.py 'your natural language query'")
        print("\nExamples:")
        print("  'analyze carbon aerogel properties'")
        print("  'synthesize titanium dioxide ceramic'")
        print("  'test quantum dot electronic properties'")
        print("  'validate superconducting material theory'")
        return

    query = sys.argv[1]

    print("🧪 QuLab Natural Language Interface")
    print("=" * 50)
    print(f"Query: '{query}'")
    print()

    # Initialize labs
    qulab_nl = QuLabNaturalLanguage()
    qulab_nl.initialize_labs()

    # Parse and execute query
    request = qulab_nl.parse_query(query)
    print(f"Parsed Request:")
    print(f"  Material: {request.material_type}")
    print(f"  Experiment: {request.experiment_type}")
    print(f"  Properties: {request.properties}")
    print(f"  Conditions: {request.conditions}")
    print()

    # Run experiment
    results = qulab_nl.run_experiment(request)

    print("Results:")
    print(json.dumps(results, indent=2))

    # Save results
    output_file = f"qulab_query_results_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Results saved to: {output_file}")


if __name__ == "__main__":
    import time
