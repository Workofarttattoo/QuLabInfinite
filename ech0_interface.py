#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ECH0 Interface to QuLabInfinite
Provides ECH0 with access to:
- Materials database (1,059 materials)
- Quantum computing simulation
- Physics simulation
- Chemistry simulation
- Multi-modal analysis
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from pathlib import Path


class ECH0_QuLabInterface:
    """
    Unified interface for ECH0 to access QuLabInfinite capabilities.

    Designed for autonomous agent use with simple, high-level methods.
    """

    def __init__(self):
        """Initialize interface with lazy loading of components."""
        self._materials_db = None
        self._quantum_lab = None
        self._physics_core = None
        self._chemistry_lab = None

    # ========== MATERIALS DATABASE ==========

    @property
    def materials_db(self):
        """Lazy load materials database."""
        if self._materials_db is None:
            from materials_lab.materials_database import MaterialsDatabase
            self._materials_db = MaterialsDatabase()
        return self._materials_db

    def find_material(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Find material by name.

        Args:
            name: Material name (e.g., '304 Stainless Steel', 'Graphene')

        Returns:
            Material properties dict or None if not found
        """
        mat = self.materials_db.get_material(name)
        return mat.to_dict() if mat else None

    def search_materials(self,
                        category: Optional[str] = None,
                        min_strength: Optional[float] = None,
                        max_density: Optional[float] = None,
                        max_cost: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Search materials by criteria.

        Args:
            category: Material category (metal, ceramic, polymer, composite, nanomaterial)
            min_strength: Minimum tensile strength (MPa)
            max_density: Maximum density (kg/m³)
            max_cost: Maximum cost ($/kg)

        Returns:
            List of matching materials
        """
        results = []

        for name, mat in self.materials_db.materials.items():
            # Check category
            if category and mat.category != category:
                continue

            # Check strength
            if min_strength and mat.tensile_strength < min_strength:
                continue

            # Check density
            if max_density and mat.density > max_density:
                continue

            # Check cost
            if max_cost and mat.cost_per_kg > max_cost:
                continue

            results.append({
                'name': name,
                'category': mat.category,
                'density': mat.density,
                'tensile_strength': mat.tensile_strength,
                'thermal_conductivity': mat.thermal_conductivity,
                'cost_per_kg': mat.cost_per_kg,
                'availability': mat.availability
            })

        return results

    def recommend_material(self,
                          application: str,
                          constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Recommend best material for an application.

        Args:
            application: One of 'aerospace', 'thermal', 'electrical', 'structural', 'cost_sensitive'
            constraints: Dict with keys like 'max_density', 'min_strength', 'max_cost'

        Returns:
            Dict with recommended material and reasoning
        """
        constraints = constraints or {}

        if application == 'aerospace':
            # High strength-to-weight ratio
            candidates = self.search_materials(
                min_strength=constraints.get('min_strength', 500),
                max_density=constraints.get('max_density', 5000)
            )

            # Sort by strength-to-weight ratio
            candidates = [
                {**c, 'ratio': c['tensile_strength'] / (c['density']/1000)}
                for c in candidates
                if c['tensile_strength'] > 0 and c['density'] > 0
            ]
            candidates.sort(key=lambda x: x['ratio'], reverse=True)

            if candidates:
                best = candidates[0]
                return {
                    'material': best['name'],
                    'reason': f"Best strength-to-weight ratio: {best['ratio']:.1f} MPa/(Mg/m³)",
                    'properties': best
                }

        elif application == 'thermal':
            # High thermal conductivity
            candidates = []
            for name, mat in self.materials_db.materials.items():
                if mat.thermal_conductivity > 100:
                    candidates.append({
                        'name': name,
                        'thermal_conductivity': mat.thermal_conductivity,
                        'cost_per_kg': mat.cost_per_kg
                    })

            candidates.sort(key=lambda x: x['thermal_conductivity'], reverse=True)

            if candidates:
                best = candidates[0]
                return {
                    'material': best['name'],
                    'reason': f"Highest thermal conductivity: {best['thermal_conductivity']:.0f} W/(m·K)",
                    'properties': best
                }

        elif application == 'cost_sensitive':
            # Best performance per dollar
            candidates = self.search_materials(
                min_strength=constraints.get('min_strength', 100)
            )

            candidates = [
                {**c, 'value': c['tensile_strength'] / c['cost_per_kg']}
                for c in candidates
                if c['tensile_strength'] > 0 and c['cost_per_kg'] > 0
            ]
            candidates.sort(key=lambda x: x['value'], reverse=True)

            if candidates:
                best = candidates[0]
                return {
                    'material': best['name'],
                    'reason': f"Best value: {best['value']:.1f} MPa per $/kg",
                    'properties': best
                }

        return {'material': None, 'reason': 'No suitable material found'}

    # ========== QUANTUM COMPUTING ==========

    @property
    def quantum_lab(self):
        """Lazy load quantum lab."""
        if self._quantum_lab is None:
            from quantum_lab.quantum_lab import QuantumLab
            self._quantum_lab = QuantumLab()
        return self._quantum_lab

    def run_quantum_circuit(self,
                           num_qubits: int,
                           gates: List[Tuple[str, List[int]]],
                           measure: bool = True) -> Dict[str, Any]:
        """
        Run a quantum circuit.

        Args:
            num_qubits: Number of qubits (1-30)
            gates: List of (gate_name, qubit_indices) tuples
                   e.g., [('H', [0]), ('CNOT', [0, 1])]
            measure: Whether to measure all qubits

        Returns:
            Dict with measurement results or statevector
        """
        try:
            circuit = self.quantum_lab.create_circuit(num_qubits)

            for gate_name, qubits in gates:
                if gate_name == 'H':
                    circuit.h(*qubits)
                elif gate_name == 'X':
                    circuit.x(*qubits)
                elif gate_name == 'Y':
                    circuit.y(*qubits)
                elif gate_name == 'Z':
                    circuit.z(*qubits)
                elif gate_name == 'CNOT':
                    circuit.cx(*qubits)
                elif gate_name == 'CZ':
                    circuit.cz(*qubits)
                elif gate_name == 'RX':
                    circuit.rx(qubits[0], qubits[1])  # angle, qubit
                elif gate_name == 'RY':
                    circuit.ry(qubits[0], qubits[1])
                elif gate_name == 'RZ':
                    circuit.rz(qubits[0], qubits[1])

            if measure:
                circuit.measure_all()
                result = self.quantum_lab.execute(circuit, shots=1024)
                return {
                    'counts': result.get_counts(),
                    'success': True
                }
            else:
                result = self.quantum_lab.execute(circuit, shots=1)
                return {
                    'statevector': result.get_statevector().tolist(),
                    'success': True
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def quantum_optimization(self,
                           cost_function: str,
                           num_params: int,
                           max_iterations: int = 100) -> Dict[str, Any]:
        """
        Run quantum optimization (VQE or QAOA).

        Args:
            cost_function: Python expression for cost (e.g., "x[0]**2 + x[1]**2")
            num_params: Number of parameters
            max_iterations: Max optimization iterations

        Returns:
            Dict with optimal parameters and final cost
        """
        try:
            # Use quantum lab's VQE functionality
            from quantum_lab.optimization import VQE

            vqe = VQE(num_qubits=num_params)
            result = vqe.optimize(cost_function, max_iter=max_iterations)

            return {
                'optimal_params': result['params'].tolist(),
                'final_cost': float(result['cost']),
                'iterations': result['iterations'],
                'success': True
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    # ========== PHYSICS SIMULATION ==========

    def simulate_mechanics(self,
                          particles: List[Dict[str, Any]],
                          duration: float,
                          timestep: float = 0.001) -> Dict[str, Any]:
        """
        Run mechanical physics simulation.

        Args:
            particles: List of particle dicts with 'mass', 'position', 'velocity'
            duration: Simulation duration (seconds)
            timestep: Integration timestep (seconds)

        Returns:
            Dict with final positions and velocities
        """
        try:
            from physics_engine.mechanics import MechanicsEngine, Particle

            engine = MechanicsEngine()

            # Add particles
            for p_dict in particles:
                p = Particle(
                    mass=p_dict['mass'],
                    position=np.array(p_dict['position']),
                    velocity=np.array(p_dict['velocity']),
                    force=np.zeros(3),
                    radius=p_dict.get('radius', 0.1)
                )
                engine.add_particle(p)

            # Run simulation
            steps = int(duration / timestep)
            for _ in range(steps):
                engine.step(timestep)

            # Extract results
            results = []
            for p in engine.particles:
                results.append({
                    'position': p.position.tolist(),
                    'velocity': p.velocity.tolist()
                })

            return {
                'particles': results,
                'total_energy': engine.total_energy(),
                'success': True
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    # ========== CHEMISTRY SIMULATION ==========

    def molecular_properties(self, smiles: str) -> Dict[str, Any]:
        """
        Calculate molecular properties from SMILES string.

        Args:
            smiles: SMILES molecular representation

        Returns:
            Dict with molecular properties
        """
        try:
            from chemistry_lab.chemistry_lab import ChemistryLab

            lab = ChemistryLab()
            props = lab.molecular_properties(smiles)

            return {
                'molecular_weight': props.get('molecular_weight'),
                'num_atoms': props.get('num_atoms'),
                'num_bonds': props.get('num_bonds'),
                'logP': props.get('logP'),
                'success': True
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    # ========== UTILITY METHODS ==========

    def get_capabilities(self) -> Dict[str, List[str]]:
        """
        Get list of all available capabilities.

        Returns:
            Dict mapping capability categories to method names
        """
        return {
            'materials': [
                'find_material',
                'search_materials',
                'recommend_material'
            ],
            'quantum': [
                'run_quantum_circuit',
                'quantum_optimization'
            ],
            'physics': [
                'simulate_mechanics'
            ],
            'chemistry': [
                'molecular_properties'
            ]
        }

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about available data.

        Returns:
            Dict with database statistics
        """
        stats = {
            'total_materials': len(self.materials_db.materials),
            'categories': {}
        }

        for name, mat in self.materials_db.materials.items():
            cat = mat.category
            stats['categories'][cat] = stats['categories'].get(cat, 0) + 1

        return stats


# ========== CONVENIENCE FUNCTIONS FOR ECH0 ==========

def ech0_analyze_material(material_name: str) -> str:
    """
    Quick material analysis for ECH0.
    Returns human-readable summary.
    """
    interface = ECH0_QuLabInterface()
    mat = interface.find_material(material_name)

    if not mat:
        return f"Material '{material_name}' not found in database."

    summary = f"Material Analysis: {mat['name']}\n"
    summary += f"Category: {mat['category']}\n"
    summary += f"Density: {mat['density']:,.0f} kg/m³\n"

    if mat['tensile_strength'] > 0:
        summary += f"Tensile Strength: {mat['tensile_strength']:,.0f} MPa\n"

    if mat['thermal_conductivity'] > 0:
        summary += f"Thermal Conductivity: {mat['thermal_conductivity']:,.1f} W/(m·K)\n"

    if mat['cost_per_kg'] > 0:
        summary += f"Cost: ${mat['cost_per_kg']:.2f}/kg\n"

    summary += f"Availability: {mat['availability']}\n"

    return summary


def ech0_design_selector(application: str, budget_per_kg: float = 100.0) -> str:
    """
    Material design selector for ECH0.
    Returns recommendation with reasoning.
    """
    interface = ECH0_QuLabInterface()

    result = interface.recommend_material(
        application=application,
        constraints={'max_cost': budget_per_kg}
    )

    if result['material']:
        summary = f"Recommended Material: {result['material']}\n"
        summary += f"Reasoning: {result['reason']}\n"
        summary += f"\nKey Properties:\n"
        for key, value in result['properties'].items():
            if key not in ['name', 'reason', 'ratio', 'value']:
                summary += f"  {key}: {value}\n"
        return summary
    else:
        return f"No suitable material found for {application} application."


# ========== EXAMPLE USAGE ==========

if __name__ == "__main__":
    print("="*70)
    print("  ECH0 INTERFACE TO QULAB INFINITE")
    print("="*70)

    interface = ECH0_QuLabInterface()

    # Test materials database
    print("\n1. Database Statistics:")
    stats = interface.get_database_stats()
    print(f"   Total materials: {stats['total_materials']}")
    print(f"   Categories: {list(stats['categories'].keys())}")

    # Test material search
    print("\n2. Search for aerospace materials:")
    materials = interface.search_materials(
        min_strength=1000,
        max_density=3000
    )
    print(f"   Found {len(materials)} materials")
    for mat in materials[:3]:
        print(f"   - {mat['name']}: {mat['tensile_strength']:.0f} MPa, {mat['density']:.0f} kg/m³")

    # Test recommendation
    print("\n3. Material recommendation for aerospace:")
    rec = interface.recommend_material('aerospace')
    print(f"   {rec['material']}")
    print(f"   {rec['reason']}")

    # Test convenience functions
    print("\n4. ECH0 Material Analysis:")
    analysis = ech0_analyze_material('Graphene (Single Layer)')
    print(analysis)

    print("\n✅ ECH0 Interface Ready!")
