#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

quick_test - Part of Quantum Lab
"""

"""
Quick test of quantum laboratory functionality
"""

import sys
sys.path.insert(0, '/Users/noone/QuLabInfinite/quantum_lab')

from quantum_lab import QuantumLabSimulator, create_bell_pair
from quantum_chemistry import Molecule

print("="*60)
print("QUANTUM LABORATORY QUICK TEST")
print("="*60)

# Test 1: Basic simulator
print("\n1️⃣  Basic Simulator")
lab = QuantumLabSimulator(num_qubits=5, verbose=False)
lab.h(0).cnot(0, 1)
print("   ✅ 5-qubit simulator operational")

# Test 2: Bell state
print("\n2️⃣  Bell State")
bell = create_bell_pair(verbose=False)
probs = bell.get_probabilities()
print(f"   Probabilities: |00⟩={probs.get('00', 0):.3f}, |11⟩={probs.get('11', 0):.3f}")
print("   ✅ Bell state created")

# Test 3: Chemistry
print("\n3️⃣  Quantum Chemistry")
h2 = Molecule.hydrogen_molecule(bond_length=0.74)
print(f"   H₂ electrons: {h2.num_electrons}")
orbitals = lab.chemistry.molecular_orbitals(h2)
print(f"   HOMO-LUMO gap: {orbitals['gap']:.4f} Ha")
print("   ✅ Chemistry module operational")

# Test 4: Materials
print("\n4️⃣  Quantum Materials")
gap = lab.materials.compute_band_gap('silicon')
print(f"   Silicon band gap: {gap:.3f} eV")
print("   ✅ Materials module operational")

# Test 5: Sensors
print("\n5️⃣  Quantum Sensors")
sens = lab.sensors.magnetometry_sensitivity(num_qubits=5)
print(f"   Magnetometry: {sens*1e15:.2f} fT/√Hz")
print("   ✅ Sensors module operational")

print("\n" + "="*60)
print("✅ ALL MODULES OPERATIONAL!")
print("="*60)
