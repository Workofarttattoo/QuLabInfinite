import pytest
import sys
import os
import numpy as np

# Ensure the module can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from cancer_drug_quantum_discovery_ENHANCED import cancer_drug_hamiltonian_v2

class MockQuantumState:
    def __init__(self, n_qubits, probabilities):
        self.n_qubits = n_qubits
        self._probs = probabilities

    def get_probabilities(self):
        return self._probs

def test_cancer_drug_hamiltonian_v2_deterministic():
    """Test that the function yields deterministic outputs given the same state."""
    # State with 4 qubits, uniform distribution over 16 states
    probs = [1.0/16] * 16
    state = MockQuantumState(n_qubits=4, probabilities=probs)

    result1 = cancer_drug_hamiltonian_v2(state, target_id=0)
    result2 = cancer_drug_hamiltonian_v2(state, target_id=0)

    assert result1 == result2

def test_cancer_drug_hamiltonian_v2_target_wrap():
    """Test that target_id wraps around using modulo 5."""
    probs = [0.1, 0.2, 0.3, 0.4]
    state = MockQuantumState(n_qubits=2, probabilities=probs)

    # target 0 and target 5 should use the same parameters
    result0 = cancer_drug_hamiltonian_v2(state, target_id=0)
    result5 = cancer_drug_hamiltonian_v2(state, target_id=5)

    assert result0 == result5

def test_cancer_drug_hamiltonian_v2_different_targets():
    """Test that different target_ids produce different energies for the same state."""
    # Give all weight to state '11'
    probs = [0.0, 0.0, 0.0, 1.0]
    state = MockQuantumState(n_qubits=2, probabilities=probs)

    result0 = cancer_drug_hamiltonian_v2(state, target_id=0)
    result1 = cancer_drug_hamiltonian_v2(state, target_id=1)

    assert result0 != result1

def test_cancer_drug_hamiltonian_v2_low_probability_skipped():
    """Test that states with probabilities below 1e-10 are skipped and contribute 0."""
    probs1 = [1e-11, 0.0, 0.0, 0.0]
    state1 = MockQuantumState(n_qubits=2, probabilities=probs1)
    result1 = cancer_drug_hamiltonian_v2(state1, target_id=0)
    assert result1 == 0.0

    # To be totally sure it skips, try adding a probability just above 1e-10
    probs2 = [1e-9, 0.0, 0.0, 0.0]
    state2 = MockQuantumState(n_qubits=2, probabilities=probs2)
    result2 = cancer_drug_hamiltonian_v2(state2, target_id=0)
    assert result2 != 0.0

def test_cancer_drug_hamiltonian_v2_manual_calculation_00():
    """Manually trace the calculation for the '00' state and target_id=0."""
    # For target_id=0:
    # torsion: 0.52, hbond: 1.35, clash: 0.75, elec: 0.45
    # n_qubits = 2. config '00' (i=0)
    #
    # Torsional energy:
    # j=0: bit='0' -> + torsion * 0.6 * sin(0) = 0
    # j=1: bit='0' -> + torsion * 0.6 * sin(pi/2) = 0.52 * 0.6 * 1.0 = 0.312
    # Total torsion = 0.312
    #
    # Hydrogen bonding: count('11') = 0 -> 0
    # Steric clashes: count('00') = 1 -> + 0.75 * 1 / 2 = 0.375
    # Electrostatic: alternations = 0 -> 0
    # Hydrophobic: blocks(3..5) = 0 -> 0
    #
    # binding_score = 0.312 + 0.375 = 0.687

    probs = [1.0, 0.0, 0.0, 0.0]
    state = MockQuantumState(n_qubits=2, probabilities=probs)
    result = cancer_drug_hamiltonian_v2(state, target_id=0)

    # Use pytest.approx due to floating point arithmetic
    assert result == pytest.approx(0.687, rel=1e-5)

def test_cancer_drug_hamiltonian_v2_manual_calculation_11():
    """Manually trace the calculation for the '11' state and target_id=0."""
    # For target_id=0:
    # torsion: 0.52, hbond: 1.35, clash: 0.75, elec: 0.45
    # n_qubits = 2. config '11' (i=3)
    #
    # Torsional energy:
    # j=0: bit='1' -> - torsion * cos(0) = -0.52
    # j=1: bit='1' -> - torsion * cos(pi/2) = 0
    # Total torsion = -0.52
    #
    # Hydrogen bonding: count('11') = 1 -> - 1.35 * 1 / 2 = -0.675
    # Steric clashes: count('00') = 0 -> 0
    # Electrostatic: alternations = 0 -> 0
    # Hydrophobic: blocks(3..5) = 0 -> 0
    #
    # binding_score = -0.52 - 0.675 = -1.195

    probs = [0.0, 0.0, 0.0, 1.0]
    state = MockQuantumState(n_qubits=2, probabilities=probs)
    result = cancer_drug_hamiltonian_v2(state, target_id=0)

    assert result == pytest.approx(-1.195, rel=1e-5)
