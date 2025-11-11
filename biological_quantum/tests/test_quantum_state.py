"""
Tests for True Quantum State Implementation

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import unittest
import numpy as np
import sys
sys.path.append('..')
from core.quantum_state import QuantumState, create_bell_state, create_ghz_state
from core.quantum_gates import apply_hadamard, apply_x, apply_cnot


class TestQuantumState(unittest.TestCase):
    """Test suite for QuantumState class."""

    def test_initialization(self):
        """Test quantum state initialization."""
        # Single qubit initialized to |0⟩
        state = QuantumState(1)
        self.assertEqual(state.n_qubits, 1)
        self.assertEqual(state.dim, 2)
        np.testing.assert_array_almost_equal(state.state_vector, [1+0j, 0+0j])

        # Two qubits initialized to |00⟩
        state2 = QuantumState(2)
        self.assertEqual(state2.dim, 4)
        expected = np.array([1+0j, 0, 0, 0])
        np.testing.assert_array_almost_equal(state2.state_vector, expected)

    def test_normalization(self):
        """Test that states remain normalized after operations."""
        state = QuantumState(2)
        apply_hadamard(state, 0)
        apply_hadamard(state, 1)

        # Check normalization: Σ|αᵢ|² = 1
        norm_squared = np.sum(np.abs(state.state_vector)**2)
        self.assertAlmostEqual(norm_squared, 1.0, places=10)

    def test_hadamard_superposition(self):
        """Test Hadamard creates equal superposition."""
        state = QuantumState(1)
        apply_hadamard(state, 0)

        # H|0⟩ = (|0⟩ + |1⟩)/√2
        expected = np.array([1, 1]) / np.sqrt(2)
        np.testing.assert_array_almost_equal(state.state_vector, expected)

        # Check probabilities are 50/50
        probs = state.get_probabilities()
        np.testing.assert_array_almost_equal(probs, [0.5, 0.5])

    def test_pauli_x(self):
        """Test Pauli-X (NOT) gate."""
        state = QuantumState(1)
        apply_x(state, 0)

        # X|0⟩ = |1⟩
        expected = np.array([0+0j, 1+0j])
        np.testing.assert_array_almost_equal(state.state_vector, expected)

    def test_bell_state_creation(self):
        """Test Bell state creation."""
        bell = create_bell_state("Phi+")

        # |Φ+⟩ = (|00⟩ + |11⟩)/√2
        expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
        np.testing.assert_array_almost_equal(bell.state_vector, expected)

    def test_measurement_collapses_state(self):
        """Test that measurement collapses the state."""
        # Create superposition
        state = QuantumState(1)
        apply_hadamard(state, 0)

        # Measure (should get 0 or 1)
        outcome, collapsed_state = state.measure()
        self.assertIn(outcome, [0, 1])

        # Collapsed state should be deterministic
        if outcome == 0:
            np.testing.assert_array_almost_equal(collapsed_state.state_vector, [1+0j, 0+0j])
        else:
            np.testing.assert_array_almost_equal(collapsed_state.state_vector, [0+0j, 1+0j])

    def test_entanglement(self):
        """Test CNOT creates entanglement."""
        # Prepare |+0⟩ = (|00⟩ + |10⟩)/√2
        state = QuantumState(2)
        apply_hadamard(state, 0)

        # Apply CNOT(0,1) → (|00⟩ + |11⟩)/√2 (Bell state)
        apply_cnot(state, 0, 1)

        # Check probabilities: should only have |00⟩ and |11⟩
        probs = state.get_probabilities()
        np.testing.assert_array_almost_equal(probs, [0.5, 0, 0, 0.5])

    def test_ghz_state(self):
        """Test GHZ state creation."""
        ghz = create_ghz_state(3)

        # |GHZ⟩ = (|000⟩ + |111⟩)/√2
        probs = ghz.get_probabilities()
        expected_probs = np.zeros(8)
        expected_probs[0] = 0.5  # |000⟩
        expected_probs[7] = 0.5  # |111⟩

        np.testing.assert_array_almost_equal(probs, expected_probs)

    def test_fidelity(self):
        """Test fidelity calculation."""
        # Identical states should have fidelity = 1
        state1 = QuantumState(1)
        state2 = QuantumState(1)
        self.assertAlmostEqual(state1.fidelity(state2), 1.0, places=10)

        # Orthogonal states should have fidelity = 0
        apply_x(state2, 0)  # |0⟩ → |1⟩
        self.assertAlmostEqual(state1.fidelity(state2), 0.0, places=10)

    def test_statistical_randomness(self):
        """Test that measurements are truly random."""
        # Create uniform superposition: H^⊗n|0⟩^⊗n
        n_qubits = 3
        state = QuantumState(n_qubits)
        for i in range(n_qubits):
            apply_hadamard(state, i)

        # Measure many times and check distribution
        n_samples = 10000
        outcomes = []
        for _ in range(n_samples):
            outcome, _ = state.measure()
            outcomes.append(outcome)

        # Should have roughly uniform distribution over 2^n states
        n_states = 2**n_qubits
        expected_count = n_samples / n_states

        for i in range(n_states):
            count = outcomes.count(i)
            # Allow 3-sigma deviation
            self.assertTrue(abs(count - expected_count) < 3 * np.sqrt(expected_count))


class TestQuantumGates(unittest.TestCase):
    """Test suite for quantum gates."""

    def test_gates_are_unitary(self):
        """Test that all gates are unitary (U†U = I)."""
        from core.quantum_gates import (
            hadamard, pauli_x, pauli_y, pauli_z,
            phase_gate, rotation_x, cnot, cz, swap
        )

        gates = [
            hadamard(),
            pauli_x(),
            pauli_y(),
            pauli_z(),
            phase_gate(np.pi/4),
            rotation_x(np.pi/3),
            cnot(),
            cz(),
            swap(),
        ]

        for gate in gates:
            # Check U†U = I
            identity = gate.conj().T @ gate
            expected_identity = np.eye(gate.shape[0], dtype=complex)
            np.testing.assert_array_almost_equal(identity, expected_identity)


if __name__ == '__main__':
    unittest.main()
