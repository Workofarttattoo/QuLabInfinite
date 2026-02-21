"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

QUANTUM COMPUTING LAB
Advanced quantum computing simulations including gate operations, circuit simulation,
Grover's algorithm, quantum error correction, and fidelity calculations.
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable
from dataclasses import dataclass, field
from scipy import linalg
import itertools


# Pauli matrices
PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Common quantum gates
HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
PHASE_S = np.array([[1, 0], [0, 1j]], dtype=complex)
PHASE_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)
SWAP = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]], dtype=complex)


@dataclass
class QuantumGate:
    """Represents a quantum gate operation."""
    name: str
    matrix: np.ndarray
    qubits: List[int]
    params: Optional[Dict] = field(default_factory=dict)


class QuantumComputingLab:
    """Advanced quantum computing simulation laboratory."""

    def __init__(self, n_qubits: int = 5):
        """
        Initialize quantum computing lab.

        Args:
            n_qubits: Default number of qubits for circuits
        """
        self.name = "Quantum Computing Laboratory"
        self.version = "2.0.0"
        self.default_n_qubits = n_qubits

    def create_quantum_state(self, n_qubits: int,
                           initial_state: Optional[Union[str, np.ndarray]] = None) -> np.ndarray:
        """
        Create a quantum state vector.

        Args:
            n_qubits: Number of qubits
            initial_state: Either a binary string like '101' or state vector

        Returns:
            Quantum state vector
        """
        dim = 2**n_qubits

        if initial_state is None:
            # Default to |00...0>
            state = np.zeros(dim, dtype=complex)
            state[0] = 1.0
        elif isinstance(initial_state, str):
            # Binary string like '101' -> |101>
            if len(initial_state) != n_qubits:
                raise ValueError(f"Binary string must have {n_qubits} bits")
            idx = int(initial_state, 2)
            state = np.zeros(dim, dtype=complex)
            state[idx] = 1.0
        else:
            # Custom state vector
            state = np.array(initial_state, dtype=complex)
            if len(state) != dim:
                raise ValueError(f"State vector must have dimension {dim}")
            # Normalize
            state = state / np.linalg.norm(state)

        return state

    def apply_gate(self, state: np.ndarray, gate_matrix: np.ndarray,
                  target_qubits: List[int]) -> np.ndarray:
        """
        Apply a quantum gate to specific qubits in the state.

        Args:
            state: Current quantum state vector
            gate_matrix: Matrix representation of the gate
            target_qubits: List of qubit indices to apply gate to

        Returns:
            New quantum state after gate application
        """
        n_qubits = int(np.log2(len(state)))
        n_gate_qubits = int(np.log2(gate_matrix.shape[0]))

        if len(target_qubits) != n_gate_qubits:
            raise ValueError(f"Gate requires {n_gate_qubits} qubits, got {len(target_qubits)}")

        # Build the full operator
        full_op = self._build_operator(gate_matrix, target_qubits, n_qubits)

        # Apply to state
        return full_op @ state

    def _build_operator(self, gate: np.ndarray, qubits: List[int], n_total: int) -> np.ndarray:
        """
        Build full operator matrix for gate acting on specific qubits.

        Args:
            gate: Gate matrix
            qubits: Target qubit indices
            n_total: Total number of qubits

        Returns:
            Full operator matrix
        """
        dim = 2**n_total
        op = np.eye(dim, dtype=complex)

        # Get all basis states
        for state_idx in range(dim):
            # Get binary representation
            state_binary = format(state_idx, f'0{n_total}b')

            # Extract target qubit values
            target_vals = ''.join(state_binary[q] for q in qubits)
            target_idx = int(target_vals, 2)

            # Apply gate to this subspace
            for new_target_idx in range(2**len(qubits)):
                if gate[new_target_idx, target_idx] != 0:
                    # Build new state
                    new_binary = list(state_binary)
                    new_target_binary = format(new_target_idx, f'0{len(qubits)}b')
                    for i, q in enumerate(qubits):
                        new_binary[q] = new_target_binary[i]
                    new_state_idx = int(''.join(new_binary), 2)

                    op[new_state_idx, state_idx] = gate[new_target_idx, target_idx]

        return op

    def hadamard_gate(self, state: np.ndarray, qubit: int) -> np.ndarray:
        """Apply Hadamard gate to a qubit."""
        return self.apply_gate(state, HADAMARD, [qubit])

    def pauli_x_gate(self, state: np.ndarray, qubit: int) -> np.ndarray:
        """Apply Pauli-X (NOT) gate to a qubit."""
        return self.apply_gate(state, PAULI_X, [qubit])

    def pauli_y_gate(self, state: np.ndarray, qubit: int) -> np.ndarray:
        """Apply Pauli-Y gate to a qubit."""
        return self.apply_gate(state, PAULI_Y, [qubit])

    def pauli_z_gate(self, state: np.ndarray, qubit: int) -> np.ndarray:
        """Apply Pauli-Z gate to a qubit."""
        return self.apply_gate(state, PAULI_Z, [qubit])

    def cnot_gate(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
        """Apply CNOT gate with given control and target qubits."""
        # Build CNOT for specific qubits
        n_qubits = int(np.log2(len(state)))
        dim = 2**n_qubits
        cnot_full = np.eye(dim, dtype=complex)

        for state_idx in range(dim):
            state_binary = format(state_idx, f'0{n_qubits}b')
            if state_binary[control] == '1':
                # Flip target bit
                new_binary = list(state_binary)
                new_binary[target] = '0' if new_binary[target] == '1' else '1'
                new_idx = int(''.join(new_binary), 2)
                cnot_full[new_idx, state_idx] = 1
                cnot_full[state_idx, state_idx] = 0

        return cnot_full @ state

    def toffoli_gate(self, state: np.ndarray, control1: int, control2: int,
                    target: int) -> np.ndarray:
        """Apply Toffoli (CCNOT) gate."""
        n_qubits = int(np.log2(len(state)))
        dim = 2**n_qubits
        toffoli = np.eye(dim, dtype=complex)

        for state_idx in range(dim):
            state_binary = format(state_idx, f'0{n_qubits}b')
            if state_binary[control1] == '1' and state_binary[control2] == '1':
                # Flip target bit
                new_binary = list(state_binary)
                new_binary[target] = '0' if new_binary[target] == '1' else '1'
                new_idx = int(''.join(new_binary), 2)
                toffoli[new_idx, state_idx] = 1
                toffoli[state_idx, state_idx] = 0

        return toffoli @ state

    def rotation_gate(self, state: np.ndarray, qubit: int, axis: str,
                     angle: float) -> np.ndarray:
        """
        Apply rotation gate around given axis.

        Args:
            state: Quantum state
            qubit: Target qubit
            axis: Rotation axis ('x', 'y', or 'z')
            angle: Rotation angle in radians

        Returns:
            Rotated state
        """
        if axis.lower() == 'x':
            gate = np.array([[np.cos(angle/2), -1j*np.sin(angle/2)],
                           [-1j*np.sin(angle/2), np.cos(angle/2)]], dtype=complex)
        elif axis.lower() == 'y':
            gate = np.array([[np.cos(angle/2), -np.sin(angle/2)],
                           [np.sin(angle/2), np.cos(angle/2)]], dtype=complex)
        elif axis.lower() == 'z':
            gate = np.array([[np.exp(-1j*angle/2), 0],
                           [0, np.exp(1j*angle/2)]], dtype=complex)
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")

        return self.apply_gate(state, gate, [qubit])

    def phase_gate(self, state: np.ndarray, qubit: int, phase: float) -> np.ndarray:
        """Apply phase gate with given phase."""
        gate = np.array([[1, 0], [0, np.exp(1j * phase)]], dtype=complex)
        return self.apply_gate(state, gate, [qubit])

    def measure(self, state: np.ndarray, qubits: Optional[List[int]] = None) -> Tuple[str, np.ndarray]:
        """
        Measure qubits in computational basis.

        Args:
            state: Quantum state to measure
            qubits: List of qubits to measure (None = all)

        Returns:
            Tuple of (measurement result as binary string, collapsed state)
        """
        n_qubits = int(np.log2(len(state)))

        if qubits is None:
            qubits = list(range(n_qubits))

        # Calculate probabilities
        probs = np.abs(state)**2

        # Sample outcome
        outcome_idx = np.random.choice(len(probs), p=probs)
        outcome_binary = format(outcome_idx, f'0{n_qubits}b')

        # Collapse state (for measured qubits only)
        collapsed_state = np.zeros_like(state)

        for idx in range(len(state)):
            idx_binary = format(idx, f'0{n_qubits}b')
            # Check if this state is consistent with measurement
            consistent = all(idx_binary[q] == outcome_binary[q] for q in qubits)
            if consistent:
                collapsed_state[idx] = state[idx]

        # Renormalize
        norm = np.linalg.norm(collapsed_state)
        if norm > 0:
            collapsed_state = collapsed_state / norm

        # Return measured bits
        measured_bits = ''.join(outcome_binary[q] for q in qubits)

        return measured_bits, collapsed_state

    def quantum_circuit_simulate(self, gates: List[QuantumGate], n_qubits: int,
                                initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Simulate a quantum circuit defined by a sequence of gates.

        Args:
            gates: List of QuantumGate objects
            n_qubits: Number of qubits in circuit
            initial_state: Initial state (default |0...0>)

        Returns:
            Final quantum state
        """
        # Initialize state
        if initial_state is None:
            state = self.create_quantum_state(n_qubits)
        else:
            state = initial_state.copy()

        # Apply gates sequentially
        for gate in gates:
            state = self.apply_gate(state, gate.matrix, gate.qubits)

        return state

    def grovers_algorithm(self, oracle: Callable, n_qubits: int,
                         n_iterations: Optional[int] = None) -> Tuple[str, float]:
        """
        Implement Grover's quantum search algorithm.

        Args:
            oracle: Function that marks target states (returns True for targets)
            n_qubits: Number of qubits
            n_iterations: Number of Grover iterations (auto-calculated if None)

        Returns:
            Tuple of (found state as binary string, success probability)
        """
        N = 2**n_qubits

        # Count number of marked items (for optimal iteration count)
        M = sum(1 for x in range(N) if oracle(x))

        if M == 0:
            raise ValueError("No marked items in oracle")

        # Calculate optimal number of iterations
        if n_iterations is None:
            n_iterations = int(np.pi / 4 * np.sqrt(N / M))

        # Initialize in superposition
        state = np.ones(N, dtype=complex) / np.sqrt(N)

        for _ in range(n_iterations):
            # Oracle operator
            for x in range(N):
                if oracle(x):
                    state[x] *= -1

            # Diffusion operator (inversion about average)
            avg = np.mean(state)
            state = 2 * avg - state

        # Measure
        probs = np.abs(state)**2
        measured = np.random.choice(N, p=probs)

        # Check if we found a marked item
        success = oracle(measured)
        success_prob = sum(probs[x] for x in range(N) if oracle(x))

        return format(measured, f'0{n_qubits}b'), success_prob

    def quantum_fourier_transform(self, state: np.ndarray, qubits: Optional[List[int]] = None) -> np.ndarray:
        """
        Apply Quantum Fourier Transform to specified qubits.

        Args:
            state: Input quantum state
            qubits: Qubits to apply QFT to (default: all)

        Returns:
            State after QFT
        """
        n_total = int(np.log2(len(state)))

        if qubits is None:
            qubits = list(range(n_total))

        n = len(qubits)

        # Apply QFT circuit
        for j in range(n):
            # Hadamard on qubit j
            state = self.hadamard_gate(state, qubits[j])

            # Controlled rotations
            for k in range(j + 1, n):
                angle = 2 * np.pi / (2**(k - j + 1))
                # Controlled phase rotation
                state = self._controlled_phase(state, qubits[k], qubits[j], angle)

        # Swap qubits (reverse order)
        for j in range(n // 2):
            state = self._swap_qubits(state, qubits[j], qubits[n - 1 - j])

        return state

    def _controlled_phase(self, state: np.ndarray, control: int, target: int,
                         phase: float) -> np.ndarray:
        """Apply controlled phase gate."""
        n_qubits = int(np.log2(len(state)))
        dim = 2**n_qubits
        cp_gate = np.eye(dim, dtype=complex)

        for idx in range(dim):
            binary = format(idx, f'0{n_qubits}b')
            if binary[control] == '1' and binary[target] == '1':
                cp_gate[idx, idx] = np.exp(1j * phase)

        return cp_gate @ state

    def _swap_qubits(self, state: np.ndarray, qubit1: int, qubit2: int) -> np.ndarray:
        """Swap two qubits."""
        n_qubits = int(np.log2(len(state)))
        dim = 2**n_qubits
        swap = np.eye(dim, dtype=complex)

        for idx in range(dim):
            binary = list(format(idx, f'0{n_qubits}b'))
            # Swap bits
            new_binary = binary.copy()
            new_binary[qubit1], new_binary[qubit2] = binary[qubit2], binary[qubit1]
            new_idx = int(''.join(new_binary), 2)
            swap[new_idx, idx] = 1
            if new_idx != idx:
                swap[idx, idx] = 0

        return swap @ state

    def quantum_error_correction_3qubit(self, state: np.ndarray,
                                       error_type: str = 'bit_flip') -> np.ndarray:
        """
        Implement 3-qubit error correction code.

        Args:
            state: Single qubit state to protect
            error_type: 'bit_flip' or 'phase_flip'

        Returns:
            Corrected state
        """
        # Encode single qubit into 3 qubits
        # |0> -> |000>, |1> -> |111>
        alpha, beta = state[0], state[1]

        encoded = np.zeros(8, dtype=complex)
        encoded[0] = alpha  # |000>
        encoded[7] = beta   # |111>

        # Simulate random error on one qubit
        error_qubit = np.random.randint(3)

        if error_type == 'bit_flip':
            # Apply bit flip error
            encoded = self.pauli_x_gate(encoded, error_qubit)
        elif error_type == 'phase_flip':
            # Apply phase flip error
            encoded = self.pauli_z_gate(encoded, error_qubit)

        # Error syndrome detection
        # Measure parity of pairs
        syndrome = []

        # Parity of qubits 0 and 1
        parity_01 = self._measure_parity(encoded, 0, 1)
        syndrome.append(parity_01)

        # Parity of qubits 1 and 2
        parity_12 = self._measure_parity(encoded, 1, 2)
        syndrome.append(parity_12)

        # Determine which qubit has error based on syndrome
        if syndrome == [1, 0]:  # Error on qubit 0
            if error_type == 'bit_flip':
                encoded = self.pauli_x_gate(encoded, 0)
            else:
                encoded = self.pauli_z_gate(encoded, 0)
        elif syndrome == [1, 1]:  # Error on qubit 1
            if error_type == 'bit_flip':
                encoded = self.pauli_x_gate(encoded, 1)
            else:
                encoded = self.pauli_z_gate(encoded, 1)
        elif syndrome == [0, 1]:  # Error on qubit 2
            if error_type == 'bit_flip':
                encoded = self.pauli_x_gate(encoded, 2)
            else:
                encoded = self.pauli_z_gate(encoded, 2)
        # syndrome == [0, 0] means no error

        # Decode back to single qubit
        decoded = np.array([encoded[0], encoded[7]], dtype=complex)
        decoded = decoded / np.linalg.norm(decoded)

        return decoded

    def _measure_parity(self, state: np.ndarray, qubit1: int, qubit2: int) -> int:
        """Measure parity of two qubits (0 if same, 1 if different)."""
        n_qubits = int(np.log2(len(state)))
        parity = 0
        prob_different = 0

        for idx in range(len(state)):
            if abs(state[idx])**2 > 1e-10:
                binary = format(idx, f'0{n_qubits}b')
                if binary[qubit1] != binary[qubit2]:
                    prob_different += abs(state[idx])**2

        return 1 if prob_different > 0.5 else 0

    def calculate_fidelity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Calculate fidelity between two quantum states.

        Args:
            state1: First quantum state
            state2: Second quantum state

        Returns:
            Fidelity (0 to 1)
        """
        # For pure states, fidelity is |<ψ1|ψ2>|²
        overlap = np.vdot(state1, state2)
        return abs(overlap)**2

    def calculate_entanglement_entropy(self, state: np.ndarray,
                                      partition: List[int]) -> float:
        """
        Calculate entanglement entropy for bipartite system.

        Args:
            state: Quantum state vector
            partition: List of qubit indices for first partition

        Returns:
            Von Neumann entropy of reduced density matrix
        """
        n_qubits = int(np.log2(len(state)))
        partition_b = [q for q in range(n_qubits) if q not in partition]

        # Construct reduced density matrix
        dim_a = 2**len(partition)
        dim_b = 2**len(partition_b)

        # Reshape state for partial trace
        state_matrix = state.reshape(dim_a, dim_b)

        # Reduced density matrix for partition A
        rho_a = state_matrix @ state_matrix.conj().T

        # Calculate von Neumann entropy
        eigenvalues = np.linalg.eigvalsh(rho_a)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove zeros

        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))

        return entropy

    def bell_state_preparation(self, bell_type: str = 'phi_plus') -> np.ndarray:
        """
        Prepare a Bell state (maximally entangled 2-qubit state).

        Args:
            bell_type: Type of Bell state ('phi_plus', 'phi_minus', 'psi_plus', 'psi_minus')

        Returns:
            Bell state vector
        """
        # Start with |00>
        state = self.create_quantum_state(2, '00')

        # Apply Hadamard to first qubit
        state = self.hadamard_gate(state, 0)

        # Apply CNOT
        state = self.cnot_gate(state, 0, 1)

        # Apply additional gates for different Bell states
        if bell_type == 'phi_minus':
            state = self.pauli_z_gate(state, 0)
        elif bell_type == 'psi_plus':
            state = self.pauli_x_gate(state, 0)
        elif bell_type == 'psi_minus':
            state = self.pauli_x_gate(state, 0)
            state = self.pauli_z_gate(state, 0)

        return state

    def quantum_teleportation(self, state_to_teleport: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        Simulate quantum teleportation protocol.

        Args:
            state_to_teleport: Single qubit state to teleport

        Returns:
            Tuple of (measurement results, Bob's final state)
        """
        # Prepare 3-qubit system: Alice's qubit + entangled pair
        alpha, beta = state_to_teleport[0], state_to_teleport[1]

        # Initial state: |ψ>|00> where |ψ> = α|0> + β|1>
        state = np.zeros(8, dtype=complex)
        state[0] = alpha  # |000>
        state[4] = beta   # |100>

        # Create entangled pair between qubits 1 and 2
        state = self.hadamard_gate(state, 1)
        state = self.cnot_gate(state, 1, 2)

        # Alice performs Bell measurement on qubits 0 and 1
        state = self.cnot_gate(state, 0, 1)
        state = self.hadamard_gate(state, 0)

        # Measure Alice's qubits
        measurement, state = self.measure(state, [0, 1])

        # Bob applies corrections based on measurement
        if measurement[1] == '1':  # Second bit
            state = self.pauli_x_gate(state, 2)
        if measurement[0] == '1':  # First bit
            state = self.pauli_z_gate(state, 2)

        # Extract Bob's qubit state
        # We need to trace out Alice's qubits
        bob_state = np.zeros(2, dtype=complex)
        for idx in range(8):
            binary = format(idx, '03b')
            if binary[:2] == measurement:
                bob_idx = int(binary[2])
                bob_state[bob_idx] = state[idx]

        # Normalize
        bob_state = bob_state / np.linalg.norm(bob_state)

        return measurement, bob_state


def run_demo():
    """Demonstrate quantum computing operations."""
    lab = QuantumComputingLab()
    print(f"Initializing {lab.name} v{lab.version}")
    print("=" * 60)

    # 1. Basic gate operations
    print("\n1. Basic Quantum Gates:")
    state = lab.create_quantum_state(2, '00')
    print(f"   Initial state |00>: {state}")

    state = lab.hadamard_gate(state, 0)
    print(f"   After Hadamard on qubit 0: {state}")

    state = lab.cnot_gate(state, 0, 1)
    print(f"   After CNOT(0,1) - Bell state: {state}")

    # 2. Quantum circuit simulation
    print("\n2. Quantum Circuit Simulation:")
    gates = [
        QuantumGate("H", HADAMARD, [0]),
        QuantumGate("H", HADAMARD, [1]),
        QuantumGate("CNOT", CNOT, [0, 1]),
    ]
    final_state = lab.quantum_circuit_simulate(gates, 2)
    print(f"   Final state after H-H-CNOT circuit: {final_state}")

    # 3. Grover's algorithm
    print("\n3. Grover's Algorithm (3 qubits, searching for |101>):")

    def oracle(x):
        return x == 5  # Binary 101

    result, prob = lab.grovers_algorithm(oracle, 3)
    print(f"   Found state: |{result}>")
    print(f"   Success probability: {prob:.3f}")

    # 4. Quantum Fourier Transform
    print("\n4. Quantum Fourier Transform:")
    state = lab.create_quantum_state(3, '101')
    qft_state = lab.quantum_fourier_transform(state)
    print(f"   Input |101>")
    print(f"   After QFT (first 4 amplitudes): {qft_state[:4]}")

    # 5. Error correction
    print("\n5. Quantum Error Correction (3-qubit code):")
    # Create a superposition state
    single_qubit = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    print(f"   Original state: {single_qubit}")

    corrected = lab.quantum_error_correction_3qubit(single_qubit, 'bit_flip')
    print(f"   After error + correction: {corrected}")

    fidelity = lab.calculate_fidelity(single_qubit, corrected)
    print(f"   Fidelity with original: {fidelity:.4f}")

    # 6. Bell states and entanglement
    print("\n6. Bell States and Entanglement:")
    bell = lab.bell_state_preparation('phi_plus')
    print(f"   |Φ+> Bell state: {bell}")

    entropy = lab.calculate_entanglement_entropy(bell, [0])
    print(f"   Entanglement entropy: {entropy:.3f} bits")

    # 7. Quantum teleportation
    print("\n7. Quantum Teleportation:")
    to_teleport = np.array([0.6, 0.8], dtype=complex)
    print(f"   State to teleport: {to_teleport}")

    measurement, bob_state = lab.quantum_teleportation(to_teleport)
    print(f"   Alice's measurement: {measurement}")
    print(f"   Bob's received state: {bob_state}")

    fidelity = lab.calculate_fidelity(to_teleport, bob_state)
    print(f"   Teleportation fidelity: {fidelity:.4f}")

    # 8. Rotation gates
    print("\n8. Rotation Gates:")
    state = lab.create_quantum_state(1, '0')
    state = lab.rotation_gate(state, 0, 'y', np.pi/4)
    print(f"   After Ry(π/4) on |0>: {state}")

    print("\n" + "=" * 60)
    print("Quantum Computing Lab demonstration complete!")


if __name__ == "__main__":
    run_demo()