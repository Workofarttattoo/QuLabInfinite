"""
Quantum Gate Operations

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Standard quantum gates implemented as unitary matrices.
These gates preserve the norm of the quantum state (unitary transformations).
"""

import numpy as np
from typing import List
from .quantum_state import QuantumState


# ============================================================================
# SINGLE-QUBIT GATES
# ============================================================================

def hadamard() -> np.ndarray:
    """
    Hadamard gate: Creates superposition.

    H|0⟩ = (|0⟩ + |1⟩)/√2
    H|1⟩ = (|0⟩ - |1⟩)/√2

    Matrix:
        H = 1/√2 [[1,  1],
                  [1, -1]]
    """
    return np.array([[1, 1],
                     [1, -1]], dtype=complex) / np.sqrt(2)


def pauli_x() -> np.ndarray:
    """
    Pauli-X gate (NOT gate): Bit flip.

    X|0⟩ = |1⟩
    X|1⟩ = |0⟩

    Matrix:
        X = [[0, 1],
             [1, 0]]
    """
    return np.array([[0, 1],
                     [1, 0]], dtype=complex)


def pauli_y() -> np.ndarray:
    """
    Pauli-Y gate: Bit flip with phase.

    Y|0⟩ = i|1⟩
    Y|1⟩ = -i|0⟩

    Matrix:
        Y = [[0, -i],
             [i,  0]]
    """
    return np.array([[0, -1j],
                     [1j, 0]], dtype=complex)


def pauli_z() -> np.ndarray:
    """
    Pauli-Z gate: Phase flip.

    Z|0⟩ = |0⟩
    Z|1⟩ = -|1⟩

    Matrix:
        Z = [[1,  0],
             [0, -1]]
    """
    return np.array([[1, 0],
                     [0, -1]], dtype=complex)


def phase_gate(theta: float) -> np.ndarray:
    """
    Phase gate: Adds phase to |1⟩.

    P(θ)|0⟩ = |0⟩
    P(θ)|1⟩ = e^(iθ)|1⟩

    Matrix:
        P(θ) = [[1,       0     ],
                [0, e^(iθ)      ]]
    """
    return np.array([[1, 0],
                     [0, np.exp(1j * theta)]], dtype=complex)


def rotation_x(theta: float) -> np.ndarray:
    """
    Rotation around X-axis of Bloch sphere.

    RX(θ) = [[cos(θ/2),  -i·sin(θ/2)],
             [-i·sin(θ/2), cos(θ/2)]]
    """
    return np.array([[np.cos(theta/2), -1j * np.sin(theta/2)],
                     [-1j * np.sin(theta/2), np.cos(theta/2)]], dtype=complex)


def rotation_y(theta: float) -> np.ndarray:
    """
    Rotation around Y-axis of Bloch sphere.

    RY(θ) = [[cos(θ/2), -sin(θ/2)],
             [sin(θ/2),  cos(θ/2)]]
    """
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2), np.cos(theta/2)]], dtype=complex)


def rotation_z(theta: float) -> np.ndarray:
    """
    Rotation around Z-axis of Bloch sphere.

    RZ(θ) = [[e^(-iθ/2),    0      ],
             [0,         e^(iθ/2)  ]]
    """
    return np.array([[np.exp(-1j * theta/2), 0],
                     [0, np.exp(1j * theta/2)]], dtype=complex)


def s_gate() -> np.ndarray:
    """
    S gate (Phase gate with θ=π/2).

    S = [[1, 0],
         [0, i]]
    """
    return np.array([[1, 0],
                     [0, 1j]], dtype=complex)


def t_gate() -> np.ndarray:
    """
    T gate (Phase gate with θ=π/4).

    T = [[1,         0        ],
         [0, e^(iπ/4) = (1+i)/√2]]
    """
    return np.array([[1, 0],
                     [0, np.exp(1j * np.pi/4)]], dtype=complex)


# ============================================================================
# TWO-QUBIT GATES
# ============================================================================

def cnot() -> np.ndarray:
    """
    CNOT (Controlled-NOT) gate: Flips target if control is |1⟩.

    CNOT|00⟩ = |00⟩
    CNOT|01⟩ = |01⟩
    CNOT|10⟩ = |11⟩  (flip!)
    CNOT|11⟩ = |10⟩  (flip!)

    Matrix:
        CNOT = [[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]]
    """
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=complex)


def cz() -> np.ndarray:
    """
    Controlled-Z gate: Applies phase flip if both qubits are |1⟩.

    CZ|11⟩ = -|11⟩

    Matrix:
        CZ = [[1, 0, 0,  0],
              [0, 1, 0,  0],
              [0, 0, 1,  0],
              [0, 0, 0, -1]]
    """
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]], dtype=complex)


def swap() -> np.ndarray:
    """
    SWAP gate: Swaps two qubits.

    SWAP|01⟩ = |10⟩
    SWAP|10⟩ = |01⟩

    Matrix:
        SWAP = [[1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]]
    """
    return np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]], dtype=complex)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def apply_hadamard(state: QuantumState, qubit: int):
    """Apply Hadamard gate to specific qubit."""
    state.apply_gate(hadamard(), [qubit])


def apply_x(state: QuantumState, qubit: int):
    """Apply Pauli-X (NOT) gate to specific qubit."""
    state.apply_gate(pauli_x(), [qubit])


def apply_y(state: QuantumState, qubit: int):
    """Apply Pauli-Y gate to specific qubit."""
    state.apply_gate(pauli_y(), [qubit])


def apply_z(state: QuantumState, qubit: int):
    """Apply Pauli-Z gate to specific qubit."""
    state.apply_gate(pauli_z(), [qubit])


def apply_phase(state: QuantumState, qubit: int, theta: float):
    """Apply phase gate to specific qubit."""
    state.apply_gate(phase_gate(theta), [qubit])


def apply_rx(state: QuantumState, qubit: int, theta: float):
    """Apply RX rotation gate to specific qubit."""
    state.apply_gate(rotation_x(theta), [qubit])


def apply_ry(state: QuantumState, qubit: int, theta: float):
    """Apply RY rotation gate to specific qubit."""
    state.apply_gate(rotation_y(theta), [qubit])


def apply_rz(state: QuantumState, qubit: int, theta: float):
    """Apply RZ rotation gate to specific qubit."""
    state.apply_gate(rotation_z(theta), [qubit])


def apply_cnot(state: QuantumState, control: int, target: int):
    """Apply CNOT gate with specified control and target qubits."""
    state.apply_gate(cnot(), [control, target])


def apply_cz(state: QuantumState, control: int, target: int):
    """Apply CZ gate with specified control and target qubits."""
    state.apply_gate(cz(), [control, target])


def apply_swap(state: QuantumState, qubit1: int, qubit2: int):
    """Apply SWAP gate to two qubits."""
    state.apply_gate(swap(), [qubit1, qubit2])


# ============================================================================
# GATE VERIFICATION
# ============================================================================

def verify_unitary(gate: np.ndarray) -> bool:
    """
    Verify that a matrix is unitary (U†U = I).

    A gate is valid if U†U = I (preserves norm).
    """
    identity = np.eye(gate.shape[0], dtype=complex)
    product = gate.conj().T @ gate
    return np.allclose(product, identity, atol=1e-10)


if __name__ == "__main__":
    print("=" * 60)
    print("QUANTUM GATES DEMONSTRATION")
    print("=" * 60)

    # Test 1: Verify all gates are unitary
    print("\n1. Verifying Gate Unitarity:")
    gates_to_test = {
        "Hadamard": hadamard(),
        "Pauli-X": pauli_x(),
        "Pauli-Y": pauli_y(),
        "Pauli-Z": pauli_z(),
        "Phase(π/4)": phase_gate(np.pi/4),
        "RX(π/2)": rotation_x(np.pi/2),
        "RY(π/2)": rotation_y(np.pi/2),
        "RZ(π/2)": rotation_z(np.pi/2),
        "S": s_gate(),
        "T": t_gate(),
        "CNOT": cnot(),
        "CZ": cz(),
        "SWAP": swap(),
    }

    for name, gate in gates_to_test.items():
        is_unitary = verify_unitary(gate)
        status = "✓" if is_unitary else "✗"
        print(f"   {status} {name}: {'Unitary' if is_unitary else 'NOT Unitary'}")

    # Test 2: Hadamard creates superposition
    print("\n2. Hadamard Gate Creates Superposition:")
    state = QuantumState(1)
    print(f"Before: {state}")
    apply_hadamard(state, 0)
    print(f"After:  {state}")

    # Test 3: CNOT creates entanglement
    print("\n3. CNOT Creates Entanglement:")
    state = QuantumState(2)
    apply_hadamard(state, 0)  # Create superposition on qubit 0
    print(f"After H on qubit 0:\n{state}")
    apply_cnot(state, 0, 1)  # Entangle with qubit 1
    print(f"After CNOT(0,1) - Bell State:\n{state}")

    print("\n" + "=" * 60)
