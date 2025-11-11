"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

QUANTUM COMPUTING LAB
Free gift to the scientific community from QuLabInfinite.
"""

from typing import List, Tuple
import numpy as np
from dataclasses import dataclass, field
import scipy.constants
from scipy.constants import physical_constants

# Constants and configuration
kB = scipy.constants.k  # Boltzmann constant
N_A = scipy.constants.Avogadro  # Avogadro number
g = scipy.constants.g  # gravitational acceleration
c = scipy.constants.c  # speed of light
hbar = scipy.constants.h / (2 * np.pi)  # reduced Planck constant
eV = physical_constants['electron volt'][0]  # electron volt

# Quantum gate definitions using NumPy arrays
I_1 = np.eye(2, dtype=np.complex128)
X_gate = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y_gate = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z_gate = np.array([[1, 0], [0, -1]], dtype=np.complex128)

@dataclass
class QuantumState:
    state_vector: np.ndarray

@dataclass
class QuantumCircuit:
    qubits: int = field(default=1)  # default number of qubits
    gates_list: List[Tuple[str, Tuple[int]]] = field(default_factory=list)
    
    def __post_init__(self):
        self.state_vector = np.zeros(2 ** self.qubits, dtype=np.float64)
        self.state_vector[0] = 1.0  # Initial state |0>
        
    def apply_gate(self, gate_name: str, qubit_index: int) -> None:
        matrix = getattr(self, gate_name)
        tensor_product = np.eye(2 ** (self.qubits - qubit_index - 1), dtype=np.float64)
        operation_matrix = np.kron(tensor_product, matrix)
        
        if qubit_index > 0:
            operation_matrix = np.kron(np.eye(2 ** qubit_index, dtype=np.float64), operation_matrix)

        self.state_vector = (operation_matrix @ self.state_vector).astype(np.float64) % 1
        
    def measure(self):
        probabilities = np.abs(self.state_vector) ** 2
        outcome = np.random.choice(range(len(probabilities)), p=probabilities)
        return outcome
    
@dataclass
class QuantumComputer:
    circuit: QuantumCircuit

def run_demo():
    qc = QuantumComputer(QuantumCircuit(qubits=1))
    
    # Apply Hadamard gate (Hadamard matrix) to the first qubit
    H_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    for i in range(1):
        qc.circuit.apply_gate("X_gate", i)

    print(f"Quantum State Vector: {qc.circuit.state_vector}")
    
if __name__ == '__main__':
    run_demo()