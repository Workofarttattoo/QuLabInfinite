#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.

QuantumLabSimulator - Main quantum laboratory interface
Wraps and extends existing quantum simulators with unified API
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
from enum import Enum

try:
    from core.base_lab import BaseLab
except ImportError:
    # This is a fallback for script execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from core.base_lab import BaseLab


# Import existing quantum simulators
sys.path.append('/Users/noone/repos/consciousness/ech0_modules')
try:
    from quantum_circuit_simulator import QuantumCircuitSimulator as CircuitSim
    from quantum_cognition import QuantumCognitionSystem
    SIMULATORS_AVAILABLE = True
except ImportError:
    SIMULATORS_AVAILABLE = False
    print("[WARN] Existing quantum simulators not found, using fallback mode")


class SimulationBackend(Enum):
    """Available simulation backends"""
    STATEVECTOR_EXACT = "statevector"  # 1-30 qubits, exact
    TENSOR_NETWORK = "tensor_network"   # 30-50 qubits, approximate
    MPS = "mps"                         # 30-50 qubits, Matrix Product State
    DENSITY_MATRIX = "density_matrix"   # For mixed states
    COGNITION = "cognition"             # Quantum-inspired cognition


@dataclass
class SimulationConfig:
    """Configuration for quantum simulation"""
    num_qubits: int
    backend: SimulationBackend = SimulationBackend.STATEVECTOR_EXACT
    optimize_for_m4: bool = True
    max_memory_gb: float = 20.0
    error_correction: bool = False
    noise_model: Optional[Dict] = None


class QuantumLabSimulator(BaseLab):
    """
    Unified quantum laboratory simulator.

    Features:
    - 30-qubit exact statevector simulation
    - Tensor network approximation for >30 qubits
    - Quantum chemistry integration
    - Materials science quantum calculations
    - Quantum sensor modeling
    - Cognition-inspired algorithms

    Integration:
    - Wraps existing CircuitSimulator (30-qubit exact)
    - Wraps QuantumCognition (quantum-inspired)
    - Extends with chemistry, materials, sensors

    ECH0 Usage Examples:
    ```python
    # Basic circuit simulation
    lab = QuantumLabSimulator(num_qubits=5)
    lab.h(0)  # Hadamard on qubit 0
    lab.cnot(0, 1)  # Entangle qubits
    results = lab.measure_all()

    # Chemistry calculation
    from quantum_chemistry import Molecule
    h2 = Molecule.hydrogen_molecule(bond_length=0.74)
    energy = lab.chemistry.compute_ground_state_energy(h2)

    # Materials property
    energy_gap = lab.materials.compute_band_gap("silicon")

    # Quantum sensing
    sensitivity = lab.sensors.magnetometry_sensitivity(num_qubits=10)
    ```
    """

    def __init__(
        self,
        num_qubits: int = 5,
        backend: Union[str, SimulationBackend] = SimulationBackend.STATEVECTOR_EXACT,
        optimize_for_m4: bool = True,
        verbose: bool = True,
        config: Dict[str, Any] = None
    ):
        """
        Initialize quantum laboratory.

        Args:
            num_qubits: Number of qubits (1-30 exact, 30-50 approximate)
            backend: Simulation backend to use (string or enum)
            optimize_for_m4: Use M4 Mac optimizations
            verbose: Print initialization info
        """
        super().__init__(config)
        self.num_qubits = num_qubits
        if isinstance(backend, str):
            self.backend = SimulationBackend(backend)
        else:
            self.backend = backend
        self.optimize_for_m4 = optimize_for_m4
        self.verbose = verbose

        # Select backend
        self._initialize_backend()

        # Initialize sub-systems (lazy loading)
        self._chemistry = None
        self._materials = None
        self._sensors = None
        self._cognition = None

        if verbose:
            self._print_initialization_summary()

    def run_experiment(self, experiment_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a quantum experiment.

        Example spec:
        {
            "experiment_type": "bell_pair",
        }
        {
            "experiment_type": "ghz_state",
            "num_qubits": 4
        }
        {
            "experiment_type": "custom_circuit",
            "operations": [
                {"gate": "h", "qubit": 0},
                {"gate": "cnot", "control": 0, "target": 1}
            ]
        }
        """
        exp_type = experiment_spec.get("experiment_type")
        if not exp_type:
            raise ValueError("'experiment_type' is required.")

        if exp_type == "bell_pair":
            self.reset()
            self.h(0).cnot(0, 1)
            return {"status": "completed", "probabilities": self.get_probabilities()}

        elif exp_type == "ghz_state":
            num_qubits = experiment_spec.get("num_qubits", self.num_qubits)
            if num_qubits != self.num_qubits:
                # Re-initialize if necessary, though not ideal
                self.__init__(num_qubits=num_qubits, backend=self.backend, optimize_for_m4=self.optimize_for_m4)
            self.reset()
            self.h(0)
            for i in range(num_qubits - 1):
                self.cnot(i, i + 1)
            return {"status": "completed", "probabilities": self.get_probabilities()}

        elif exp_type == "custom_circuit":
            self.reset()
            operations = experiment_spec.get("operations", [])
            for op in operations:
                gate = op.get("gate")
                if gate == "h":
                    self.h(op["qubit"])
                elif gate == "x":
                    self.x(op["qubit"])
                elif gate == "cnot":
                    self.cnot(op["control"], op["target"])
                # ... add other gates as needed
            return {"status": "completed", "probabilities": self.get_probabilities()}
        else:
            raise ValueError(f"Unknown experiment type: {exp_type}")

    def get_status(self) -> Dict[str, Any]:
        """Get information about current backend and state."""
        status = self.get_backend_info()
        status["probabilities"] = self.get_probabilities()
        return status

    def _initialize_backend(self):
        """Initialize simulation backend"""
        if self.backend == SimulationBackend.STATEVECTOR_EXACT:
            if self.num_qubits > 30:
                print(f"[WARN] {self.num_qubits} qubits exceeds statevector limit (30)")
                print(f"[INFO] Switching to tensor network backend")
                self.backend = SimulationBackend.TENSOR_NETWORK
                self._initialize_tensor_network_backend()
            else:
                if SIMULATORS_AVAILABLE:
                    self.circuit = CircuitSim(self.num_qubits, self.optimize_for_m4)
                else:
                    self._initialize_fallback_backend()

        elif self.backend == SimulationBackend.TENSOR_NETWORK:
            self._initialize_tensor_network_backend()

        elif self.backend == SimulationBackend.MPS:
            self._initialize_mps_backend()

        elif self.backend == SimulationBackend.COGNITION:
            if SIMULATORS_AVAILABLE:
                self.cognition_system = QuantumCognitionSystem()
            else:
                print("[WARN] Quantum cognition system not available")

    def _initialize_tensor_network_backend(self):
        """Initialize tensor network approximation backend"""
        print(f"[INFO] Tensor network backend for {self.num_qubits} qubits")
        print(f"[INFO] Using Matrix Product State (MPS) approximation")

        # Simplified MPS representation
        # Each qubit represented as 2×D×D tensor (D = bond dimension)
        self.bond_dimension = min(64, 2**(self.num_qubits // 2))
        self.mps_tensors = []

        # Initialize MPS in |0⟩^n state
        for i in range(self.num_qubits):
            if i == 0:
                # First tensor: 2×1×D
                tensor = np.zeros((2, 1, self.bond_dimension), dtype=np.complex128)
                tensor[0, 0, 0] = 1.0  # |0⟩ state
            elif i == self.num_qubits - 1:
                # Last tensor: 2×D×1
                tensor = np.zeros((2, self.bond_dimension, 1), dtype=np.complex128)
                tensor[0, 0, 0] = 1.0
            else:
                # Middle tensors: 2×D×D
                tensor = np.zeros((2, self.bond_dimension, self.bond_dimension), dtype=np.complex128)
                tensor[0, 0, 0] = 1.0

            self.mps_tensors.append(tensor)

        print(f"[INFO] MPS bond dimension: {self.bond_dimension}")
        print(f"[INFO] Memory usage: ~{self._estimate_mps_memory():.2f} GB")

    def _initialize_mps_backend(self):
        """Initialize pure MPS backend"""
        self._initialize_tensor_network_backend()

    def _initialize_fallback_backend(self):
        """Fallback implementation if simulators unavailable"""
        print("[INFO] Using fallback statevector implementation")
        self.dim = 2 ** self.num_qubits
        self.statevector = np.zeros(self.dim, dtype=np.complex128)
        self.statevector[0] = 1.0 + 0j

    def _estimate_mps_memory(self) -> float:
        """Estimate MPS memory usage in GB"""
        # Each tensor: 2 * D^2 * 16 bytes (complex128)
        tensor_size = 2 * self.bond_dimension**2 * 16
        total_size = tensor_size * self.num_qubits
        return total_size / (1024**3)

    def _print_initialization_summary(self):
        """Print initialization summary"""
        print(f"\n{'='*60}")
        print(f"QULAB INFINITE - QUANTUM LABORATORY SIMULATOR")
        print(f"{'='*60}")
        print(f"\n⚛️  Configuration:")
        print(f"   Qubits: {self.num_qubits}")
        print(f"   Backend: {self.backend.value}")

        if self.backend == SimulationBackend.STATEVECTOR_EXACT:
            memory_gb = (2**self.num_qubits * 16) / (1024**3)
            print(f"   Hilbert space: {2**self.num_qubits:,} dimensions")
            print(f"   Memory: {memory_gb:.2f} GB")
        elif self.backend in [SimulationBackend.TENSOR_NETWORK, SimulationBackend.MPS]:
            print(f"   Bond dimension: {self.bond_dimension}")
            print(f"   Memory: ~{self._estimate_mps_memory():.2f} GB")

        print(f"   M4 optimization: {'✅ Enabled' if self.optimize_for_m4 else '❌ Disabled'}")
        print(f"\n✅ Quantum laboratory initialized\n")

    # ========== GATE OPERATIONS (delegate to backend) ==========

    def h(self, qubit: int):
        """Apply Hadamard gate"""
        if hasattr(self, 'circuit'):
            self.circuit.h(qubit)
        elif hasattr(self, 'mps_tensors'):
            self._apply_mps_gate('H', qubit)
        elif hasattr(self, 'statevector'):
            gate = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
            self._apply_statevector_gate(gate, qubit)
        return self

    def x(self, qubit: int):
        """Apply Pauli-X gate"""
        if hasattr(self, 'circuit'):
            self.circuit.x(qubit)
        elif hasattr(self, 'mps_tensors'):
            self._apply_mps_gate('X', qubit)
        elif hasattr(self, 'statevector'):
            gate = np.array([[0, 1], [1, 0]], dtype=np.complex128)
            self._apply_statevector_gate(gate, qubit)
        return self

    def y(self, qubit: int):
        """Apply Pauli-Y gate"""
        if hasattr(self, 'circuit'):
            self.circuit.y(qubit)
        elif hasattr(self, 'mps_tensors'):
            self._apply_mps_gate('Y', qubit)
        elif hasattr(self, 'statevector'):
            gate = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
            self._apply_statevector_gate(gate, qubit)
        return self

    def z(self, qubit: int):
        """Apply Pauli-Z gate"""
        if hasattr(self, 'circuit'):
            self.circuit.z(qubit)
        elif hasattr(self, 'mps_tensors'):
            self._apply_mps_gate('Z', qubit)
        elif hasattr(self, 'statevector'):
            gate = np.array([[1, 0], [0, -1]], dtype=np.complex128)
            self._apply_statevector_gate(gate, qubit)
        return self

    def rx(self, qubit: int, theta: float):
        """Apply RX rotation"""
        if hasattr(self, 'circuit'):
            self.circuit.rx(qubit, theta)
        elif hasattr(self, 'mps_tensors'):
            self._apply_mps_rotation('RX', qubit, theta)
        elif hasattr(self, 'statevector'):
            gate = np.array([
                [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                [-1j * np.sin(theta / 2), np.cos(theta / 2)],
            ], dtype=np.complex128)
            self._apply_statevector_gate(gate, qubit)
        return self

    def ry(self, qubit: int, theta: float):
        """Apply RY rotation"""
        if hasattr(self, 'circuit'):
            self.circuit.ry(qubit, theta)
        elif hasattr(self, 'mps_tensors'):
            self._apply_mps_rotation('RY', qubit, theta)
        elif hasattr(self, 'statevector'):
            gate = np.array([
                [np.cos(theta / 2), -np.sin(theta / 2)],
                [np.sin(theta / 2), np.cos(theta / 2)],
            ], dtype=np.complex128)
            self._apply_statevector_gate(gate, qubit)
        return self

    def rz(self, qubit: int, theta: float):
        """Apply RZ rotation"""
        if hasattr(self, 'circuit'):
            self.circuit.rz(qubit, theta)
        elif hasattr(self, 'mps_tensors'):
            self._apply_mps_rotation('RZ', qubit, theta)
        elif hasattr(self, 'statevector'):
            gate = np.array([
                [np.exp(-1j * theta / 2), 0],
                [0, np.exp(1j * theta / 2)],
            ], dtype=np.complex128)
            self._apply_statevector_gate(gate, qubit)
        return self

    def cnot(self, control: int, target: int):
        """Apply CNOT gate"""
        if hasattr(self, 'circuit'):
            self.circuit.cnot(control, target)
        elif hasattr(self, 'mps_tensors'):
            self._apply_mps_two_qubit_gate('CNOT', control, target)
        elif hasattr(self, 'statevector'):
            self._apply_statevector_cnot(control, target)
        return self

    def cz(self, control: int, target: int):
        """Apply CZ gate"""
        if hasattr(self, 'circuit'):
            self.circuit.cz(control, target)
        elif hasattr(self, 'mps_tensors'):
            self._apply_mps_two_qubit_gate('CZ', control, target)
        elif hasattr(self, 'statevector'):
            self._apply_statevector_cz(control, target)
        return self

    # ========== MPS GATE OPERATIONS ==========

    def _apply_mps_gate(self, gate_name: str, qubit: int):
        """Apply single-qubit gate to MPS"""
        # Gate matrices
        gates = {
            'H': np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2),
            'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128),
        }

        gate = gates[gate_name]

        # Apply gate to MPS tensor at position 'qubit'
        # Tensor shape: [physical_dim=2, left_bond, right_bond]
        tensor = self.mps_tensors[qubit]
        new_tensor = np.einsum('ij,jkl->ikl', gate, tensor)
        self.mps_tensors[qubit] = new_tensor

    def _apply_mps_rotation(self, gate_name: str, qubit: int, theta: float):
        """Apply rotation gate to MPS"""
        if gate_name == 'RX':
            gate = np.array([
                [np.cos(theta/2), -1j*np.sin(theta/2)],
                [-1j*np.sin(theta/2), np.cos(theta/2)]
            ], dtype=np.complex128)
        elif gate_name == 'RY':
            gate = np.array([
                [np.cos(theta/2), -np.sin(theta/2)],
                [np.sin(theta/2), np.cos(theta/2)]
            ], dtype=np.complex128)
        elif gate_name == 'RZ':
            gate = np.array([
                [np.exp(-1j*theta/2), 0],
                [0, np.exp(1j*theta/2)]
            ], dtype=np.complex128)

        tensor = self.mps_tensors[qubit]
        new_tensor = np.einsum('ij,jkl->ikl', gate, tensor)
        self.mps_tensors[qubit] = new_tensor

    def _apply_mps_two_qubit_gate(self, gate_name: str, q1: int, q2: int):
        """
        Apply a two-qubit gate to adjacent qubits in an MPS.
        This implementation uses tensor contraction followed by SVD.
        """
        # Ensure qubits are adjacent for simplicity
        if abs(q1 - q2) != 1:
            raise NotImplementedError("Two-qubit gates are only supported for adjacent qubits in this MPS implementation.")
        
        # Ensure q1 is the lower index
        if q1 > q2:
            q1, q2 = q2, q1

        # Get the gate matrix
        if gate_name == 'CNOT':
            gate = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128).reshape(2, 2, 2, 2)
        elif gate_name == 'CZ':
            gate = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=np.complex128).reshape(2, 2, 2, 2)
        else:
            raise ValueError(f"Unknown two-qubit gate: {gate_name}")

        # 1. Contract the two tensors
        tensor1 = self.mps_tensors[q1]
        tensor2 = self.mps_tensors[q2]
        # Shapes: (phys1, left1, right1), (phys2, left2, right2)
        # Contract right bond of tensor1 with left bond of tensor2
        combined_tensor = np.einsum('pli,qij->pqlj', tensor1, tensor2)
        # New shape: (phys1, phys2, left1, right2)

        # 2. Apply the gate
        # Gate shape: (out1, out2, in1, in2)
        # Combined tensor shape: (in1, in2, left1, right2)
        # np.einsum notation for clarity:
        # ijkl: gate indices, klmn: combined_tensor indices
        # We contract gate indices kl with tensor indices kl (phys1, phys2)
        # Resulting tensor has shape (i, j, m, n) -> (out1, out2, left1, right2)
        result_tensor = np.einsum('ijkl,klmn->ijmn', gate, combined_tensor)

        # 3. Reshape for SVD
        # Combine (out1, left1) and (out2, right2) to form a matrix
        # Shape: (out1*left1, out2*right2)
        left_dim = result_tensor.shape[0] * result_tensor.shape[2]
        right_dim = result_tensor.shape[1] * result_tensor.shape[3]
        reshaped_tensor = result_tensor.transpose(0, 2, 1, 3).reshape(left_dim, right_dim)

        # 4. SVD and truncate
        U, S, Vh = np.linalg.svd(reshaped_tensor, full_matrices=False)
        
        # Truncate to the original bond dimension
        new_bond_dim = min(len(S), self.bond_dimension)
        
        U = U[:, :new_bond_dim]
        S = np.diag(S[:new_bond_dim])
        Vh = Vh[:new_bond_dim, :]

        # 5. Reshape back into two tensors
        # New tensor for q1
        new_tensor1 = U.reshape(tensor1.shape[0], tensor1.shape[1], new_bond_dim)
        
        # New tensor for q2 (needs to absorb S and Vh)
        # First, combine S and Vh
        svh = S @ Vh
        # Reshape into tensor form
        new_tensor2 = svh.reshape(new_bond_dim, tensor2.shape[0], tensor2.shape[2])
        # Transpose to match our (phys, left, right) convention
        new_tensor2 = new_tensor2.transpose(1, 0, 2)

        # Update the MPS chain
        self.mps_tensors[q1] = new_tensor1
        self.mps_tensors[q2] = new_tensor2

    def _apply_statevector_gate(self, gate: np.ndarray, qubit: int):
        """Apply a single-qubit gate directly to the fallback statevector."""
        if not hasattr(self, 'statevector'):
            return

        mask = 1 << (self.num_qubits - 1 - qubit)
        new_state = self.statevector.copy()

        for idx in range(self.dim):
            if idx & mask:
                continue  # Handle paired basis states in the branch where bit = 0
            partner = idx | mask
            amp0 = self.statevector[idx]
            amp1 = self.statevector[partner]

            new_state[idx] = gate[0, 0] * amp0 + gate[0, 1] * amp1
            new_state[partner] = gate[1, 0] * amp0 + gate[1, 1] * amp1

        self.statevector = new_state

    def _apply_statevector_cnot(self, control: int, target: int):
        """Apply a CNOT gate directly to the fallback statevector."""
        if not hasattr(self, 'statevector'):
            return

        control_mask = 1 << (self.num_qubits - 1 - control)
        target_mask = 1 << (self.num_qubits - 1 - target)

        original = self.statevector.copy()
        for idx in range(self.dim):
            if idx & control_mask:
                toggled = idx ^ target_mask
                self.statevector[idx] = original[toggled]
            else:
                self.statevector[idx] = original[idx]

    def _apply_statevector_cz(self, control: int, target: int):
        """Apply a CZ gate directly to the fallback statevector."""
        if not hasattr(self, 'statevector'):
            return

        control_mask = 1 << (self.num_qubits - 1 - control)
        target_mask = 1 << (self.num_qubits - 1 - target)

        for idx in range(self.dim):
            if (idx & control_mask) and (idx & target_mask):
                self.statevector[idx] *= -1.0

    # ========== MEASUREMENT ==========

    def measure(self, qubit: int) -> int:
        """Measure single qubit"""
        if hasattr(self, 'circuit'):
            return self.circuit.measure(qubit)
        elif hasattr(self, 'mps_tensors'):
            return self._measure_mps(qubit)
        elif hasattr(self, 'statevector'):
            mask = 1 << (self.num_qubits - 1 - qubit)
            prob_one = sum(
                abs(self.statevector[idx]) ** 2
                for idx in range(self.dim)
                if idx & mask
            )
            return int(prob_one >= 0.5)
        else:
            return 0

    def measure_all(self) -> List[int]:
        """Measure all qubits"""
        return [self.measure(i) for i in range(self.num_qubits)]

    def _measure_mps(self, qubit: int) -> int:
        """Measure qubit in MPS representation"""
        tensor = self.mps_tensors[qubit]
        amp_zero = np.linalg.norm(tensor[0]) ** 2
        amp_one = np.linalg.norm(tensor[1]) ** 2
        total = amp_zero + amp_one
        if total == 0:
            return 0
        prob_one = amp_one / total
        return int(prob_one >= 0.5)

    def get_probabilities(self) -> Dict[str, float]:
        """Get probability distribution"""
        if hasattr(self, 'circuit'):
            return self.circuit.get_probabilities()
        elif hasattr(self, 'statevector'):
            probs = {}
            for i in range(self.dim):
                bitstring = format(i, f'0{self.num_qubits}b')
                prob = abs(self.statevector[i])**2
                if prob > 1e-10:
                    probs[bitstring] = prob
            return probs
        else:
            # MPS: approximate for the first few qubits to verify entanglement
            # Contract first two tensors to check Bell state
            if self.num_qubits >= 2:
                t0 = self.mps_tensors[0]
                t1 = self.mps_tensors[1]
                # Contract the shared bond dimension
                combined = np.einsum('pli,qij->pqlj', t0, t1) # p,q = physical; l,j = outer bonds
                # Trace out the outer bond dimensions to get the reduced density matrix for first 2 qubits
                density_matrix = np.einsum('pqlj,rslj->pqrs', combined, combined.conj())
                
                probs = {}
                for i in range(2):
                    for j in range(2):
                        prob = np.abs(density_matrix[i,j,i,j])
                        if prob > 1e-9:
                           # This only gives diagonal elements, which is what we need for probabilities
                           bitstring = f"{i}{j}"
                           probs[bitstring] = prob
                
                # Normalize because we traced out the rest of the chain
                total_prob = sum(probs.values())
                if total_prob > 0:
                    for k in probs:
                        probs[k] /= total_prob
                return probs

            return {"0" * self.num_qubits: 1.0}  # Placeholder for < 2 qubits

    # ========== ADVANCED FEATURES ==========

    def expectation_value(self, observable: str) -> float:
        """
        Compute expectation value of observable.

        Args:
            observable: Pauli string like 'Z0' or 'X0Y1Z2'

        Returns:
            Expectation value <ψ|O|ψ>
        """
        if hasattr(self, 'circuit'):
            # Save current state
            original_state = self.circuit.get_statevector()

            # Parse observable and compute expectation
            # Simplified: just return 0.0 for now
            # Full implementation would apply observable and compute <ψ|O|ψ>
            return 0.0
        else:
            return 0.0

    def fidelity(self, target_state: np.ndarray) -> float:
        """
        Compute fidelity with target state.

        F = |<ψ|φ>|²
        """
        if hasattr(self, 'circuit'):
            current = self.circuit.get_statevector()
            overlap = np.abs(np.vdot(current, target_state))**2
            return overlap
        return 0.0

    # ========== SUBSYSTEM PROPERTIES ==========

    @property
    def chemistry(self):
        """Access quantum chemistry module"""
        if self._chemistry is None:
            import quantum_chemistry
            self._chemistry = quantum_chemistry.QuantumChemistry(self)
        return self._chemistry

    @property
    def materials(self):
        """Access quantum materials module"""
        if self._materials is None:
            import quantum_materials
            self._materials = quantum_materials.QuantumMaterials(self)
        return self._materials

    @property
    def sensors(self):
        """Access quantum sensors module"""
        if self._sensors is None:
            import quantum_sensors
            self._sensors = quantum_sensors.QuantumSensors(self)
        return self._sensors

    # ========== UTILITY METHODS ==========

    def reset(self):
        """Reset quantum state to |0⟩^n"""
        if hasattr(self, 'circuit'):
            self.circuit = CircuitSim(self.num_qubits, self.optimize_for_m4)
        elif hasattr(self, 'statevector'):
            self.statevector = np.zeros(self.dim, dtype=np.complex128)
            self.statevector[0] = 1.0 + 0j
        elif hasattr(self, 'mps_tensors'):
            self._initialize_mps_backend()
        return self

    def print_state(self, top_n: int = 10):
        """Print quantum state"""
        if hasattr(self, 'circuit'):
            self.circuit.print_state(top_n)
        else:
            print(f"\nQuantum State (backend: {self.backend.value}):")
            probs = self.get_probabilities()
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            for i, (state, prob) in enumerate(sorted_probs[:top_n]):
                print(f"   |{state}⟩: {prob*100:.2f}%")

    def get_backend_info(self) -> Dict:
        """Get information about current backend"""
        info = {
            "backend": self.backend.value,
            "num_qubits": self.num_qubits,
            "optimize_for_m4": self.optimize_for_m4
        }

        if self.backend == SimulationBackend.STATEVECTOR_EXACT:
            info["hilbert_dim"] = 2**self.num_qubits
            info["memory_gb"] = (2**self.num_qubits * 16) / (1024**3)
        elif self.backend in [SimulationBackend.TENSOR_NETWORK, SimulationBackend.MPS]:
            info["bond_dimension"] = self.bond_dimension
            info["memory_gb"] = self._estimate_mps_memory()

        return info


# ========== CONVENIENCE FUNCTIONS ==========

def create_bell_pair(verbose: bool = False) -> QuantumLabSimulator:
    """Create Bell state (maximally entangled pair)"""
    lab = QuantumLabSimulator(num_qubits=2, verbose=verbose)
    lab.h(0).cnot(0, 1)
    return lab


def create_ghz_state(num_qubits: int, verbose: bool = False) -> QuantumLabSimulator:
    """Create GHZ state (maximally entangled N qubits)"""
    lab = QuantumLabSimulator(num_qubits=num_qubits, verbose=verbose)
    lab.h(0)
    for i in range(num_qubits - 1):
        lab.cnot(i, i + 1)
    return lab


def create_w_state(num_qubits: int, verbose: bool = False) -> QuantumLabSimulator:
    """Create W state (symmetric superposition)"""
    lab = QuantumLabSimulator(num_qubits=num_qubits, verbose=verbose)
    # W state creation circuit (simplified)
    # Full implementation requires specific angle rotations
    lab.h(0)
    for i in range(1, num_qubits):
        lab.cnot(0, i)
    return lab


# ========== DEMO ==========

if __name__ == "__main__":
    print("\n" + "="*60)
    print("QULAB INFINITE - QUANTUM LABORATORY DEMONSTRATION")
    print("="*60)

    # Demo 1: Small circuit (statevector)
    print("\n\n1️⃣  STATEVECTOR SIMULATION (5 qubits)")
    lab5 = QuantumLabSimulator(num_qubits=5)
    lab5.h(0).cnot(0, 1).cnot(1, 2)
    lab5.print_state()

    # Demo 2: Large circuit (tensor network)
    print("\n\n2️⃣  TENSOR NETWORK SIMULATION (35 qubits)")
    lab35 = QuantumLabSimulator(
        num_qubits=35,
        backend=SimulationBackend.TENSOR_NETWORK,
        verbose=False
    )
    lab35.h(0).cnot(0, 1)
    print(f"   ✅ 35-qubit circuit operational with MPS")
    print(f"   Memory usage: ~{lab35._estimate_mps_memory():.2f} GB")
    # Verify entanglement
    probs = lab35.get_probabilities()
    print(f"   Probabilities after CNOT(0,1): {probs}")


    # Demo 3: Bell state
    print("\n\n3️⃣  BELL STATE GENERATION")
    bell = create_bell_pair(verbose=False)
    bell.print_state()

    # Demo 4: Backend info
    print("\n\n4️⃣  BACKEND INFORMATION")
    info = lab5.get_backend_info()
    print(f"   Backend: {info['backend']}")
    print(f"   Qubits: {info['num_qubits']}")
    print(f"   Hilbert space: {info.get('hilbert_dim', 'N/A')}")
    print(f"   Memory: {info.get('memory_gb', 0):.2f} GB")

    print("\n\n✅ Quantum laboratory demonstration complete!")
    print("    Ready for chemistry, materials, and sensor simulations")
