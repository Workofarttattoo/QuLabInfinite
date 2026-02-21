"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

tomography - Part of Quantum Lab
"""

"""
Quantum state tomography and fidelity calculation.

Implements quantum state tomography for reconstructing unknown quantum states
and calculating fidelity between quantum states.

References:
- James, D. F. V., et al. (2001). Measurement of qubits. Physical Review A, 64(5), 052312.
- Nielsen, M. A., & Chuang, I. L. (2010). Quantum computation and quantum information.
"""

from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity
from qiskit_aer import AerSimulator
from scipy.optimize import minimize
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class TomographyResult(BaseModel):
    """Result of quantum state tomography."""
    
    reconstructed_state: np.ndarray = Field(..., description="Reconstructed density matrix")
    fidelity: float = Field(..., description="Fidelity of reconstruction")
    measurement_counts: Dict[str, Dict[str, int]] = Field(..., description="Measurement counts by basis")
    confidence_interval: Tuple[float, float] = Field(..., description="95% confidence interval for fidelity")
    
    class Config:
        arbitrary_types_allowed = True


class StateTomography:
    """
    Quantum state tomography for single qubit states.
    
    Implements maximum likelihood estimation (MLE) for reconstructing
    unknown quantum states from measurement data in multiple bases.
    
    The standard tomography protocol measures in three bases:
    - Computational basis {|0⟩, |1⟩}
    - X basis {|+⟩, |-⟩} where |±⟩ = (|0⟩ ± |1⟩)/√2
    - Y basis {|+i⟩, |-i⟩} where |±i⟩ = (|0⟩ ± i|1⟩)/√2
    """
    
    def __init__(self, backend: Optional[AerSimulator] = None):
        """
        Initialize state tomography.
        
        Args:
            backend: Qiskit backend for simulation
        """
        self.backend = backend or AerSimulator()
        self.measurement_bases = self._create_measurement_bases()
    
    def _create_measurement_bases(self) -> Dict[str, QuantumCircuit]:
        """Create measurement circuits for different bases."""
        bases = {}
        
        # Computational basis (Z basis)
        qc_z = QuantumCircuit(1, 1)
        qc_z.measure(0, 0)
        bases['Z'] = qc_z
        
        # X basis
        qc_x = QuantumCircuit(1, 1)
        qc_x.h(0)  # Hadamard to rotate to X basis
        qc_x.measure(0, 0)
        bases['X'] = qc_x
        
        # Y basis
        qc_y = QuantumCircuit(1, 1)
        qc_y.sdg(0)  # S† gate
        qc_y.h(0)    # Hadamard gate
        qc_y.measure(0, 0)
        bases['Y'] = qc_y
        
        return bases
    
    def measure_state(self, state: Union[Statevector, np.ndarray], 
                     shots: int = 1000) -> Dict[str, Dict[str, int]]:
        """
        Measure quantum state in multiple bases.
        
        Args:
            state: Quantum state to measure
            shots: Number of measurement shots per basis
            
        Returns:
            Dictionary of measurement counts by basis
        """
        if isinstance(state, np.ndarray):
            state = Statevector(state)
        
        # Create circuit with state initialization
        qreg = QuantumRegister(1, 'q')
        creg = ClassicalRegister(1, 'c')
        qc_init = QuantumCircuit(qreg, creg)
        qc_init.initialize(state.data, qreg[0])
        
        measurement_counts = {}
        
        for basis_name, basis_circuit in self.measurement_bases.items():
            # Combine initialization and measurement
            qc = qc_init.compose(basis_circuit, qubits=[0], clbits=[0])
            
            # Execute measurement
            job = self.backend.run(qc, shots=shots)
            result = job.result()
            counts = result.get_counts()
            
            # Normalize counts
            total = sum(counts.values())
            normalized_counts = {k: v / total for k, v in counts.items()}
            measurement_counts[basis_name] = normalized_counts
        
        return measurement_counts
    
    def reconstruct_state(self, measurement_counts: Dict[str, Dict[str, int]], 
                         method: str = 'mle') -> TomographyResult:
        """
        Reconstruct quantum state from measurement data.
        
        Args:
            measurement_counts: Measurement counts by basis
            method: Reconstruction method ('mle' or 'linear')
            
        Returns:
            TomographyResult with reconstructed state and fidelity
        """
        if method == 'mle':
            return self._reconstruct_mle(measurement_counts)
        elif method == 'linear':
            return self._reconstruct_linear(measurement_counts)
        else:
            raise ValueError(f"Unknown reconstruction method: {method}")
    
    def _reconstruct_linear(self, measurement_counts: Dict[str, Dict[str, int]]) -> TomographyResult:
        """
        Linear reconstruction method.
        
        Uses the linear inversion method to reconstruct the density matrix
        from measurement probabilities.
        """
        # Extract probabilities
        p_z0 = measurement_counts['Z'].get('0', 0.0)
        p_z1 = measurement_counts['Z'].get('1', 0.0)
        p_x0 = measurement_counts['X'].get('0', 0.0)
        p_x1 = measurement_counts['X'].get('1', 0.0)
        p_y0 = measurement_counts['Y'].get('0', 0.0)
        p_y1 = measurement_counts['Y'].get('1', 0.0)
        
        # Calculate Pauli expectation values
        # ⟨Z⟩ = p_z0 - p_z1
        # ⟨X⟩ = p_x0 - p_x1  
        # ⟨Y⟩ = p_y0 - p_y1
        z_exp = p_z0 - p_z1
        x_exp = p_x0 - p_x1
        y_exp = p_y0 - p_y1
        
        # Reconstruct density matrix
        # ρ = (I + ⟨X⟩X + ⟨Y⟩Y + ⟨Z⟩Z) / 2
        pauli_x = np.array([[0, 1], [1, 0]])
        pauli_y = np.array([[0, -1j], [1j, 0]])
        pauli_z = np.array([[1, 0], [0, -1]])
        identity = np.eye(2)
        
        rho = (identity + x_exp * pauli_x + y_exp * pauli_y + z_exp * pauli_z) / 2
        
        # Ensure positive semidefinite (project onto valid density matrices)
        rho = self._project_to_valid_density_matrix(rho)
        
        # Calculate fidelity (simplified - in practice you'd compare with known state)
        fidelity_val = np.real(np.trace(rho @ rho))  # Purity
        
        # Calculate confidence interval (simplified)
        n_shots = 1000  # Assume 1000 shots per basis
        std_error = np.sqrt(fidelity_val * (1 - fidelity_val) / n_shots)
        confidence_interval = (fidelity_val - 1.96 * std_error, 
                             fidelity_val + 1.96 * std_error)
        
        return TomographyResult(
            reconstructed_state=rho,
            fidelity=fidelity_val,
            measurement_counts=measurement_counts,
            confidence_interval=confidence_interval
        )
    
    def _reconstruct_mle(self, measurement_counts: Dict[str, Dict[str, int]]) -> TomographyResult:
        """
        Maximum likelihood estimation reconstruction.
        
        Uses MLE to find the density matrix that maximizes the likelihood
        of observing the given measurement data.
        """
        # Initial guess (maximally mixed state)
        rho_init = np.eye(2) / 2
        
        # Define likelihood function
        def negative_log_likelihood(rho_params):
            # Convert parameters to density matrix
            rho = self._params_to_density_matrix(rho_params)
            
            # Calculate expected probabilities for each basis
            log_likelihood = 0.0
            
            for basis_name, counts in measurement_counts.items():
                if basis_name == 'Z':
                    # Computational basis
                    p0_expected = np.real(rho[0, 0])
                    p1_expected = np.real(rho[1, 1])
                elif basis_name == 'X':
                    # X basis: |±⟩ = (|0⟩ ± |1⟩)/√2
                    p0_expected = np.real((rho[0, 0] + rho[1, 1] + 2 * np.real(rho[0, 1])) / 2)
                    p1_expected = np.real((rho[0, 0] + rho[1, 1] - 2 * np.real(rho[0, 1])) / 2)
                elif basis_name == 'Y':
                    # Y basis: |±i⟩ = (|0⟩ ± i|1⟩)/√2
                    p0_expected = np.real((rho[0, 0] + rho[1, 1] + 2 * np.imag(rho[0, 1])) / 2)
                    p1_expected = np.real((rho[0, 0] + rho[1, 1] - 2 * np.imag(rho[0, 1])) / 2)
                
                # Add to log likelihood
                n_shots = 1000  # Assume 1000 shots
                n0 = counts.get('0', 0) * n_shots
                n1 = counts.get('1', 0) * n_shots
                
                if p0_expected > 1e-10:
                    log_likelihood += n0 * np.log(p0_expected)
                if p1_expected > 1e-10:
                    log_likelihood += n1 * np.log(p1_expected)
            
            return -log_likelihood
        
        # Optimize
        initial_params = self._density_matrix_to_params(rho_init)
        result = minimize(negative_log_likelihood, initial_params, method='L-BFGS-B')
        
        # Convert back to density matrix
        rho = self._params_to_density_matrix(result.x)
        
        # Calculate fidelity
        fidelity_val = np.real(np.trace(rho @ rho))
        
        # Calculate confidence interval
        n_shots = 1000
        std_error = np.sqrt(fidelity_val * (1 - fidelity_val) / n_shots)
        confidence_interval = (fidelity_val - 1.96 * std_error, 
                             fidelity_val + 1.96 * std_error)
        
        return TomographyResult(
            reconstructed_state=rho,
            fidelity=fidelity_val,
            measurement_counts=measurement_counts,
            confidence_interval=confidence_interval
        )
    
    def _density_matrix_to_params(self, rho: np.ndarray) -> np.ndarray:
        """Convert density matrix to parameter vector for optimization."""
        # For 2x2 density matrix, we need 3 real parameters
        # ρ = [[a, b+ic], [b-ic, 1-a]] where a ∈ [0,1], b,c ∈ ℝ
        a = np.real(rho[0, 0])
        b = np.real(rho[0, 1])
        c = np.imag(rho[0, 1])
        return np.array([a, b, c])
    
    def _params_to_density_matrix(self, params: np.ndarray) -> np.ndarray:
        """Convert parameter vector to density matrix."""
        a, b, c = params
        # Ensure a ∈ [0, 1]
        a = max(0, min(1, a))
        
        rho = np.array([
            [a, b + 1j * c],
            [b - 1j * c, 1 - a]
        ])
        
        return self._project_to_valid_density_matrix(rho)
    
    def _project_to_valid_density_matrix(self, rho: np.ndarray) -> np.ndarray:
        """Project matrix onto the set of valid density matrices."""
        # Ensure Hermitian
        rho = (rho + rho.conj().T) / 2
        
        # Ensure trace = 1
        rho = rho / np.trace(rho)
        
        # Ensure positive semidefinite
        eigenvals, eigenvecs = np.linalg.eigh(rho)
        eigenvals = np.maximum(eigenvals, 0)  # Set negative eigenvalues to 0
        rho = eigenvecs @ np.diag(eigenvals) @ eigenvecs.conj().T
        
        # Renormalize
        rho = rho / np.trace(rho)
        
        return rho


class FidelityCalculator:
    """
    Calculate fidelity between quantum states.
    
    Implements various fidelity measures:
    - State fidelity: F(ρ,σ) = |⟨ψ|φ⟩|² for pure states
    - Uhlmann fidelity: F(ρ,σ) = Tr(√(√ρ σ √ρ))² for mixed states
    """
    
    @staticmethod
    def state_fidelity(state1: Union[Statevector, np.ndarray], 
                      state2: Union[Statevector, np.ndarray]) -> float:
        """
        Calculate fidelity between two pure states.
        
        F(ψ,φ) = |⟨ψ|φ⟩|²
        
        Args:
            state1: First quantum state
            state2: Second quantum state
            
        Returns:
            Fidelity value between 0 and 1
        """
        if isinstance(state1, np.ndarray):
            state1 = Statevector(state1)
        if isinstance(state2, np.ndarray):
            state2 = Statevector(state2)
        
        overlap = np.abs(state1.inner(state2))**2
        return float(overlap)
    
    @staticmethod
    def uhlmann_fidelity(rho1: np.ndarray, rho2: np.ndarray) -> float:
        """
        Calculate Uhlmann fidelity between two density matrices.
        
        F(ρ,σ) = Tr(√(√ρ σ √ρ))²
        
        Args:
            rho1: First density matrix
            rho2: Second density matrix
            
        Returns:
            Fidelity value between 0 and 1
        """
        # Use Qiskit's fidelity function for numerical stability
        return float(state_fidelity(DensityMatrix(rho1), DensityMatrix(rho2)))
    
    @staticmethod
    def average_fidelity(states1: List[Union[Statevector, np.ndarray]], 
                        states2: List[Union[Statevector, np.ndarray]]) -> float:
        """
        Calculate average fidelity over a set of states.
        
        Args:
            states1: List of first states
            states2: List of second states
            
        Returns:
            Average fidelity
        """
        if len(states1) != len(states2):
            raise ValueError("Lists must have the same length")
        
        fidelities = []
        for s1, s2 in zip(states1, states2):
            f = FidelityCalculator.state_fidelity(s1, s2)
            fidelities.append(f)
        
        return np.mean(fidelities)
    
    @staticmethod
    def process_fidelity(ideal_circuit: QuantumCircuit, 
                        noisy_circuit: QuantumCircuit,
                        initial_states: List[Statevector]) -> float:
        """
        Calculate process fidelity between ideal and noisy quantum circuits.
        
        Args:
            ideal_circuit: Ideal quantum circuit
            noisy_circuit: Noisy quantum circuit
            initial_states: List of initial states to test
            
        Returns:
            Process fidelity
        """
        fidelities = []
        
        for initial_state in initial_states:
            # Simulate ideal circuit
            ideal_state = initial_state.evolve(ideal_circuit)
            
            # Simulate noisy circuit
            noisy_state = initial_state.evolve(noisy_circuit)
            
            # Calculate fidelity
            f = FidelityCalculator.state_fidelity(ideal_state, noisy_state)
            fidelities.append(f)
        
        return np.mean(fidelities)
