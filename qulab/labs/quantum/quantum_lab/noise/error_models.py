"""
Quantum error models and noise simulation.

Implements various quantum error models including depolarizing noise,
amplitude damping, phase damping, and custom noise models for
realistic quantum circuit simulation.

References:
- Nielsen, M. A., & Chuang, I. L. (2010). Quantum computation and quantum information.
- Preskill, J. (2018). Quantum computing in the NISQ era and beyond.
"""

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error, 
    amplitude_damping_error, 
    phase_damping_error,
    pauli_error,
    thermal_relaxation_error
)
from qiskit.quantum_info import Kraus
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class NoiseModelConfig(BaseModel):
    """Configuration for quantum noise models."""
    
    depolarizing_prob: float = Field(0.0, ge=0.0, le=1.0, description="Depolarizing error probability")
    amplitude_damping_gamma: float = Field(0.0, ge=0.0, le=1.0, description="Amplitude damping rate")
    phase_damping_gamma: float = Field(0.0, ge=0.0, le=1.0, description="Phase damping rate")
    t1: float = Field(50e-6, gt=0.0, description="T1 relaxation time in seconds")
    t2: float = Field(50e-6, gt=0.0, description="T2 dephasing time in seconds")
    gate_time: float = Field(1e-6, gt=0.0, description="Gate time in seconds")
    readout_error: float = Field(0.0, ge=0.0, le=1.0, description="Readout error probability")
    
    class Config:
        arbitrary_types_allowed = True


class DepolarizingChannel:
    """
    Depolarizing noise channel.
    
    The depolarizing channel is defined as:
    E(ρ) = (1-p)ρ + p/3(XρX + YρY + ZρZ)
    
    where p is the depolarizing probability and X, Y, Z are Pauli matrices.
    """
    
    def __init__(self, probability: float):
        """
        Initialize depolarizing channel.
        
        Args:
            probability: Depolarizing probability (0 ≤ p ≤ 1)
        """
        if not 0 <= probability <= 1:
            raise ValueError("Depolarizing probability must be between 0 and 1")
        self.probability = probability
    
    def apply(self, rho: np.ndarray) -> np.ndarray:
        """
        Apply depolarizing channel to density matrix.
        
        Args:
            rho: Input density matrix
            
        Returns:
            Output density matrix after depolarizing noise
        """
        # Pauli matrices
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        
        # Apply depolarizing channel
        result = (1 - self.probability) * rho
        result += (self.probability / 3) * (X @ rho @ X + Y @ rho @ Y + Z @ rho @ Z)
        
        return result
    
    def to_qiskit_error(self) -> pauli_error:
        """Convert to Qiskit error model."""
        return depolarizing_error(self.probability, 1)


class AmplitudeDampingChannel:
    """
    Amplitude damping noise channel.
    
    Models energy dissipation in quantum systems. The Kraus operators are:
    E₀ = [[1, 0], [0, √(1-γ)]]
    E₁ = [[0, √γ], [0, 0]]
    
    where γ is the damping rate.
    """
    
    def __init__(self, gamma: float):
        """
        Initialize amplitude damping channel.
        
        Args:
            gamma: Damping rate (0 ≤ γ ≤ 1)
        """
        if not 0 <= gamma <= 1:
            raise ValueError("Damping rate must be between 0 and 1")
        self.gamma = gamma
    
    def apply(self, rho: np.ndarray) -> np.ndarray:
        """
        Apply amplitude damping channel to density matrix.
        
        Args:
            rho: Input density matrix
            
        Returns:
            Output density matrix after amplitude damping
        """
        # Kraus operators
        E0 = np.array([[1, 0], [0, np.sqrt(1 - self.gamma)]])
        E1 = np.array([[0, np.sqrt(self.gamma)], [0, 0]])
        
        # Apply channel
        result = E0 @ rho @ E0.conj().T + E1 @ rho @ E1.conj().T
        
        return result
    
    def to_qiskit_error(self) -> amplitude_damping_error:
        """Convert to Qiskit error model."""
        return amplitude_damping_error(self.gamma)


class PhaseDampingChannel:
    """
    Phase damping noise channel.
    
    Models pure dephasing without energy loss. The Kraus operators are:
    E₀ = [[1, 0], [0, √(1-γ)]]
    E₁ = [[0, 0], [0, √γ]]
    
    where γ is the dephasing rate.
    """
    
    def __init__(self, gamma: float):
        """
        Initialize phase damping channel.
        
        Args:
            gamma: Dephasing rate (0 ≤ γ ≤ 1)
        """
        if not 0 <= gamma <= 1:
            raise ValueError("Dephasing rate must be between 0 and 1")
        self.gamma = gamma
    
    def apply(self, rho: np.ndarray) -> np.ndarray:
        """
        Apply phase damping channel to density matrix.
        
        Args:
            rho: Input density matrix
            
        Returns:
            Output density matrix after phase damping
        """
        # Kraus operators
        E0 = np.array([[1, 0], [0, np.sqrt(1 - self.gamma)]])
        E1 = np.array([[0, 0], [0, np.sqrt(self.gamma)]])
        
        # Apply channel
        result = E0 @ rho @ E0.conj().T + E1 @ rho @ E1.conj().T
        
        return result
    
    def to_qiskit_error(self) -> phase_damping_error:
        """Convert to Qiskit error model."""
        return phase_damping_error(self.gamma)


class NoiseModel:
    """
    Comprehensive noise model for quantum circuits.
    
    Combines multiple noise sources including depolarizing noise,
    amplitude damping, phase damping, and readout errors.
    """
    
    def __init__(self, config: NoiseModelConfig):
        """
        Initialize noise model.
        
        Args:
            config: Noise model configuration
        """
        self.config = config
        self.qiskit_noise_model = self._create_qiskit_noise_model()
    
    def _create_qiskit_noise_model(self) -> NoiseModel:
        """Create Qiskit noise model from configuration."""
        noise_model = NoiseModel()
        
        # Add depolarizing noise
        if self.config.depolarizing_prob > 0:
            depol_error = depolarizing_error(self.config.depolarizing_prob, 1)
            noise_model.add_all_qubit_quantum_error(depol_error, ['h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg'])
        
        # Add amplitude damping
        if self.config.amplitude_damping_gamma > 0:
            amp_damp_error = amplitude_damping_error(self.config.amplitude_damping_gamma)
            noise_model.add_all_qubit_quantum_error(amp_damp_error, ['h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg'])
        
        # Add phase damping
        if self.config.phase_damping_gamma > 0:
            phase_damp_error = phase_damping_error(self.config.phase_damping_gamma)
            noise_model.add_all_qubit_quantum_error(phase_damp_error, ['h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg'])
        
        # Add thermal relaxation
        if self.config.t1 > 0 and self.config.t2 > 0:
            thermal_error = thermal_relaxation_error(
                self.config.t1, 
                self.config.t2, 
                self.config.gate_time
            )
            noise_model.add_all_qubit_quantum_error(thermal_error, ['h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg'])
        
        # Add readout errors
        if self.config.readout_error > 0:
            readout_error_matrix = [
                [1 - self.config.readout_error, self.config.readout_error],
                [self.config.readout_error, 1 - self.config.readout_error]
            ]
            noise_model.add_all_qubit_readout_error(readout_error_matrix)
        
        return noise_model
    
    def create_noisy_backend(self) -> AerSimulator:
        """
        Create noisy backend with this noise model.
        
        Returns:
            AerSimulator with noise model applied
        """
        return AerSimulator(noise_model=self.qiskit_noise_model)
    
    def get_error_rates(self) -> Dict[str, float]:
        """
        Get error rates for different noise sources.
        
        Returns:
            Dictionary of error rates
        """
        return {
            "depolarizing": self.config.depolarizing_prob,
            "amplitude_damping": self.config.amplitude_damping_gamma,
            "phase_damping": self.config.phase_damping_gamma,
            "readout": self.config.readout_error,
            "t1": self.config.t1,
            "t2": self.config.t2,
        }
    
    def calculate_fidelity_degradation(self, circuit_depth: int) -> float:
        """
        Estimate fidelity degradation for a circuit of given depth.
        
        Args:
            circuit_depth: Number of gates in the circuit
            
        Returns:
            Estimated fidelity after noise
        """
        # Simplified model: exponential decay with circuit depth
        # F ≈ F₀^d where F₀ is single-gate fidelity and d is depth
        
        # Calculate single-gate fidelity
        single_gate_fidelity = 1.0
        
        # Depolarizing contribution
        single_gate_fidelity *= (1 - self.config.depolarizing_prob)
        
        # Amplitude damping contribution
        single_gate_fidelity *= (1 - self.config.amplitude_damping_gamma / 2)
        
        # Phase damping contribution
        single_gate_fidelity *= (1 - self.config.phase_damping_gamma / 2)
        
        # Thermal relaxation contribution
        if self.config.t1 > 0:
            thermal_decay = np.exp(-self.config.gate_time / self.config.t1)
            single_gate_fidelity *= thermal_decay
        
        # Overall fidelity
        total_fidelity = single_gate_fidelity ** circuit_depth
        
        return max(0.0, min(1.0, total_fidelity))
    
    def optimize_circuit_depth(self, target_fidelity: float) -> int:
        """
        Find maximum circuit depth for target fidelity.
        
        Args:
            target_fidelity: Target fidelity threshold
            
        Returns:
            Maximum circuit depth
        """
        if target_fidelity <= 0 or target_fidelity >= 1:
            raise ValueError("Target fidelity must be between 0 and 1")
        
        # Binary search for maximum depth
        low, high = 1, 1000
        
        while low < high:
            mid = (low + high + 1) // 2
            fidelity = self.calculate_fidelity_degradation(mid)
            
            if fidelity >= target_fidelity:
                low = mid
            else:
                high = mid - 1
        
        return low


class CustomNoiseModel:
    """
    Custom noise model with user-defined Kraus operators.
    
    Allows definition of arbitrary quantum channels using Kraus operators.
    """
    
    def __init__(self, kraus_operators: List[np.ndarray]):
        """
        Initialize custom noise model.
        
        Args:
            kraus_operators: List of Kraus operators (must satisfy ∑ᵢ Kᵢ†Kᵢ = I)
        """
        self.kraus_operators = kraus_operators
        self._validate_kraus_operators()
    
    def _validate_kraus_operators(self) -> None:
        """Validate that Kraus operators form a valid quantum channel."""
        # Check completeness relation: ∑ᵢ Kᵢ†Kᵢ = I
        completeness = sum(K.conj().T @ K for K in self.kraus_operators)
        identity = np.eye(completeness.shape[0])
        
        if not np.allclose(completeness, identity, atol=1e-10):
            raise ValueError("Kraus operators do not satisfy completeness relation")
    
    def apply(self, rho: np.ndarray) -> np.ndarray:
        """
        Apply custom noise channel to density matrix.
        
        Args:
            rho: Input density matrix
            
        Returns:
            Output density matrix after noise
        """
        result = sum(K @ rho @ K.conj().T for K in self.kraus_operators)
        return result
    
    def to_qiskit_kraus(self) -> Kraus:
        """Convert to Qiskit Kraus representation."""
        return Kraus(self.kraus_operators)
