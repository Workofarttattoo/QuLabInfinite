"""
Bloch sphere representation and phase gate operations.

Implements Bloch sphere visualization and phase gate operations for quantum states.
The Bloch sphere provides a geometric representation of quantum states on a unit sphere.
"""

from typing import Tuple, Optional, List
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from pydantic import BaseModel, Field


class BlochSphere(BaseModel):
    """
    Bloch sphere representation of quantum states.
    
    A quantum state |ψ⟩ = α|0⟩ + β|1⟩ can be represented on the Bloch sphere
    with coordinates (x, y, z) where:
    - x = 2*Re(α*β*)
    - y = 2*Im(α*β*)  
    - z = |α|² - |β|²
    
    The state is normalized: x² + y² + z² = 1
    """
    
    x: float = Field(..., description="X coordinate on Bloch sphere")
    y: float = Field(..., description="Y coordinate on Bloch sphere") 
    z: float = Field(..., description="Z coordinate on Bloch sphere")
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 1.0, **data):
        super().__init__(x=x, y=y, z=z, **data)
        self._normalize()
    
    def _normalize(self) -> None:
        """Normalize the Bloch vector to unit length."""
        norm = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        if norm > 1e-10:  # Avoid division by zero
            self.x /= norm
            self.y /= norm
            self.z /= norm
    
    @classmethod
    def from_statevector(cls, statevector: Statevector) -> "BlochSphere":
        """
        Create Bloch sphere from Qiskit Statevector.
        
        Args:
            statevector: Qiskit Statevector object
            
        Returns:
            BlochSphere representation
        """
        if len(statevector) != 2:
            raise ValueError("Statevector must be 2-dimensional for single qubit")
        
        alpha = statevector[0]
        beta = statevector[1]
        
        x = 2 * np.real(alpha * np.conj(beta))
        y = 2 * np.imag(alpha * np.conj(beta))
        z = abs(alpha)**2 - abs(beta)**2
        
        return cls(x=x, y=y, z=z)
    
    @classmethod
    def from_amplitudes(cls, alpha: complex, beta: complex) -> "BlochSphere":
        """
        Create Bloch sphere from complex amplitudes.
        
        Args:
            alpha: Amplitude of |0⟩ state
            beta: Amplitude of |1⟩ state
            
        Returns:
            BlochSphere representation
        """
        # Normalize the state
        norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
        if norm > 1e-10:
            alpha /= norm
            beta /= norm
        
        x = 2 * np.real(alpha * np.conj(beta))
        y = 2 * np.imag(alpha * np.conj(beta))
        z = abs(alpha)**2 - abs(beta)**2
        
        return cls(x=x, y=y, z=z)
    
    def to_statevector(self) -> Statevector:
        """
        Convert Bloch sphere to Qiskit Statevector.
        
        Returns:
            Statevector representation
        """
        # Convert from Bloch coordinates to amplitudes
        # z = |α|² - |β|², x + iy = 2α*β
        # Solving: α = √((1+z)/2), β = (x+iy)/(2α*)
        
        if abs(self.z + 1) < 1e-10:  # South pole
            alpha = 0.0
            beta = 1.0
        else:
            alpha = np.sqrt((1 + self.z) / 2)
            if abs(alpha) > 1e-10:
                beta = (self.x + 1j * self.y) / (2 * np.conj(alpha))
            else:
                beta = 1.0
        
        return Statevector([alpha, beta])
    
    def to_amplitudes(self) -> Tuple[complex, complex]:
        """
        Convert Bloch sphere to complex amplitudes.
        
        Returns:
            Tuple of (alpha, beta) amplitudes
        """
        statevector = self.to_statevector()
        return statevector[0], statevector[1]
    
    def apply_rotation(self, axis: Tuple[float, float, float], angle: float) -> "BlochSphere":
        """
        Apply rotation around arbitrary axis.
        
        Args:
            axis: Rotation axis (x, y, z)
            angle: Rotation angle in radians
            
        Returns:
            New BlochSphere after rotation
        """
        # Normalize axis
        axis_norm = np.sqrt(sum(a**2 for a in axis))
        if axis_norm > 1e-10:
            axis = tuple(a / axis_norm for a in axis)
        
        # Rotation matrix using Rodrigues' formula
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # Cross product matrix
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        # Rotation matrix
        R = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
        
        # Apply rotation
        old_vector = np.array([self.x, self.y, self.z])
        new_vector = np.dot(R, old_vector)
        
        return BlochSphere(x=new_vector[0], y=new_vector[1], z=new_vector[2])
    
    def plot_matplotlib(self, ax: Optional[plt.Axes] = None, 
                       color: str = 'red', size: int = 100) -> plt.Axes:
        """
        Plot Bloch sphere using matplotlib.
        
        Args:
            ax: Matplotlib axes (3D)
            color: Point color
            size: Point size
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # Draw sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='lightblue')
        
        # Draw axes
        ax.plot([-1, 1], [0, 0], [0, 0], 'k-', alpha=0.3)
        ax.plot([0, 0], [-1, 1], [0, 0], 'k-', alpha=0.3)
        ax.plot([0, 0], [0, 0], [-1, 1], 'k-', alpha=0.3)
        
        # Plot state
        ax.scatter([self.x], [self.y], [self.z], c=color, s=size)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Bloch Sphere')
        
        return ax
    
    def plot_plotly(self, fig: Optional[go.Figure] = None, 
                   color: str = 'red', size: int = 10) -> go.Figure:
        """
        Plot Bloch sphere using plotly.
        
        Args:
            fig: Plotly figure
            color: Point color
            size: Point size
            
        Returns:
            Plotly figure
        """
        if fig is None:
            fig = go.Figure()
        
        # Draw sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            opacity=0.1, colorscale='Blues', showscale=False
        ))
        
        # Draw axes
        fig.add_trace(go.Scatter3d(
            x=[-1, 1], y=[0, 0], z=[0, 0],
            mode='lines', line=dict(color='black', width=2),
            showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[-1, 1], z=[0, 0],
            mode='lines', line=dict(color='black', width=2),
            showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[-1, 1],
            mode='lines', line=dict(color='black', width=2),
            showlegend=False
        ))
        
        # Plot state
        fig.add_trace(go.Scatter3d(
            x=[self.x], y=[self.y], z=[self.z],
            mode='markers', marker=dict(color=color, size=size),
            name='Quantum State'
        ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            title='Bloch Sphere'
        )
        
        return fig


class PhaseGate:
    """
    Phase gate operations for quantum states.
    
    Implements various phase gates including:
    - S gate: π/2 phase shift
    - T gate: π/4 phase shift  
    - P gate: arbitrary phase shift
    - Z gate: π phase shift
    """
    
    @staticmethod
    def s_gate(alpha: complex, beta: complex) -> Tuple[complex, complex]:
        """
        Apply S gate (π/2 phase shift).
        
        S = [[1, 0], [0, i]]
        
        Args:
            alpha: Amplitude of |0⟩ state
            beta: Amplitude of |1⟩ state
            
        Returns:
            New amplitudes after S gate
        """
        return alpha, 1j * beta
    
    @staticmethod
    def t_gate(alpha: complex, beta: complex) -> Tuple[complex, complex]:
        """
        Apply T gate (π/4 phase shift).
        
        T = [[1, 0], [0, e^(iπ/4)]]
        
        Args:
            alpha: Amplitude of |0⟩ state
            beta: Amplitude of |1⟩ state
            
        Returns:
            New amplitudes after T gate
        """
        phase = np.exp(1j * np.pi / 4)
        return alpha, phase * beta
    
    @staticmethod
    def p_gate(alpha: complex, beta: complex, phase: float) -> Tuple[complex, complex]:
        """
        Apply P gate (arbitrary phase shift).
        
        P(λ) = [[1, 0], [0, e^(iλ)]]
        
        Args:
            alpha: Amplitude of |0⟩ state
            beta: Amplitude of |1⟩ state
            phase: Phase shift in radians
            
        Returns:
            New amplitudes after P gate
        """
        phase_factor = np.exp(1j * phase)
        return alpha, phase_factor * beta
    
    @staticmethod
    def z_gate(alpha: complex, beta: complex) -> Tuple[complex, complex]:
        """
        Apply Z gate (π phase shift).
        
        Z = [[1, 0], [0, -1]]
        
        Args:
            alpha: Amplitude of |0⟩ state
            beta: Amplitude of |1⟩ state
            
        Returns:
            New amplitudes after Z gate
        """
        return alpha, -beta
    
    @staticmethod
    def rz_gate(alpha: complex, beta: complex, angle: float) -> Tuple[complex, complex]:
        """
        Apply RZ gate (rotation around Z-axis).
        
        RZ(θ) = [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
        
        Args:
            alpha: Amplitude of |0⟩ state
            beta: Amplitude of |1⟩ state
            angle: Rotation angle in radians
            
        Returns:
            New amplitudes after RZ gate
        """
        phase_0 = np.exp(-1j * angle / 2)
        phase_1 = np.exp(1j * angle / 2)
        return phase_0 * alpha, phase_1 * beta
