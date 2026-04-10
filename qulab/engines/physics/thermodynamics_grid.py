"""
Grid-based Thermodynamics Engine using the Finite Difference Method.
"""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from .thermodynamics import MaterialProperties, MATERIALS

class FiniteDifferenceThermodynamicsEngine:
    """
    A grid-based thermodynamics engine for simulating heat transfer in continuous media.
    
    Uses the finite difference method to solve the heat equation.
    """
    def __init__(self, grid_shape: Tuple[int, ...], dx: float, material: MaterialProperties):
        self.grid_shape = grid_shape
        self.dx = dx
        self.material = material
        
        self.temperature_grid = np.full(self.grid_shape, 300.0) # Initial temp: 300K
        self.thermal_diffusivity = (
            material.thermal_conductivity / (material.density * material.specific_heat)
        )

    def set_boundary_conditions(self, boundaries: dict):
        """
        Set boundary conditions for the simulation.
        
        Args:
            boundaries: A dictionary specifying the temperature at each boundary
                        (e.g., {'left': 400, 'right': 300}).
        """
        # This is a placeholder for a more robust boundary condition system
        pass

    def step(self, dt: float):
        """
        Advance the simulation by one timestep using an implicit method.
        """
        # This is a simplified 1D implementation for now
        if len(self.grid_shape) != 1:
            raise NotImplementedError("Only 1D grid is supported for now.")
            
        N = self.grid_shape[0]
        alpha = self.thermal_diffusivity
        
        # Implicit method for stability (Crank-Nicolson)
        gamma = alpha * dt / (2 * self.dx**2)
        
        # Create the tridiagonal matrix for the linear system
        main_diag = np.full(N, 1 + 2 * gamma)
        off_diag = np.full(N - 1, -gamma)
        A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(N, N))
        
        # Create the right-hand side vector
        d = np.zeros(N)
        T = self.temperature_grid
        d[1:-1] = gamma * T[:-2] + (1 - 2 * gamma) * T[1:-1] + gamma * T[2:]
        
        # Apply boundary conditions (Dirichlet)
        # These would be set by set_boundary_conditions in a real implementation
        T_left, T_right = 300.0, 400.0
        A[0, 0], A[0, 1] = 1, 0
        A[N-1, N-1], A[N-1, N-2] = 1, 0
        d[0] = T_left
        d[N-1] = T_right
        
        # Solve the linear system
        self.temperature_grid = spsolve(A.tocsc(), d)
