"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

thermodynamics_grid - Part of Physics Engine
"""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_banded

from .thermodynamics import MaterialProperties, MATERIALS

class FiniteDifferenceThermodynamicsEngine:
    """
    A grid-based thermodynamics engine for simulating heat transfer in continuous media.
    
    Uses the finite difference method to solve the heat equation.
    Optimized to use cached banded matrices for tridiagonal systems.
    """
    def __init__(self, grid_shape: Tuple[int, ...], dx: float, material: MaterialProperties):
        self.grid_shape = grid_shape
        self.dx = dx
        self.material = material
        
        self.temperature_grid = np.full(self.grid_shape, 300.0) # Initial temp: 300K
        self.thermal_diffusivity = (
            material.thermal_conductivity / (material.density * material.specific_heat)
        )

        # Caching for performance optimization
        self.cached_matrix = None
        self.cached_dt = None

    def set_boundary_conditions(self, boundaries: dict):
        """
        Set boundary conditions for the simulation.
        
        Args:
            boundaries: A dictionary specifying the temperature at each boundary
                        (e.g., {'left': 400, 'right': 300}).
        """
        # This is a placeholder for a more robust boundary condition system
        pass

    def _build_banded_matrix(self, dt: float) -> NDArray[np.float64]:
        """
        Construct the banded matrix for the linear system.
        """
        N = self.grid_shape[0]
        alpha = self.thermal_diffusivity
        gamma = alpha * dt / (2 * self.dx**2)

        # Banded matrix format (3, N) for tridiagonal
        # Row 0: Upper diagonal (prefixed with 0, effectively shifted right)
        # Row 1: Main diagonal
        # Row 2: Lower diagonal (suffixed with 0, effectively shifted left)

        ab = np.zeros((3, N))

        # Fill diagonals
        ab[0, 1:] = -gamma         # Upper diagonal: A[i, i+1]
        ab[1, :] = 1 + 2 * gamma   # Main diagonal: A[i, i]
        ab[2, :-1] = -gamma        # Lower diagonal: A[i, i-1]

        # Apply Dirichlet boundary conditions (Identity rows)
        # Row 0 (index 0): A[0,0]=1, A[0,1]=0
        ab[1, 0] = 1.0
        ab[0, 1] = 0.0

        # Row N-1 (index N-1): A[N-1,N-1]=1, A[N-1,N-2]=0
        ab[1, N-1] = 1.0
        ab[2, N-2] = 0.0

        return ab

    def step(self, dt: float):
        """
        Advance the simulation by one timestep using an implicit method.
        """
        # This is a simplified 1D implementation for now
        if len(self.grid_shape) != 1:
            raise NotImplementedError("Only 1D grid is supported for now.")
            
        N = self.grid_shape[0]
        
        # Rebuild matrix if dt changes
        if dt != self.cached_dt or self.cached_matrix is None:
            self.cached_matrix = self._build_banded_matrix(dt)
            self.cached_dt = dt

        alpha = self.thermal_diffusivity
        gamma = alpha * dt / (2 * self.dx**2)
        
        # Create the right-hand side vector d
        d = np.zeros(N)
        T = self.temperature_grid

        # Standard Crank-Nicolson RHS: (I - A_explicit) * T_old
        # d[i] = gamma*T[i-1] + (1-2gamma)*T[i] + gamma*T[i+1]
        d[1:-1] = gamma * T[:-2] + (1 - 2 * gamma) * T[1:-1] + gamma * T[2:]
        
        # Apply boundary conditions (Dirichlet)
        # These would be set by set_boundary_conditions in a real implementation
        T_left, T_right = 300.0, 400.0
        d[0] = T_left
        d[N-1] = T_right
        
        # Solve the linear system using the specialized banded solver
        # (1, 1) specifies the number of lower and upper diagonals
        self.temperature_grid = solve_banded((1, 1), self.cached_matrix, d)
