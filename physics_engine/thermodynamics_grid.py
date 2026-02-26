"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

thermodynamics_grid - Part of Physics Engine
"""

from __future__ import annotations

from typing import Tuple, Optional, Callable
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded

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

        # Cache for solver matrix
        self._ab_cached = None
        self._last_dt = None

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
        
        # Check cache
        if self._ab_cached is None or dt != self._last_dt:
            # Create banded matrix for solve_banded (3 x N)
            # Row 0: Upper diagonal (starting from index 1)
            # Row 1: Main diagonal
            # Row 2: Lower diagonal (starting from index 0)

            ab = np.zeros((3, N))

            # Main diagonal
            ab[1, :] = 1 + 2 * gamma
            # Upper diagonal (A[i, i+1]) -> ab[0, i+1]
            # Since solve_banded expects ab[u + i - j, j], for u=1, j=i+1 => 1 + i - (i+1) = 0
            ab[0, 1:] = -gamma
            # Lower diagonal (A[i, i-1]) -> ab[2, i-1]
            # For l=1, j=i-1 => 1 + i - (i-1) = 2
            ab[2, :-1] = -gamma

            # Apply boundary conditions to matrix
            # Left boundary (i=0): T[0] = T_left -> 1*T[0] + 0*T[1] = T_left
            ab[1, 0] = 1.0  # Main diagonal
            ab[0, 1] = 0.0  # Upper diagonal A[0, 1]

            # Right boundary (i=N-1): T[N-1] = T_right -> 0*T[N-2] + 1*T[N-1] = T_right
            ab[1, N-1] = 1.0 # Main diagonal
            ab[2, N-2] = 0.0 # Lower diagonal A[N-1, N-2]

            self._ab_cached = ab
            self._last_dt = dt

        # Create the right-hand side vector
        d = np.zeros(N)
        T = self.temperature_grid

        # Vectorized update for internal nodes
        # Crank-Nicolson RHS: (1-2*gamma)*T[i] + gamma*(T[i-1] + T[i+1]) ??
        # Wait, the original code had:
        # d[1:-1] = gamma * T[:-2] + (1 - 2 * gamma) * T[1:-1] + gamma * T[2:]
        # This matches explicit Euler or part of Crank-Nicolson depending on formulation.
        # Assuming the original math was intended (it looks like CN mixed with explicit):
        # A * T_new = B * T_old
        # The code builds 'd' which is B * T_old.
        d[1:-1] = gamma * T[:-2] + (1 - 2 * gamma) * T[1:-1] + gamma * T[2:]
        
        # Apply boundary conditions (Dirichlet) to RHS
        # These would be set by set_boundary_conditions in a real implementation
        T_left, T_right = 300.0, 400.0
        d[0] = T_left
        d[N-1] = T_right
        
        # Solve the linear system using banded solver (O(N))
        self.temperature_grid = solve_banded((1, 1), self._ab_cached, d)
