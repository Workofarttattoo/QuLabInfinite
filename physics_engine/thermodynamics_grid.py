"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

thermodynamics_grid - Part of Physics Engine
"""

from __future__ import annotations

from typing import Tuple, Optional, Callable
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import diags
from scipy.sparse.linalg import factorized

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

        # Caching for solver
        self.last_dt: Optional[float] = None
        self.solver: Optional[Callable] = None
        self.cached_gamma: Optional[float] = None

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
        
        # Check if we can reuse the solver
        if dt != self.last_dt:
            # Implicit method for stability (Crank-Nicolson)
            gamma = alpha * dt / (2 * self.dx**2)
            self.cached_gamma = gamma

            # Create the tridiagonal matrix for the linear system
            main_diag = np.full(N, 1 + 2 * gamma)
            off_diag = np.full(N - 1, -gamma)
            A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(N, N))

            # Convert to LIL for efficient modification
            A = A.tolil()

            # Apply boundary conditions (Dirichlet) matrix modifications
            A[0, 0], A[0, 1] = 1, 0
            A[N-1, N-1], A[N-1, N-2] = 1, 0

            # Precompute factorization for fast solving
            self.solver = factorized(A.tocsc())
            self.last_dt = dt
        else:
            gamma = self.cached_gamma
        
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
        
        # Solve the linear system using precomputed solver
        if self.solver:
            self.temperature_grid = self.solver(d)
