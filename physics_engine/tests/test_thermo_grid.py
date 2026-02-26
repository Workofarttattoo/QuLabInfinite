"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Unit tests for the finite difference thermodynamics grid engine.
"""

import sys
import unittest
from unittest.mock import MagicMock

# Mock mendeleev before importing physics_engine
sys.modules["mendeleev"] = MagicMock()

import numpy as np
from physics_engine.thermodynamics_grid import FiniteDifferenceThermodynamicsEngine
from physics_engine.thermodynamics import MaterialProperties

class TestThermoGrid(unittest.TestCase):
    """Test cases for FiniteDifferenceThermodynamicsEngine."""

    def setUp(self):
        # Mock material
        self.material = MaterialProperties(
            name="Test",
            density=1.0,
            specific_heat=1.0,
            thermal_conductivity=1.0
        )
        # Small grid for testing
        self.N = 10
        self.grid_shape = (self.N,)
        self.dx = 1.0
        self.dt = 1.0

        self.engine = FiniteDifferenceThermodynamicsEngine(self.grid_shape, self.dx, self.material)

    def test_initialization(self):
        self.assertEqual(self.engine.grid_shape, (self.N,))
        self.assertEqual(len(self.engine.temperature_grid), self.N)
        self.assertTrue(np.all(self.engine.temperature_grid == 300.0))

    def test_boundary_conditions(self):
        # Run one step
        self.engine.step(self.dt)

        # Check boundary conditions
        # Based on the code, left is 300.0, right is 400.0
        self.assertEqual(self.engine.temperature_grid[0], 300.0)
        self.assertEqual(self.engine.temperature_grid[-1], 400.0)

    def test_diffusion(self):
        # Run multiple steps
        for _ in range(10):
            self.engine.step(self.dt)

        # Check if heat diffused from right (400) to left (300)
        # The right side should be hotter than the middle, which should be hotter than the left
        # T[N-1] > T[N-2] > ... > T[0]
        # Note: Depending on diffusivity and time, the gradient might not be fully established,
        # but T[-2] should certainly be > 300 if T[-1] is 400.

        self.assertGreater(self.engine.temperature_grid[-2], 300.0)
        self.assertLess(self.engine.temperature_grid[1], 400.0)

        # Monotonicity check (simple diffusion case)
        # T[i] <= T[i+1]
        diffs = np.diff(self.engine.temperature_grid)
        self.assertTrue(np.all(diffs >= -1e-10), f"Temperature grid is not monotonic: {self.engine.temperature_grid}")

if __name__ == '__main__':
    unittest.main()
