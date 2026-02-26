import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from physics_engine.thermodynamics_grid import FiniteDifferenceThermodynamicsEngine
from physics_engine.thermodynamics import MATERIALS

class TestFiniteDifferenceThermodynamicsEngine(unittest.TestCase):
    def test_heat_diffusion(self):
        """Test that heat diffuses correctly in 1D grid."""
        N = 100
        dx = 0.01
        dt = 0.1
        material = MATERIALS["copper"]

        engine = FiniteDifferenceThermodynamicsEngine(grid_shape=(N,), dx=dx, material=material)

        # Set initial condition: hot in the middle
        engine.temperature_grid[:] = 300.0
        mid = N // 2
        # Set a block of 400K in the middle
        engine.temperature_grid[mid-5:mid+5] = 400.0

        # Step
        steps = 100
        for _ in range(steps):
            engine.step(dt)

        # The right boundary is fixed at 400.0, so max temp will be 400.0.
        # We check if the middle block has cooled down due to diffusion.

        mid_temp = np.mean(engine.temperature_grid[mid-5:mid+5])
        self.assertLess(mid_temp, 390.0, "Middle block did not cool down significantly")

        # Check boundary conditions are respected
        self.assertAlmostEqual(engine.temperature_grid[0], 300.0, delta=1e-5)
        self.assertAlmostEqual(engine.temperature_grid[-1], 400.0, delta=1e-5)

if __name__ == "__main__":
    unittest.main()
