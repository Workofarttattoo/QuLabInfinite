import unittest
import numpy as np
from physics_engine.thermodynamics_grid import FiniteDifferenceThermodynamicsEngine
from physics_engine.thermodynamics import MATERIALS

class TestFiniteDifferenceThermodynamicsEngine(unittest.TestCase):
    def test_grid_initialization(self):
        N = 100
        dx = 0.01
        material = MATERIALS["copper"]
        engine = FiniteDifferenceThermodynamicsEngine((N,), dx, material)

        self.assertEqual(engine.grid_shape, (N,))
        self.assertEqual(engine.dx, dx)
        self.assertTrue(np.all(engine.temperature_grid == 300.0))

    def test_step_execution(self):
        N = 100
        dx = 0.01
        material = MATERIALS["copper"]
        engine = FiniteDifferenceThermodynamicsEngine((N,), dx, material)

        # Should not raise error
        engine.step(0.01)

        # Should have cached the matrix
        self.assertIsNotNone(engine._ab_cached)
        self.assertEqual(engine._last_dt, 0.01)

    def test_steady_state(self):
        N = 10
        dx = 0.1
        material = MATERIALS["copper"]
        engine = FiniteDifferenceThermodynamicsEngine((N,), dx, material)

        # Run to steady state
        # Copper diffusivity approx 1.1e-4 m^2/s
        # L = 1.0m. Time constant ~ L^2/alpha ~ 1/1e-4 ~ 10000s.
        # Run for 50000s
        for _ in range(500):
            engine.step(100.0)

        T = engine.temperature_grid

        # Check boundaries
        self.assertAlmostEqual(T[0], 300.0)
        self.assertAlmostEqual(T[-1], 400.0)

        # Check linearity
        expected = np.linspace(300, 400, N)
        mse = np.mean((T - expected)**2)
        self.assertLess(mse, 1e-5)

if __name__ == "__main__":
    unittest.main()
