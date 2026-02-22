
import unittest
import numpy as np
from physics_engine.thermodynamics_grid import FiniteDifferenceThermodynamicsEngine
from physics_engine.thermodynamics import MaterialProperties

class TestFiniteDifferenceThermodynamicsEngine(unittest.TestCase):
    def test_initialization(self):
        material = MaterialProperties(
            name="Test", density=1000, specific_heat=1000, thermal_conductivity=10
        )
        engine = FiniteDifferenceThermodynamicsEngine((10,), 0.1, material)
        self.assertEqual(engine.grid_shape, (10,))
        self.assertEqual(engine.temperature_grid.shape, (10,))

    def test_step_caching(self):
        material = MaterialProperties(
            name="Test", density=1000, specific_heat=1000, thermal_conductivity=10
        )
        engine = FiniteDifferenceThermodynamicsEngine((10,), 0.1, material)
        dt = 0.1

        # First step should initialize solver
        engine.step(dt)
        self.assertIsNotNone(engine.solver)
        self.assertEqual(engine.last_dt, dt)

        solver_id = id(engine.solver)

        # Second step with same dt should reuse solver
        engine.step(dt)
        self.assertEqual(id(engine.solver), solver_id)

        # Step with different dt should recreate solver
        engine.step(0.2)
        self.assertNotEqual(id(engine.solver), solver_id)
        self.assertEqual(engine.last_dt, 0.2)

if __name__ == "__main__":
    unittest.main()
