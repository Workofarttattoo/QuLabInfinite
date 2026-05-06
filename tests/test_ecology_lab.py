import unittest
import numpy as np
from ecology_lab import EcologyLab

class TestEcologyLab(unittest.TestCase):
    def setUp(self):
        self.lab = EcologyLab()

    def test_metapopulation_dynamics_vectorized(self):
        # Run a quick test to ensure the vectorized implementation returns correct shaped results
        np.random.seed(42)
        patches = 10
        time_years = 1
        result = self.lab.metapopulation_dynamics(
            patches=patches,
            colonization_rate=0.5,
            extinction_rate=0.2,
            initial_occupied=2,
            time_years=time_years
        )

        # Checking shape of output
        time_points = int(time_years * 12)
        self.assertEqual(result['occupancy_matrix'].shape, (time_points, patches))
        self.assertIn('equilibrium_occupancy', result)

if __name__ == '__main__':
    unittest.main()
