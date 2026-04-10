import unittest
from unittest.mock import patch
import io
import sys

# Mock imports since biological_quantum_lab and dependencies are missing in the environment
import sys
from unittest.mock import MagicMock
sys.modules['biological_quantum_lab'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['biological_quantum.core.quantum_state'] = MagicMock()
sys.modules['biological_quantum.algorithms.quantum_optimization'] = MagicMock()

from cancer_drug_quantum_discovery_ENHANCED import DrugCandidate, print_convergence_visualization

class TestPrintConvergenceVisualization(unittest.TestCase):
    def setUp(self):
        self.candidate = DrugCandidate(name="TestDrug", target="TestTarget", molecular_weight=100.0)

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_insufficient_data(self, mock_stdout):
        # Test case where history length < 2
        self.candidate.convergence_history = [1.0]
        print_convergence_visualization(self.candidate)
        output = mock_stdout.getvalue()

        self.assertIn("🔬 CONVERGENCE PLOT: TestDrug", output)
        self.assertIn("(Insufficient data)", output)

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_standard_visualization(self, mock_stdout):
        # Standard case with clear convergence
        self.candidate.convergence_history = [10.0, 8.0, 5.0, 2.0, 1.0, 0.5]
        print_convergence_visualization(self.candidate)
        output = mock_stdout.getvalue()

        self.assertIn("🔬 CONVERGENCE PLOT: TestDrug", output)
        self.assertIn("Initial Energy: 10.0000 a.u.", output)
        self.assertIn("Final Energy: 0.5000 a.u.", output)
        self.assertIn("95.0% improvement", output)
        self.assertIn("Iteration 6", output)

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_long_history_sampling(self, mock_stdout):
        # Case where history exceeds maximum plot width (width=50)
        self.candidate.convergence_history = [10.0 - (i * 0.1) for i in range(100)] # 100 points
        print_convergence_visualization(self.candidate)
        output = mock_stdout.getvalue()

        self.assertIn("Initial Energy: 10.0000 a.u.", output)
        self.assertIn("Iteration 100", output)
        # Check that it sampled to max width 50
        self.assertRegex(output, r"0\s+Iteration 100")
        # The width line has 50 dashes
        self.assertIn("-" * 50, output.split('\n')[3])

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_constant_energy(self, mock_stdout):
        # Case where max_energy == min_energy
        self.candidate.convergence_history = [5.0, 5.0, 5.0, 5.0]
        print_convergence_visualization(self.candidate)
        output = mock_stdout.getvalue()

        self.assertIn("Initial Energy: 5.0000 a.u.", output)
        self.assertIn("Final Energy: 5.0000 a.u.", output)
        self.assertIn("0.0% improvement", output)
        self.assertIn("Iteration 4", output)

if __name__ == '__main__':
    unittest.main()
