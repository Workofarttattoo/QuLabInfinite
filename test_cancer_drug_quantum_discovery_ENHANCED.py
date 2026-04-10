import unittest
import io
import sys
from unittest.mock import patch, MagicMock

# Mock numpy before importing the module
mock_np = MagicMock()
sys.modules["numpy"] = mock_np

# Mock biological_quantum_lab
mock_bql = MagicMock()
sys.modules["biological_quantum_lab"] = mock_bql

try:
    import cancer_drug_quantum_discovery_ENHANCED
    from cancer_drug_quantum_discovery_ENHANCED import DrugCandidate, print_fda_comparison, FDA_APPROVED_DRUGS
except ImportError as e:
    print(f"ImportError: {e}")

# In case we need to mock np.mean during tests since mock_np.mean returns a MagicMock
# which won't work correctly with math operations like avg_fda_ic50 - best_candidate.ic50
def side_effect_mean(arr):
    return sum(arr) / len(arr) if arr else 0.0
mock_np.mean.side_effect = side_effect_mean

class TestCancerDrugQuantumDiscoveryEnhanced(unittest.TestCase):
    def setUp(self):
        self.candidate = DrugCandidate(name="TestDrugX", target="TestTarget", molecular_weight=350.5)
        self.candidate.ic50 = 2.5
        self.candidate.selectivity = 95.0

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_print_fda_comparison_happy_path(self, mock_stdout):
        """Test print_fda_comparison with normal values"""
        print_fda_comparison(self.candidate)
        output = mock_stdout.getvalue()

        # Check for section header
        self.assertIn("COMPARISON WITH FDA-APPROVED CANCER DRUGS", output)

        # Check for our mock candidate data
        self.assertIn("TestDrugX", output)
        self.assertIn("2.50", output) # IC50 format
        self.assertIn("95.0", output) # Selectivity format
        self.assertIn("(Our Quantum Drug)", output)

        # Check for FDA drugs present in output
        for fda_drug in FDA_APPROVED_DRUGS.keys():
            self.assertIn(fda_drug, output)

        # Check for improvement metrics section
        self.assertIn("IMPROVEMENT METRICS:", output)
        self.assertIn("Potency Improvement:", output)
        self.assertIn("Selectivity Improvement:", output)
        self.assertIn("Cost Reduction:", output)

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_print_fda_comparison_zero_values(self, mock_stdout):
        """Test print_fda_comparison with zero IC50 and Selectivity values"""
        zero_candidate = DrugCandidate(name="ZeroDrug", target="None", molecular_weight=100.0)
        zero_candidate.ic50 = 0.0
        zero_candidate.selectivity = 0.0

        print_fda_comparison(zero_candidate)
        output = mock_stdout.getvalue()

        self.assertIn("ZeroDrug", output)
        self.assertIn("0.00", output) # ic50 format
        self.assertIn("0.0", output) # selectivity format

        # A 0 IC50 means 100% potency improvement
        self.assertIn("Potency Improvement: +100.0%", output)

        # A 0 selectivity means -100% improvement
        self.assertIn("Selectivity Improvement: -100.0%", output)

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_print_fda_comparison_negative_improvement(self, mock_stdout):
        """Test print_fda_comparison with worse values than FDA average"""
        worse_candidate = DrugCandidate(name="WorseDrug", target="None", molecular_weight=100.0)
        worse_candidate.ic50 = 100.0  # Much higher than FDA avg
        worse_candidate.selectivity = 10.0 # Much lower than FDA avg

        print_fda_comparison(worse_candidate)
        output = mock_stdout.getvalue()

        # Check that negative improvements are formatted correctly (e.g., -608.5% or similar)
        # Using regex or just checking the minus sign
        self.assertIn("Potency Improvement: -", output)
        self.assertIn("Selectivity Improvement: -", output)

if __name__ == '__main__':
    unittest.main()
