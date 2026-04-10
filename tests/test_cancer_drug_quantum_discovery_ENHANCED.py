import sys
import unittest
from unittest.mock import MagicMock
import io

# Mock dependencies before import
sys.modules['biological_quantum_lab'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Now import the module to test
import cancer_drug_quantum_discovery_ENHANCED as drug_module
from cancer_drug_quantum_discovery_ENHANCED import DrugCandidate, print_market_analysis

class TestMarketAnalysis(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            DrugCandidate("QuantumCure-A", "Target-A", 400.0),
            DrugCandidate("QuantumCure-B", "Target-B", 450.0)
        ]

        # Override values for testing
        self.candidates[0].ic50 = 4.5
        self.candidates[0].selectivity = 92.0
        self.candidates[0].predicted_efficacy = 120.0

        self.candidates[1].ic50 = 8.0
        self.candidates[1].selectivity = 85.0
        self.candidates[1].predicted_efficacy = 110.0

    def test_print_market_analysis_content(self):
        """Test that print_market_analysis prints the correct sections and candidate data."""
        captured_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output

        try:
            print_market_analysis(self.candidates)
        finally:
            sys.stdout = original_stdout

        output = captured_output.getvalue()

        # Check section headers
        self.assertIn("MARKET IMPACT ANALYSIS", output)
        self.assertIn("TOTAL ADDRESSABLE MARKET (TAM):", output)
        self.assertIn("PORTFOLIO VALUE ESTIMATE:", output)
        self.assertIn("COST SAVINGS VS TRADITIONAL DRUG DISCOVERY:", output)
        self.assertIn("TIME-TO-MARKET ADVANTAGE:", output)

        # Check candidate names
        self.assertIn("QuantumCure-A:", output)
        self.assertIn("QuantumCure-B:", output)

    def test_print_market_analysis_empty(self):
        """Test print_market_analysis handles an empty list of candidates."""
        captured_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output

        try:
            print_market_analysis([])
        finally:
            sys.stdout = original_stdout

        output = captured_output.getvalue()

        # Ensure no errors occur and sections are still present
        self.assertIn("MARKET IMPACT ANALYSIS", output)
        self.assertIn("TOTAL ADDRESSABLE MARKET (TAM):", output)
        self.assertIn("PORTFOLIO VALUE ESTIMATE:", output)

if __name__ == '__main__':
    unittest.main()
