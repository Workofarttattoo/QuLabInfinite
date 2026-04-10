import sys
import unittest
from unittest.mock import MagicMock, patch

class TestCancerDrugQuantumDiscovery(unittest.TestCase):

    def setUp(self):
        # We use a patch.dict context manager to temporarily inject mocks into sys.modules
        # This prevents polluting the test runner globally for other test files.
        self.mock_modules = {
            'numpy': MagicMock(),
            'biological_quantum.core.quantum_state': MagicMock(),
            'biological_quantum.algorithms.quantum_optimization': MagicMock(),
            'benchmarks.quantum_benchmark': MagicMock(),
            'biological_quantum.core': MagicMock(),
            'biological_quantum.algorithms': MagicMock(),
            'biological_quantum.simulation': MagicMock(),
            'biological_quantum.hardware': MagicMock(),
            'biological_quantum.experimental': MagicMock(),
            'biological_quantum.benchmarks': MagicMock(),
            'biological_quantum_lab': MagicMock()
        }

        self.patcher = patch.dict('sys.modules', self.mock_modules)
        self.patcher.start()

        # Now we can import the module to test safely inside the test methods or setup
        import cancer_drug_quantum_discovery_ENHANCED
        self.module = cancer_drug_quantum_discovery_ENHANCED

    def tearDown(self):
        self.patcher.stop()

    def test_calculate_metrics(self):
        """Test the internal metric calculation for DrugCandidate"""
        candidate = self.module.DrugCandidate("TestCand", "TestTarget", 450.0)

        # Test calculation with a specific binding energy
        binding_energy = -5.0
        convergence_history = [-1.0, -3.0, -5.0]

        candidate.calculate_metrics(binding_energy, convergence_history)

        self.assertEqual(candidate.binding_energy, binding_energy)
        self.assertEqual(candidate.convergence_history, convergence_history)

        # ic50 = 10 ** (-(binding_energy + 5.2))
        expected_ic50 = 10 ** (-(-5.0 + 5.2))
        self.assertAlmostEqual(candidate.ic50, expected_ic50)

        # selectivity = min(98.0, 88.0 - (binding_energy * 8.5))
        expected_selectivity = min(98.0, 88.0 - (-5.0 * 8.5))
        self.assertAlmostEqual(candidate.selectivity, expected_selectivity)

        # druglikeness = min(99.0, 87.0 + (binding_energy * 4.2))
        expected_druglikeness = min(99.0, 87.0 + (-5.0 * 4.2))
        self.assertAlmostEqual(candidate.druglikeness, expected_druglikeness)

        # side_effect_score = max(0.0, 35.0 + (binding_energy * 15.0))
        expected_side_effect_score = max(0.0, 35.0 + (-5.0 * 15.0))
        self.assertAlmostEqual(candidate.side_effect_score, expected_side_effect_score)

        # manufacturing_cost = 150.0 * (450.0 / 450.0)
        expected_mfg_cost = 150.0
        self.assertAlmostEqual(candidate.manufacturing_cost, expected_mfg_cost)

        # predicted_efficacy = 100.0 * (1.0 - (binding_energy * 0.18))
        expected_efficacy = 100.0 * (1.0 - (-5.0 * 0.18))
        self.assertAlmostEqual(candidate.predicted_efficacy, expected_efficacy)

    @patch('cancer_drug_quantum_discovery_ENHANCED.time.time')
    def test_optimize_drug_candidate(self, mock_time):
        """Test the full optimize_drug_candidate function with mocked lab"""
        # Set up mocks
        mock_lab = MagicMock()
        mock_lab.run_vqe.return_value = (-8.5, [0.1, 0.2, 0.3])

        # Mock time to test optimization_time calculation
        mock_time.side_effect = [100.0, 105.5] # start_time, end_time => diff is 5.5s

        with patch('biological_quantum.core.quantum_state.QuantumState') as mock_qs_class, \
             patch('biological_quantum.algorithms.quantum_optimization.VariationalQuantumEigensolver') as mock_vqe_class:

            # Mock VQE and QuantumState behavior
            mock_vqe_instance = MagicMock()
            mock_vqe_class.return_value = mock_vqe_instance

            mock_qs_instance = MagicMock()
            mock_qs_instance.measure.return_value = (42, 1.0) # 42 in binary is 00101010
            mock_qs_class.return_value = mock_qs_instance

            # Setup candidate
            candidate = self.module.DrugCandidate("TestDrug", "TestTarget", 300.0)
            target_id = 2

            # Run function
            result = self.module.optimize_drug_candidate(mock_lab, candidate, target_id)

            # Verify result is the candidate
            self.assertIs(result, candidate)

            # Verify run_vqe was called on the lab
            mock_lab.run_vqe.assert_called_once()
            args, kwargs = mock_lab.run_vqe.call_args
            self.assertEqual(kwargs['n_qubits'], 8)
            self.assertEqual(kwargs['depth'], 3)
            self.assertEqual(kwargs['max_iterations'], 30)

            # Verify candidate metrics got updated based on mock_lab result
            self.assertEqual(candidate.binding_energy, -8.5)
            self.assertEqual(candidate.optimization_time, 5.5)
            self.assertEqual(candidate.configuration, format(42, '08b'))

            # Verify VQE was instantiated properly and measurement was taken
            mock_vqe_class.assert_called_once_with(n_qubits=8, depth=3)
            mock_qs_class.assert_called_once_with(8)
            mock_vqe_instance.hardware_efficient_ansatz.assert_called_once_with(mock_qs_instance, [0.1, 0.2, 0.3])
            mock_qs_instance.measure.assert_called_once()

if __name__ == '__main__':
    unittest.main()
