import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# We will use fixtures to safely patch `sys.path` and `sys.modules` only during the test scope.

@pytest.fixture
def mock_dependencies():
    """Mock external and missing dependencies safely for the test scope."""
    import types

    # We must also mock quantum_chemistry and quantum_validation because they
    # depend on numpy which we mock, and they are imported by demo.py
    fake_chem = types.ModuleType('quantum_chemistry')
    fake_chem.Molecule = MagicMock()

    fake_val = types.ModuleType('quantum_validation')
    fake_val.QuantumValidation = MagicMock()

    fake_qlab = types.ModuleType('quantum_lab')
    fake_qlab.SimulationBackend = MagicMock()
    fake_qlab.create_bell_pair = MagicMock()
    fake_qlab.create_ghz_state = MagicMock()
    class FakeSimulator: pass
    fake_qlab.QuantumLabSimulator = FakeSimulator

    mocks = {
        'numpy': MagicMock(),
        'scipy': MagicMock(),
        'scipy.sparse': MagicMock(),
        'scipy.linalg': MagicMock(),
        'scipy.optimize': MagicMock(),
        'scipy.sparse.linalg': MagicMock(),
        'quantum_chemistry': fake_chem,
        'quantum_validation': fake_val,
        'quantum_lab': fake_qlab
    }

    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    qlab_dir = os.path.join(workspace_dir, 'quantum_lab')

    with patch.dict('sys.modules', mocks):
        # Safely patch sys.path only for the scope of the test
        original_path = list(sys.path)
        sys.path.insert(0, qlab_dir)
        sys.path.insert(0, workspace_dir)

        try:
            yield
        finally:
            sys.path = original_path

def test_demo_basic_circuits(mock_dependencies):
    """Test demo_basic_circuits covers basic circuits with proper simulator usage"""

    # Import the function locally inside the test so the sys.modules patch applies!
    # Because sys.path includes quantum_lab, we can import demo directly
    from demo import demo_basic_circuits

    @patch('demo.QuantumLabSimulator')
    @patch('builtins.input')
    @patch('builtins.print')
    def run_test(mock_print, mock_input, mock_simulator_class):
        # Setup mock for the simulator instance
        mock_lab_instance = MagicMock()
        mock_simulator_class.return_value = mock_lab_instance

        # Setup chained calls (h, cnot, reset all return the simulator instance itself)
        mock_lab_instance.h.return_value = mock_lab_instance
        mock_lab_instance.cnot.return_value = mock_lab_instance
        mock_lab_instance.reset.return_value = mock_lab_instance

        # Setup measure_all to return a mocked measurement result
        mock_lab_instance.measure_all.return_value = [0, 1, 0, 1, 0]

        # Call the target function
        demo_basic_circuits()

        # Assert QuantumLabSimulator was instantiated with specific arguments
        mock_simulator_class.assert_called_once_with(num_qubits=5, verbose=False)

        # Verify the specific methods were called the expected number of times
        # Step 1: h(0).h(1).h(2) -> 3 calls
        # Step 2: h(0) -> 1 call
        # Total: 4 calls
        assert mock_lab_instance.h.call_count == 4

        # Step 2: cnot(0, 1).cnot(1, 2) -> 2 calls
        assert mock_lab_instance.cnot.call_count == 2

        # Step 2 starts with reset()
        assert mock_lab_instance.reset.call_count == 1

        # Verify print_state was called twice with top_n=8
        assert mock_lab_instance.print_state.call_count == 2
        mock_lab_instance.print_state.assert_called_with(top_n=8)

        # Verify measure_all was called
        mock_lab_instance.measure_all.assert_called_once()

        # Verify input was called at the end to wait for user interaction
        mock_input.assert_called_once()

    run_test()
