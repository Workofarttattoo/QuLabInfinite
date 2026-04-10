import sys
import pytest
from unittest.mock import MagicMock

# Create a dummy numpy module so we bypass the numpy missing error in other parts of quantum_lab
sys.modules['numpy'] = MagicMock()

import types
if 'quantum_lab.quantum_lab' not in sys.modules:
    import core.base_lab

import importlib.util
spec = importlib.util.spec_from_file_location("quantum_lab.quantum_lab", "quantum_lab/quantum_lab.py")
quantum_lab_module = importlib.util.module_from_spec(spec)
sys.modules["quantum_lab.quantum_lab"] = quantum_lab_module

class MockSimulationBackend:
    STATEVECTOR_EXACT = 'STATEVECTOR_EXACT'
    TENSOR_NETWORK = 'TENSOR_NETWORK'

quantum_lab_module.SimulationBackend = MockSimulationBackend
quantum_lab_module.SimulationConfig = type('SimulationConfig', (), {})
quantum_lab_module.create_bell_pair = lambda: None
quantum_lab_module.create_ghz_state = lambda: None

spec.loader.exec_module(quantum_lab_module)

class TestQuantumLabSimulatorInit:
    """Test suite for QuantumLabSimulator initialization."""

    def test_init_attributes(self):
        """Test that initialization properly sets up empty attributes."""
        lab = quantum_lab_module.QuantumLabSimulator()

        assert lab.qubits == {}, "Expected qubits to be initialized as an empty dictionary"
        assert lab.circuits == {}, "Expected circuits to be initialized as an empty dictionary"

    def test_base_class_initialization(self):
        """Test that BaseLab initialization is properly handled."""
        lab = quantum_lab_module.QuantumLabSimulator()

        assert hasattr(lab, 'config'), "Expected lab to inherit config attribute from BaseLab"
        assert lab.config == {"lab_name": "Quantum Laboratory"}, "Expected config to hold lab_name"
