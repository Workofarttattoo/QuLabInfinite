"""
Unit test for quantum_lab.py
"""

import sys
from pathlib import Path

# Ensure project root import resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import directly from the file to bypass __init__.py which seems to import non-existent things
import importlib.util
spec = importlib.util.spec_from_file_location("quantum_lab_module", str(PROJECT_ROOT / "quantum_lab" / "quantum_lab.py"))
quantum_lab_module = importlib.util.module_from_spec(spec)
sys.modules["quantum_lab_module"] = quantum_lab_module
spec.loader.exec_module(quantum_lab_module)

QuantumLabSimulator = quantum_lab_module.QuantumLabSimulator

def test_validate_returns_expected_status_and_capabilities():
    """Validate method should return the expected dictionary."""
    lab = QuantumLabSimulator()
    result = lab.validate()

    assert isinstance(result, dict)
    assert result.get('lab_name') == 'Quantum Laboratory'
    assert result.get('status') == 'operational'

    capabilities = result.get('capabilities', [])
    assert 'quantum_simulation' in capabilities
    assert 'qubit_operations' in capabilities
    assert 'circuit_design' in capabilities

def test_get_status_returns_expected_info():
    """get_status method should return the current state of the lab."""
    lab = QuantumLabSimulator()
    lab.create_qubit('q0')
    lab.create_qubit('q1')

    status = lab.get_status()
    assert isinstance(status, dict)
    assert status.get('status') == 'operational'
    assert status.get('qubits_count') == 2
    assert status.get('circuits_count') == 0
