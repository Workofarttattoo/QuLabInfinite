import pytest
from quantum_lab.quantum_lab import QuantumLabSimulator

def test_create_qubit():
    """Test creating a quantum qubit."""
    lab = QuantumLabSimulator()
    result = lab.create_qubit('q1')

    assert 'q1' in lab.qubits
    assert lab.qubits['q1'] == {'state': [1, 0], 'coherence': 1.0}
    assert result == {'status': 'created', 'qubit_id': 'q1'}

def test_create_multiple_qubits():
    """Test creating multiple qubits."""
    lab = QuantumLabSimulator()
    lab.create_qubit('q1')
    lab.create_qubit('q2')

    assert len(lab.qubits) == 2
    assert 'q1' in lab.qubits
    assert 'q2' in lab.qubits

def test_overwrite_qubit():
    """Test overwriting an existing qubit."""
    lab = QuantumLabSimulator()
    lab.create_qubit('q1')

    # Modify the state to test overwrite
    lab.qubits['q1']['state'] = [0, 1]

    # Create again should reset
    result = lab.create_qubit('q1')

    assert lab.qubits['q1']['state'] == [1, 0]
    assert result == {'status': 'created', 'qubit_id': 'q1'}
