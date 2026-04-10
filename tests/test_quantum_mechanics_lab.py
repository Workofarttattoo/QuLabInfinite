import pytest
import numpy as np
from quantum_mechanics_lab import QuantumMechanicsLab

def test_initialization():
    lab = QuantumMechanicsLab()
    assert lab.name == "Quantum Mechanics Laboratory"
    assert lab.version == "2.0.0"

def test_tunneling_probability():
    lab = QuantumMechanicsLab()

    # Simple test case where particle energy > barrier
    T = lab.quantum_tunneling_probability(barrier_height=1.0, barrier_width=1e-9, particle_energy=2.0)
    assert T == 1.0

    # General sanity check for sub-barrier tunneling
    T2 = lab.quantum_tunneling_probability(barrier_height=2.0, barrier_width=1e-9, particle_energy=1.0)
    assert 0 <= T2 < 1.0
