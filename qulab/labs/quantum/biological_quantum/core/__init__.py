"""
Biological Quantum Computing - Core Module

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

from .quantum_state import QuantumState, create_bell_state, create_ghz_state
from .quantum_gates import (
    hadamard, pauli_x, pauli_y, pauli_z,
    phase_gate, rotation_x, rotation_y, rotation_z,
    s_gate, t_gate, cnot, cz, swap,
    apply_hadamard, apply_x, apply_y, apply_z,
    apply_phase, apply_rx, apply_ry, apply_rz,
    apply_cnot, apply_cz, apply_swap,
    verify_unitary
)

__all__ = [
    'QuantumState', 'create_bell_state', 'create_ghz_state',
    'hadamard', 'pauli_x', 'pauli_y', 'pauli_z',
    'phase_gate', 'rotation_x', 'rotation_y', 'rotation_z',
    's_gate', 't_gate', 'cnot', 'cz', 'swap',
    'apply_hadamard', 'apply_x', 'apply_y', 'apply_z',
    'apply_phase', 'apply_rx', 'apply_ry', 'apply_rz',
    'apply_cnot', 'apply_cz', 'apply_swap',
    'verify_unitary'
]
