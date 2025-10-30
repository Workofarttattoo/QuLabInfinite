"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.

QuLab Infinite - Quantum Laboratory
Complete quantum simulation suite with chemistry, materials, and sensors.
"""

from .quantum_lab import (
    QuantumLabSimulator,
    SimulationBackend,
    SimulationConfig,
    create_bell_pair,
    create_ghz_state,
)
from .quantum_chemistry import QuantumChemistry, Molecule
from .quantum_materials import QuantumMaterials
from .quantum_sensors import QuantumSensors

__all__ = [
    "QuantumLabSimulator",
    "SimulationBackend",
    "SimulationConfig",
    "create_bell_pair",
    "create_ghz_state",
    "QuantumChemistry",
    "Molecule",
    "QuantumMaterials",
    "QuantumSensors",
]

__version__ = '1.0.0'
