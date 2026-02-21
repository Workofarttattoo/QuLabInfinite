# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Structural Biology Laboratory Module
Molecular dynamics, protein-ligand docking, crystallography analysis, structural prediction
"""

__all__ = []

try:
    from .structural_biology_engine import StructuralBiologyEngine
    __all__.append("StructuralBiologyEngine")
except ImportError:
    StructuralBiologyEngine = None

try:
    from .molecular_dynamics import MolecularDynamics
    __all__.append("MolecularDynamics")
except ImportError:
    MolecularDynamics = None

try:
    from .docking_engine import DockingEngine
    __all__.append("DockingEngine")
except ImportError:
    DockingEngine = None

try:
    from .structure_predictor import StructurePredictor
    __all__.append("StructurePredictor")
except ImportError:
    StructurePredictor = None

__version__ = '1.0.0'
