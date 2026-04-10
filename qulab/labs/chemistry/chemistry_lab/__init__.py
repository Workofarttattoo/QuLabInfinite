"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Chemistry Laboratory Package
"""

from .chemistry_lab import ChemistryLaboratory, ChemistryLabConfig
from .molecular_dynamics import MolecularDynamics, ForceField, Integrator, Ensemble
from .reaction_simulator import ReactionSimulator, ReactionPath, Catalyst
from .synthesis_planner import (
    SynthesisPlanner, SynthesisRoute, Compound, Transformation, TransformationType
)
from .spectroscopy_predictor import SpectroscopyPredictor, SpectroscopyType
from .solvation_model import SolvationCalculator, Solute, Solvent
from .quantum_chemistry_interface import QuantumChemistryInterface, QMMethod, BasisSet, DFTFunctional
from .integration import apply_material_updates, apply_environmental_adjustments
from .calibration import calibrate_spectroscopy, calibrate_synthesis
from .reference_data import list_known_targets, list_known_molecules
from .validation import KINETICS_BENCHMARKS, run_kinetics_validation

__all__ = [
    'ChemistryLaboratory',
    'ChemistryLabConfig',
    'MolecularDynamics',
    'ForceField',
    'Integrator',
    'Ensemble',
    'ReactionSimulator',
    'ReactionPath',
    'Catalyst',
    'SynthesisPlanner',
    'SynthesisRoute',
    'Compound',
    'Transformation',
    'TransformationType',
    'SpectroscopyPredictor',
    'SpectroscopyType',
    'SolvationCalculator',
    'Solute',
    'Solvent',
    'QuantumChemistryInterface',
    'QMMethod',
    'BasisSet',
    'DFTFunctional',
    'apply_material_updates',
    'apply_environmental_adjustments',
    'calibrate_spectroscopy',
    'calibrate_synthesis',
    'list_known_targets',
    'list_known_molecules',
    'KINETICS_BENCHMARKS',
    'run_kinetics_validation',
]

__version__ = '1.0.0'
