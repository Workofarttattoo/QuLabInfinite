"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

__init__ - Part of Materials Lab
"""

"""
Materials laboratory package exports.

The original project stored most functionality in the ``materials_lab.py`` file
inside this directory.  Adding ``__init__.py`` promotes the directory to a
package so that ``import materials_lab`` works no matter where the caller is
located in the filesystem.  Existing code that imported ``MaterialsLab`` (or
related helpers) continues to work via the re-exports below.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:  # ensure legacy absolute imports continue to work
    sys.path.append(str(_PKG_DIR))

from .materials_lab import MaterialsLab
from .materials_database import MaterialsDatabase, MaterialProperties
from .material_testing import (
    TensileTest,
    CompressionTest,
    FatigueTest,
    ImpactTest,
    HardnessTest,
    ThermalTest,
    CorrosionTest,
    EnvironmentalTest,
)
from .material_designer import (
    AlloyOptimizer,
    CompositeDesigner,
    NanostructureEngineer,
    SurfaceTreatment,
    AdditiveManufacturing,
)
from .material_property_predictor import MaterialPropertyPredictor
from .material_profiles import MaterialProfileGenerator
from .phase_change import IceNucleationModel, IceCrystalGrowthModel, run_ice_analysis
from .calibration import CalibrationManager, CalibrationRecord
from .uncertainty import estimate_property_uncertainty
from .safety import SafetyData, SafetyManager

__all__ = [
    "MaterialsLab",
    "MaterialsDatabase",
    "MaterialProperties",
    "TensileTest",
    "CompressionTest",
    "FatigueTest",
    "ImpactTest",
    "HardnessTest",
    "ThermalTest",
    "CorrosionTest",
    "EnvironmentalTest",
    "AlloyOptimizer",
    "CompositeDesigner",
    "NanostructureEngineer",
    "SurfaceTreatment",
    "AdditiveManufacturing",
    "MaterialPropertyPredictor",
    "MaterialProfileGenerator",
    "IceNucleationModel",
    "IceCrystalGrowthModel",
    "run_ice_analysis",
    "CalibrationManager",
    "CalibrationRecord",
    "estimate_property_uncertainty",
    "SafetyData",
    "SafetyManager",
]
