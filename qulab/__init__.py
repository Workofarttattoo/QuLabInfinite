"""
QuLabInfinite — Infinite Scientific Simulation Platform.

100+ specialized laboratories spanning physics, chemistry, biology,
medicine, engineering, quantum computing, computer science, and more.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

__version__ = "1.0.0"
__author__ = "Joshua Hendricks Cole"
__license__ = "Proprietary"

from qulab.core.base_lab import BaseLab, register_lab, LabMetadata
from qulab.core.registry import LabRegistry
from qulab.core.simulator import UnifiedSimulator
from qulab.core.config import ConfigManager

__all__ = [
    "BaseLab",
    "register_lab",
    "LabMetadata",
    "LabRegistry",
    "UnifiedSimulator",
    "ConfigManager",
    "__version__",
]
