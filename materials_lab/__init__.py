from __future__ import annotations
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

__init__ - Part of Materials Lab
"""

import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.append(str(_PKG_DIR))

from .materials_lab import MaterialsLab
from .materials_database import MaterialsDatabase, MaterialProperties
