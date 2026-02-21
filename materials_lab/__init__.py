from __future__ import annotations

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

import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.append(str(_PKG_DIR))

from .materials_lab import MaterialsLab
from .materials_database import MaterialsDatabase, MaterialProperties
