"""Pytest configuration ensuring local packages are importable."""

import sys
from pathlib import Path

# Ensure the repository root is on sys.path when tests run without installation.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

