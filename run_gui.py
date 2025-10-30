#!/usr/bin/env python3
"""
Main entry point for launching the QuLabInfinite GUI.
"""

import sys
from pathlib import Path

# Add the project root to the Python path to allow for absolute imports
sys.path.append(str(Path(__file__).resolve().parent))

from gui.main_window import run_gui

if __name__ == "__main__":
    run_gui()
