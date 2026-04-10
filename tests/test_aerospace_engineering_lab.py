"""
Unit tests for the Aerospace Engineering Lab.
"""

import sys
from pathlib import Path
import pytest

# Ensure project root import resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aerospace_engineering_lab import AerospaceEngineeringLab

def test_standard_atmosphere_negative_altitude():
    """Test that a negative altitude raises a ValueError."""
    lab = AerospaceEngineeringLab()
    with pytest.raises(ValueError, match="Altitude must be non-negative"):
        lab.standard_atmosphere(-1.0)

def test_standard_atmosphere_zero_altitude():
    """Test atmospheric properties at sea level (0 altitude)."""
    lab = AerospaceEngineeringLab()
    atm = lab.standard_atmosphere(0.0)

    assert atm['temperature'] == pytest.approx(288.15)
    assert atm['pressure'] == pytest.approx(101325.0)
    assert atm['density'] == pytest.approx(1.225, rel=1e-3)
    # Speed of sound roughly 340.29 m/s at sea level
    assert atm['speed_of_sound'] == pytest.approx(340.294, rel=1e-3)

def test_standard_atmosphere_valid_altitude():
    """Test atmospheric properties at a valid positive altitude."""
    lab = AerospaceEngineeringLab()
    atm = lab.standard_atmosphere(10000.0) # 10 km

    # Values from standard atmosphere tables
    assert atm['temperature'] == pytest.approx(223.15, rel=1e-2)
    assert atm['pressure'] == pytest.approx(26499.9, rel=1e-2)
