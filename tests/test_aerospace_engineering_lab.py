"""
Unit tests for aerospace engineering lab.
"""

import sys
import pytest
from pathlib import Path

# Ensure project root import resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aerospace_engineering_lab import AerospaceEngineeringLab


def test_standard_atmosphere_negative_altitude():
    """Test that a negative altitude raises a ValueError."""
    lab = AerospaceEngineeringLab()
    with pytest.raises(ValueError, match="Altitude must be non-negative"):
        lab.standard_atmosphere(-10.0)


def test_standard_atmosphere_zero_altitude():
    """Test that an altitude of 0 meters returns correct sea level properties."""
    lab = AerospaceEngineeringLab()
    atm = lab.standard_atmosphere(0.0)

    # Values might have slight precision differences due to calculation with gas constant
    assert atm['temperature'] == pytest.approx(lab.temperature_sl)
    assert atm['pressure'] == pytest.approx(lab.pressure_sl)
    assert atm['density'] == pytest.approx(lab.density_sl, rel=1e-3)
    # Speed of sound is derived but we can test it's positive
    assert atm['speed_of_sound'] > 0


def test_standard_atmosphere_positive_altitude():
    """Test that standard atmosphere calculates values for a positive altitude."""
    lab = AerospaceEngineeringLab()
    atm = lab.standard_atmosphere(10000.0)

    # Values should be different from sea level but valid
    assert atm['temperature'] < lab.temperature_sl
    assert atm['pressure'] < lab.pressure_sl
    assert atm['density'] < lab.density_sl
    assert atm['speed_of_sound'] > 0
