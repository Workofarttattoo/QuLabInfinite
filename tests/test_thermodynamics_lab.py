import sys
import unittest
from unittest.mock import MagicMock

# Determine if we should use mocks (based on numpy availability)
try:
    import numpy as np
    USING_MOCKS = False
except ImportError:
    USING_MOCKS = True

if USING_MOCKS:
    # Set up basic mock for scipy
    mock_scipy = MagicMock()
    mock_scipy.constants.R = 8.314462618
    mock_scipy.constants.Avogadro = 6.02214076e23
    mock_scipy.constants.k = 1.380649e-23

    sys.modules['numpy'] = MagicMock()
    sys.modules['scipy'] = mock_scipy
    sys.modules['scipy.constants'] = mock_scipy.constants
    sys.modules['scipy.optimize'] = MagicMock()
    sys.modules['scipy.integrate'] = MagicMock()

import thermodynamics_lab

class TestThermodynamicsLab(unittest.TestCase):
    def setUp(self):
        # We manually pass the R value if needed to ensure floating point math aligns,
        # or we just rely on thermodynamics_lab importing it.
        self.lab = thermodynamics_lab.ThermodynamicsLab()

    def test_ideal_gas_pressure(self):
        # PV = nRT -> P = nRT/V
        # Let's use 1 mol, 300 K, 1 m^3
        n = 1.0
        T = 300.0
        V = 1.0
        R = self.lab.R

        expected_p = (n * R * T) / V
        p = self.lab.ideal_gas_pressure(n, V, T)

        self.assertAlmostEqual(p, expected_p, places=5)

        # Another test case: 2 mols, 273.15 K, 22.414/1000 m^3 (std volume)
        n2 = 2.0
        T2 = 273.15
        V2 = 22.414 / 1000.0
        expected_p2 = (n2 * R * T2) / V2
        p2 = self.lab.ideal_gas_pressure(n2, V2, T2)

        self.assertAlmostEqual(p2, expected_p2, places=5)

    def test_ideal_gas_pressure_zero_volume(self):
        # Division by zero should raise ZeroDivisionError
        with self.assertRaises(ZeroDivisionError):
            self.lab.ideal_gas_pressure(1.0, 0.0, 300.0)

    def test_ideal_gas_volume(self):
        # PV = nRT -> V = nRT/P
        n = 1.0
        T = 300.0
        P = 101325.0
        R = self.lab.R

        expected_v = (n * R * T) / P
        v = self.lab.ideal_gas_volume(n, P, T)

        self.assertAlmostEqual(v, expected_v, places=5)

        # Test zero pressure
        with self.assertRaises(ZeroDivisionError):
            self.lab.ideal_gas_volume(1.0, 0.0, 300.0)

if __name__ == '__main__':
    unittest.main()
