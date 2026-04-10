import unittest
import io
import sys
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch
import importlib

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class MockConstants:
    hbar = 1.054e-34
    m_e = 9.109e-31
    e = 1.602e-19
    epsilon_0 = 8.854e-12
    eV = 1.602e-19

class MockNumpyArray(list):
    """A list that pretends to be a numpy array for basic operations"""
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, data):
        super().__init__(data)

    def __add__(self, other):
        if isinstance(other, (int, float)): return MockNumpyArray([x + other for x in self])
        if isinstance(other, list) and len(other) == len(self): return MockNumpyArray([x + y for x, y in zip(self, other)])
        return super().__add__(other)

    def __radd__(self, other): return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)): return MockNumpyArray([x - other for x in self])
        if isinstance(other, list) and len(other) == len(self): return MockNumpyArray([x - y for x, y in zip(self, other)])
        return super().__sub__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)): return MockNumpyArray([x * other for x in self])
        return super().__mul__(other)

    def __rmul__(self, other): return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)): return MockNumpyArray([x / other for x in self])
        return super().__truediv__(other)

    @property
    def T(self):
        return self

sys.modules['numpy'] = MagicMock()
sys.modules['numpy'].where = MagicMock(return_value=[0])
sys.modules['numpy'].arange = MagicMock(return_value=MockNumpyArray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]))
sys.modules['numpy'].diag = MagicMock(return_value=MockNumpyArray([MockNumpyArray([0.0] * 10)] * 10))
sys.modules['numpy'].sum = MagicMock(return_value=1.0)
sys.modules['numpy'].max = MagicMock(return_value=1.0)
sys.modules['numpy'].abs = MagicMock(return_value=MagicMock(__pow__=MagicMock(return_value=MagicMock(__mul__=MagicMock(return_value=[0.0])))))
sys.modules['numpy'].random = MagicMock()
sys.modules['numpy'].random.randn = MagicMock(return_value=MockNumpyArray([MockNumpyArray([0.0] * 10)] * 10))

# Provide constants locally in scipy mock to avoid MagicMock division
sys.modules['scipy'] = MagicMock()
sys.modules['scipy'].constants = MockConstants()
sys.modules['scipy.constants'] = MockConstants()
sys.modules['scipy.special'] = MagicMock()
sys.modules['scipy.integrate'] = MagicMock()
sys.modules['scipy.linalg'] = MagicMock()
sys.modules['scipy.sparse'] = MagicMock()

import quantum_mechanics_lab
# Ensure the module has the right constants
quantum_mechanics_lab.constants = MockConstants()

try:
    from quantum_mechanics_lab import run_demo, QuantumMechanicsLab
except ImportError:
    pass

class TestQuantumMechanicsLab(unittest.TestCase):

    @patch('quantum_mechanics_lab.QuantumMechanicsLab')
    def test_run_demo(self, MockLab):
        """Test that run_demo executes without raising exceptions and generates output."""

        # Setup mock instance
        mock_lab_instance = MockLab.return_value
        mock_lab_instance.name = "Quantum Mechanics Lab"
        mock_lab_instance.version = "1.0"

        # Mock calculations returned by the lab to return real numbers so formatting works
        mock_lab_instance.solve_schrodinger_1d.return_value = (None, [1.602e-19, 3.204e-19, 4.806e-19], None)
        mock_lab_instance.harmonic_oscillator_eigenstates.return_value = (None, [1.602e-19, 3.204e-19, 4.806e-19, 6.408e-19, 8.01e-19, 9.612e-19], None)

        mock_psi_h = MagicMock()
        mock_psi_h.shape = (30, 30, 30)
        # return a mock that will be fine for absolute value and max
        mock_lab_instance.hydrogen_wavefunction.return_value = (None, mock_psi_h)

        mock_lab_instance.quantum_tunneling_probability.return_value = 0.5
        mock_lab_instance.perturbation_theory_first_order.return_value = ([0.1, 0.2, 0.3], None)
        mock_lab_instance.coherent_state.return_value = [0.1, 0.2]

        f = io.StringIO()
        with redirect_stdout(f):
            try:
                run_demo()
            except Exception as e:
                self.fail(f"run_demo() raised an exception unexpectedly: {e}")

        output = f.getvalue()

        # Verify the mock methods were called correctly
        mock_lab_instance.solve_schrodinger_1d.assert_called_once()
        mock_lab_instance.harmonic_oscillator_eigenstates.assert_called_once()
        mock_lab_instance.hydrogen_wavefunction.assert_called_once_with(n=2, l=1, m=0, n_points=30)
        mock_lab_instance.quantum_tunneling_probability.assert_called_once()
        mock_lab_instance.perturbation_theory_first_order.assert_called_once()
        mock_lab_instance.coherent_state.assert_called_once_with(2.0 + 1.0j, n_max=20)

        # Verify standard output content
        self.assertIn("Initializing Quantum Mechanics Lab", output)
        self.assertIn("Particle in a Box", output)
        self.assertIn("Quantum Harmonic Oscillator", output)
        self.assertIn("Hydrogen Atom Wavefunctions", output)
        self.assertIn("Quantum Tunneling", output)
        self.assertIn("Perturbation Theory", output)
        self.assertIn("Coherent State", output)
        self.assertIn("Quantum Mechanics Lab demonstration complete!", output)

if __name__ == '__main__':
    unittest.main()
