import pytest
import sys
import math
from unittest.mock import MagicMock

# --- CONDITIONAL MOCKING FOR NUMPY/SCIPY ---
USING_MOCKS = False

try:
    import numpy as np
    from scipy.constants import G
except ImportError:
    USING_MOCKS = True

    class MockNumpy:
        def sqrt(self, x):
            return math.sqrt(x)

        pi = math.pi
        bool_ = bool

        def __getattr__(self, name):
            if name == 'bool_':
                return bool
            return MagicMock()

    mock_numpy = MockNumpy()
    sys.modules['numpy'] = mock_numpy

    class MockScipyConstants:
        G = 6.67430e-11
        c = 299792458.0
        sigma = 5.670374419e-8
        k = 1.380649e-23
        m_p = 1.67262192369e-27
        m_e = 9.1093837015e-31
        h = 6.62607015e-34
        pi = math.pi

    sys.modules['scipy.constants'] = MockScipyConstants()

    class MockScipy:
        constants = MockScipyConstants()
        integrate = MagicMock()
        optimize = MagicMock()
        interpolate = MagicMock()

    sys.modules['scipy'] = MockScipy()

    # We also need to mock scipy.integrate etc that might be used at module level in astrophysics_lab
    sys.modules['scipy.integrate'] = MagicMock()
    sys.modules['scipy.optimize'] = MagicMock()
    sys.modules['scipy.interpolate'] = MagicMock()


# --- IMPORT MODULE UNDER TEST ---
import astrophysics_lab
from astrophysics_lab import AstrophysicsLab

# --- TEARDOWN FIXTURE ---
@pytest.fixture(scope="module", autouse=True)
def cleanup_mocks():
    yield
    if USING_MOCKS:
        for mod in ['numpy', 'scipy.constants', 'scipy', 'scipy.integrate', 'scipy.optimize', 'scipy.interpolate']:
            if mod in sys.modules:
                del sys.modules[mod]

        if 'astrophysics_lab' in sys.modules:
            del sys.modules['astrophysics_lab']

class TestAstrophysicsLab:
    @pytest.fixture
    def orbital_mechanics(self):
        lab = AstrophysicsLab()
        return lab.OrbitalMechanics()

    def test_calculate_orbital_period_earth(self, orbital_mechanics):
        # Earth's orbit around the Sun
        semi_major_axis_m = 1.496e11  # 1 AU in meters
        mass_sun_kg = 1.989e30        # Mass of Sun in kg

        period_s = orbital_mechanics.calculate_orbital_period(
            semi_major_axis=semi_major_axis_m,
            primary_mass=mass_sun_kg
        )

        # Earth's orbital period is roughly 365.25 days -> ~3.15e7 seconds
        period_days = period_s / (24 * 3600)
        assert abs(period_days - 365.25) / 365.25 < 0.05

    def test_calculate_orbital_period_moon(self, orbital_mechanics):
        # Moon's orbit around Earth
        semi_major_axis_m = 3.844e8   # Earth to Moon distance in meters
        mass_earth_kg = 5.972e24      # Mass of Earth in kg

        period_s = orbital_mechanics.calculate_orbital_period(
            semi_major_axis=semi_major_axis_m,
            primary_mass=mass_earth_kg
        )

        # Moon's orbital period is roughly 27.3 days
        period_days = period_s / (24 * 3600)
        assert abs(period_days - 27.3) / 27.3 < 0.05

    def test_calculate_orbital_period_edge_cases(self, orbital_mechanics):
        # Test with very small mass or distance
        # Period should still be calculable and positive as long as inputs > 0
        period_s = orbital_mechanics.calculate_orbital_period(
            semi_major_axis=1000.0,
            primary_mass=1000.0
        )
        assert period_s > 0

        # Exact Kepler's Third Law verification T = 2*pi * sqrt(a^3 / GM)
        a = 1e6
        M = 1e20
        # G constant is either mocked or real, both are ~6.6743e-11
        import numpy as np
        if hasattr(np, 'pi') and not isinstance(np.pi, MagicMock):
            pi = np.pi
            sqrt = np.sqrt
        else:
            pi = math.pi
            sqrt = math.sqrt
        G_val = 6.67430e-11
        expected = 2 * pi * sqrt((a**3) / (G_val * M))
        actual = orbital_mechanics.calculate_orbital_period(a, M)
        assert abs(actual - expected) / expected < 1e-4
