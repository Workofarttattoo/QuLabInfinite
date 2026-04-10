"""
Tests for materials_chemistry_lab.py
"""
import sys
import math
from unittest.mock import MagicMock
from pathlib import Path

# Ensure project root import resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------
# Mocking numpy and scipy to handle missing dependencies
# ---------------------------------------------------------
np_mock = MagicMock()
scipy_mock = MagicMock()
scipy_constants_mock = MagicMock()

# Set up numpy math functions
np_mock.pi = math.pi
np_mock.cos = math.cos
np_mock.sin = math.sin
np_mock.radians = math.radians
np_mock.sqrt = math.sqrt
np_mock.exp = math.exp
np_mock.log = lambda x: math.log(x) if x > 0 else 0.0
np_mock.arcsin = math.asin

# Mock numpy array operations so that scalar multiplication and operations work
class MockArray:
    def __init__(self, data):
        if isinstance(data, MockArray):
            self.data = data.data
        else:
            self.data = data

    def __mul__(self, other):
        if isinstance(other, MockArray):
            return MockArray([x * y for x, y in zip(self.data, other.data)])
        return MockArray([x * other for x in self.data])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, MockArray):
            return MockArray([x + y for x, y in zip(self.data, other.data)])
        return MockArray([x + other for x in self.data])

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, MockArray):
            return MockArray([x - y for x, y in zip(self.data, other.data)])
        return MockArray([x - other for x in self.data])

    def __rsub__(self, other):
        if isinstance(other, MockArray):
            return MockArray([y - x for x, y in zip(self.data, other.data)])
        return MockArray([other - x for x in self.data])

    def __pow__(self, other):
        return MockArray([x ** other for x in self.data])

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return MockArray(self.data[idx])
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __matmul__(self, other):
        # A simple implementation for cart_coords = frac_coords @ lattice_vectors
        if isinstance(self.data[0], (int, float)) and isinstance(other.data[0], MockArray):
            res = [0] * len(other.data[0].data)
            for i in range(len(self.data)):
                for j in range(len(other.data[0].data)):
                    res[j] += self.data[i] * other.data[i].data[j]
            return MockArray(res)
        return MockArray([])

np_mock.array = lambda x: MockArray(x)

def dot_mock(a, b):
    if isinstance(a, MockArray): a = a.data
    if isinstance(b, MockArray): b = b.data
    return sum(x*y for x, y in zip(a, b))
np_mock.dot = dot_mock

def cross_mock(a, b):
    if isinstance(a, MockArray): a = a.data
    if isinstance(b, MockArray): b = b.data
    return MockArray([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])
np_mock.cross = cross_mock

sys.modules['numpy'] = np_mock
sys.modules['scipy'] = scipy_mock
sys.modules['scipy.constants'] = scipy_constants_mock
sys.modules['scipy.spatial'] = MagicMock()
sys.modules['scipy.optimize'] = MagicMock()
sys.modules['scipy.special'] = MagicMock()
sys.modules['scipy.integrate'] = MagicMock()

# Provide mock values for scipy.constants used in the module
scipy_constants_mock.k = 1.380649e-23
scipy_constants_mock.R = 8.314462618
scipy_constants_mock.Avogadro = 6.02214076e23
scipy_constants_mock.physical_constants = {
    'Planck constant': (6.62607015e-34, '', ''),
    'electron mass': (9.1093837015e-31, '', ''),
    'elementary charge': (1.602176634e-19, '', '')
}

# ---------------------------------------------------------
# Actual Tests
# ---------------------------------------------------------
import pytest
from materials_chemistry_lab import MaterialsChemistryLab, CrystalStructure, Material, Defect, PhaseTransition

@pytest.fixture
def lab():
    return MaterialsChemistryLab()

@pytest.fixture
def cubic_crystal():
    return CrystalStructure(name="TestCubic", lattice_type="cubic", a=5.0)

def test_crystal_structure_volume(lab, cubic_crystal):
    vol = lab.calculate_volume(cubic_crystal)
    assert abs(vol - 125.0) < 1e-5

def test_material_density(lab, cubic_crystal):
    material = Material(
        name="TestMat",
        composition={"C": 1},
        crystal_structure=cubic_crystal,
        density=0.0,
        melting_point=1000
    )
    density = lab.calculate_density(material)
    assert density > 0.0

def test_generate_lattice_vectors(lab, cubic_crystal):
    vectors = lab.generate_lattice_vectors(cubic_crystal)
    assert len(vectors) == 3
    # Check v1
    v1 = vectors[0]
    assert abs(v1[0] - 5.0) < 1e-5
    assert abs(v1[1] - 0.0) < 1e-5
    assert abs(v1[2] - 0.0) < 1e-5

def test_phase_diagram(lab):
    f1, f2 = lab.lever_rule(overall_composition=0.4, T=1000, phase1_comp=0.2, phase2_comp=0.8)
    assert abs(f1 - 0.6666666666666666) < 1e-5
    assert abs(f2 - 0.3333333333333333) < 1e-5

def test_clausius_clapeyron(lab):
    dp_dt = lab.clausius_clapeyron(T=300, dH=6000, dV=0.5)
    assert dp_dt > 0

def test_defect_concentration(lab):
    conc = lab.vacancy_concentration(formation_energy=1.0, temperature=1000)
    assert conc > 0.0

    conc_s = lab.schottky_defect_concentration(formation_energy=2.0, temperature=1000)
    assert conc_s > 0.0

    conc_f = lab.frenkel_defect_concentration(formation_energy=1.5, temperature=1000)
    assert conc_f > 0.0

def test_kroger_vink_notation(lab):
    defect = Defect(type="vacancy", site=np_mock.array([0,0,0]), element="O", charge=2)
    notation = lab.kroger_vink_notation(defect)
    assert "V_O" in notation
    assert "••" in notation

def test_electronic_properties(lab):
    Eg = lab.band_gap_temperature(Eg0=1.17, alpha=4.73e-4, beta=636, temperature=300)
    assert Eg < 1.17
    assert Eg > 0

    n, p = lab.carrier_concentration(band_gap=1.1, temperature=300)
    assert n > 0
    assert p > 0

def test_xrd(lab, cubic_crystal):
    d = lab.d_spacing(cubic_crystal, 1, 0, 0)
    assert abs(d - 5.0) < 1e-5

    theta = lab.bragg_angle(d, wavelength=1.5418)
    assert theta is not None
    assert theta > 0

def test_mechanical_properties(lab):
    C = MockArray([166, 64, 80])
    k_v, k_r = lab.bulk_modulus_voigt_reuss(C)
    assert k_v > 0
    assert k_r > 0

    hardness = lab.hardness_estimation(bulk_modulus=98, shear_modulus=80)
    assert hardness >= 0

    strength = lab.theoretical_strength(young_modulus=130, surface_energy=1.5, lattice_parameter=5e-10)
    assert strength > 0

def test_thermal_properties(lab):
    theta_D = lab.debye_temperature(bulk_modulus=98, molar_mass=28.09, density=2.33, n_atoms=2)
    assert theta_D > 0

    alpha = lab.thermal_expansion_coefficient(gruneisen=1.5, bulk_modulus=100, heat_capacity=25, molar_volume=10)
    assert alpha > 0

def test_diffusion(lab):
    D = lab.diffusion_coefficient(D0=1e-4, activation_energy=200, temperature=1000)
    assert D > 0
    assert D < 1e-4

def test_nanoparticles(lab):
    Tm_nano = lab.nanoparticle_melting_point(bulk_melting=1687, particle_size=10, surface_energy=1.0, latent_heat=50000)
    assert Tm_nano < 1687
    assert Tm_nano > 0

    nucleation = lab.nucleation_rate(supersaturation=1.5, surface_energy=0.1, temperature=300)
    assert nucleation >= 0
