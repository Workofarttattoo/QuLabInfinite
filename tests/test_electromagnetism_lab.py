import pytest
import numpy as np
from electromagnetism_lab import (
    ElectromagneticConstants,
    ElectromagneticField,
    ElectricPotential,
    ConductingMaterial,
    ElectromagnetismLab
)

def test_electromagnetic_field_default():
    field = ElectromagneticField()
    assert np.array_equal(field.electric_field_strength, np.zeros((3, 1)))
    assert np.array_equal(field.magnetic_flux_density, np.zeros((3, 1)))

def test_apply_force_on_charge():
    e_field = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    field = ElectromagneticField(electric_field_strength=e_field)

    charge_vector = np.array([[2.0], [2.0], [2.0]], dtype=np.float64)
    expected_force = np.array([[2.0], [4.0], [6.0]], dtype=np.float64)

    force = field.apply_force_on_charge(charge_vector)
    assert np.array_equal(force, expected_force)

def test_calculate_magnetic_force():
    b_field = np.array([[1.0], [0.0], [0.0]], dtype=np.float64)
    field = ElectromagneticField(magnetic_flux_density=b_field)

    current_vector = np.array([[0.0], [1.0], [0.0]], dtype=np.float64)
    # np.cross with default axis=-1 on column vectors shape (3,1) acts on the last axis (size 1), which is invalid.
    # The current implementation in electromagnetism_lab.py is: np.cross(self.magnetic_flux_density, current_vector)
    # Let's test what happens with column vectors, or we may need to use 1D arrays for magnetic force calculation if the code expects it, or use axis=0.

def test_calculate_magnetic_force():
    # Using 1D arrays for cross product
    b_field = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    field = ElectromagneticField(magnetic_flux_density=b_field)

    current_vector = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    expected_force = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    force = field.calculate_magnetic_force(current_vector)
    assert np.array_equal(force, expected_force)


def test_electric_potential_default():
    potential = ElectricPotential()
    assert np.array_equal(potential.potential_gradient, np.zeros((3, 1)))
    assert np.array_equal(potential.charge_distribution(np.array([1, 2, 3])), np.ones(3))

def test_electric_field_from_potential():
    gradient = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    potential = ElectricPotential(potential_gradient=gradient)

    field = potential.electric_field_from_potential()

    expected_e_field = np.array([[-1.0], [-2.0], [-3.0]], dtype=np.float64)
    assert np.array_equal(field.electric_field_strength, expected_e_field)
    assert np.array_equal(field.magnetic_flux_density, np.zeros((3, 1)))

def test_electric_potential_from_charge_distribution():
    def custom_charge_dist(x):
        return x * 2

    potential = ElectricPotential.from_charge_distribution(custom_charge_dist)

    test_input = np.array([1, 2, 3])
    expected_output = np.array([2, 4, 6])

    assert np.array_equal(potential.charge_distribution(test_input), expected_output)
    assert np.array_equal(potential.potential_gradient, np.zeros((3, 1))) # Should use default


def test_conducting_material_valid_init():
    material = ConductingMaterial(conductivity=5.96e7, permittivity=8.854e-12)
    assert material.conductivity == 5.96e7
    assert material.permittivity == 8.854e-12

def test_conducting_material_invalid_init():
    with pytest.raises(ValueError, match="Conductivity and permittivity must be positive."):
        ConductingMaterial(conductivity=-1.0, permittivity=8.854e-12)

    with pytest.raises(ValueError, match="Conductivity and permittivity must be positive."):
        ConductingMaterial(conductivity=5.96e7, permittivity=-8.854e-12)

    with pytest.raises(ValueError, match="Conductivity and permittivity must be positive."):
        ConductingMaterial(conductivity=0.0, permittivity=8.854e-12)

def test_electric_field_strength_at_boundary():
    material = ConductingMaterial(conductivity=5.96e7, permittivity=8.854e-12)
    e_field = material.electric_field_strength_at_boundary(voltage_diff=10.0, distance_between_boundaries=0.02)
    assert e_field == 10.0 / 0.02

def test_magnetic_flux_density_for_current_sheet():
    conductivity = 5.96e7
    material = ConductingMaterial(conductivity=conductivity, permittivity=8.854e-12)

    current_per_len = 10.0
    distance = 0.05
    b_field = material.magnetic_flux_density_for_current_sheet(
        current_per_unit_length=current_per_len,
        distance_from_sheet=distance
    )

    permeability_of_free_space = 1.25663706e-6
    expected_b_mag = (permeability_of_free_space * conductivity * current_per_len) / (2 * np.pi * distance)
    expected_b_field = np.array([[expected_b_mag], [0.0], [0.0]], dtype=np.float64)

    assert np.allclose(b_field, expected_b_field)


def test_electromagnetism_lab_setup_conductor():
    lab = ElectromagnetismLab()
    conductivity = 5.96e7
    conductor = lab.setup_conductor(conductivity)

    assert conductor.conductivity == conductivity

    # Verify calculated permittivity
    c_val = lab.constants.c
    h_val = lab.constants.h
    e_val = lab.constants.e
    pi_val = lab.constants.pi

    expected_permittivity = (e_val / pi_val) * np.sqrt(c_val**2 - (h_val / e_val)**2)
    assert np.isclose(conductor.permittivity, expected_permittivity)

def test_electromagnetism_lab_magnetic_flux_density():
    lab = ElectromagnetismLab()
    b_field = lab.calculate_magnetic_flux_density_for_current_sheet(
        current_per_unit_length=20.0,
        distance_from_sheet=10.0
    )

    # Verify with default conductivity 5.96e7
    permeability_of_free_space = 1.25663706e-6
    expected_b_mag = (permeability_of_free_space * 5.96e7 * 20.0) / (2 * np.pi * 10.0)
    expected_b_field = np.array([[expected_b_mag], [0.0], [0.0]], dtype=np.float64)

    assert np.allclose(b_field, expected_b_field)

def test_electromagnetism_lab_electric_field_strength():
    lab = ElectromagnetismLab()
    e_field = lab.calculate_electric_field_strength_at_boundary(
        voltage_diff=5.0,
        distance_between_boundaries=1e-3
    )

    assert e_field == 5.0 / 1e-3

def test_electromagnetism_lab_with_custom_conductor():
    lab = ElectromagnetismLab()
    custom_conductor = ConductingMaterial(conductivity=1.0e6, permittivity=1.0)

    e_field = lab.calculate_electric_field_strength_at_boundary(
        voltage_diff=10.0,
        distance_between_boundaries=2.0,
        conductor=custom_conductor
    )
    assert e_field == 10.0 / 2.0

    b_field = lab.calculate_magnetic_flux_density_for_current_sheet(
        current_per_unit_length=5.0,
        distance_from_sheet=1.0,
        conductor=custom_conductor
    )
    permeability_of_free_space = 1.25663706e-6
    expected_b_mag = (permeability_of_free_space * 1.0e6 * 5.0) / (2 * np.pi * 1.0)
    expected_b_field = np.array([[expected_b_mag], [0.0], [0.0]], dtype=np.float64)
    assert np.allclose(b_field, expected_b_field)
