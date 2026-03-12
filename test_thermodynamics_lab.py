import pytest
import numpy as np
from thermodynamics_lab import ThermodynamicsLab, Component

def test_heat_capacity_integration_analytical():
    lab = ThermodynamicsLab()
    co2 = Component(
        name="CO2",
        molecular_weight=44.01,
        critical_temperature=304.13,
        critical_pressure=7.377e6,
        critical_volume=0.0941e-3,
        acentric_factor=0.224,
        normal_boiling_point=194.7,
        heat_of_vaporization=25200,
        heat_capacity_params=[22.26, 5.981e-2, -3.501e-5, 7.469e-9]
    )

    # Original implementation uses quad
    res1 = lab.heat_capacity_integration(300, 500, lab.heat_capacity_polynomial, co2.heat_capacity_params)
    assert np.isclose(res1, 8194.7184)

def test_entropy_ideal_gas():
    lab = ThermodynamicsLab()
    co2 = Component(
        name="CO2",
        molecular_weight=44.01,
        critical_temperature=304.13,
        critical_pressure=7.377e6,
        critical_volume=0.0941e-3,
        acentric_factor=0.224,
        normal_boiling_point=194.7,
        heat_of_vaporization=25200,
        heat_capacity_params=[22.26, 5.981e-2, -3.501e-5, 7.469e-9]
    )

    res1 = lab.entropy_ideal_gas(500, 101325, co2)
    assert np.isclose(res1, 220.8969374419937)
