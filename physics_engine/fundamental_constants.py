"""
from nist_constants import *

Fundamental Physical Constants

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

All values from NIST CODATA 2022 via nist_constants.py.
"""

from typing import NamedTuple
import math
import nist_constants as nist


class PhysicalConstant(NamedTuple):
    """Physical constant with value, uncertainty, and units."""
    value: float
    uncertainty: float
    units: str
    description: str


# Speed of light in vacuum
c = PhysicalConstant(
    value=nist.SPEED_OF_LIGHT,
    uncertainty=0.0,
    units="m/s",
    description="Speed of light in vacuum"
)

# Planck constant
h = PhysicalConstant(
    value=nist.PLANCK_CONSTANT,
    uncertainty=0.0,
    units="J⋅s",
    description="Planck constant"
)

# Reduced Planck constant (ℏ = h/2π)
hbar = PhysicalConstant(
    value=nist.REDUCED_PLANCK_CONSTANT,
    uncertainty=0.0,
    units="J⋅s",
    description="Reduced Planck constant"
)

# Boltzmann constant
k_B = PhysicalConstant(
    value=nist.BOLTZMANN_CONSTANT,
    uncertainty=0.0,
    units="J/K",
    description="Boltzmann constant"
)

# Gravitational constant
G = PhysicalConstant(
    value=nist.GRAVITATIONAL_CONSTANT,
    uncertainty=1.5e-15,
    units="m³/(kg⋅s²)",
    description="Newtonian constant of gravitation"
)

# Elementary charge
e = PhysicalConstant(
    value=nist.ELEMENTARY_CHARGE,
    uncertainty=0.0,
    units="C",
    description="Elementary charge"
)

# Electron mass
m_e = PhysicalConstant(
    value=nist.ELECTRON_MASS,
    uncertainty=2.8e-40,
    units="kg",
    description="Electron mass"
)

# Proton mass
m_p = PhysicalConstant(
    value=nist.PROTON_MASS,
    uncertainty=5.1e-37,
    units="kg",
    description="Proton mass"
)

# Neutron mass
m_n = PhysicalConstant(
    value=nist.NEUTRON_MASS,
    uncertainty=9.5e-37,
    units="kg",
    description="Neutron mass"
)

# Avogadro constant
N_A = PhysicalConstant(
    value=nist.AVOGADRO_CONSTANT,
    uncertainty=0.0,
    units="mol⁻¹",
    description="Avogadro constant"
)

# Gas constant
R = PhysicalConstant(
    value=nist.GAS_CONSTANT,
    uncertainty=0.0,
    units="J/(mol⋅K)",
    description="Molar gas constant"
)

# Stefan-Boltzmann constant
sigma = PhysicalConstant(
    value=nist.STEFAN_BOLTZMANN_CONSTANT,
    uncertainty=0.0,
    units="W/(m²⋅K⁴)",
    description="Stefan-Boltzmann constant"
)

# Vacuum permittivity (electric constant)
epsilon_0 = PhysicalConstant(
    value=nist.VACUUM_PERMITTIVITY,
    uncertainty=1.3e-21,
    units="F/m",
    description="Vacuum permittivity"
)

# Vacuum permeability (magnetic constant)
mu_0 = PhysicalConstant(
    value=nist.VACUUM_PERMEABILITY,
    uncertainty=1.9e-16,
    units="H/m",
    description="Vacuum permeability"
)

# Fine-structure constant
alpha = PhysicalConstant(
    value=nist.FINE_STRUCTURE_CONSTANT,
    uncertainty=1.1e-12,
    units="dimensionless",
    description="Fine-structure constant"
)

# Rydberg constant
R_inf = PhysicalConstant(
    value=nist.RYDBERG_CONSTANT,
    uncertainty=0.000021,
    units="m⁻¹",
    description="Rydberg constant"
)

# Bohr radius
a_0 = PhysicalConstant(
    value=nist.BOHR_RADIUS,
    uncertainty=8.0e-21,
    units="m",
    description="Bohr radius"
)

# Standard gravity (Standard constant, not in NIST CODATA but standard)
g_0 = PhysicalConstant(
    value=9.80665,
    uncertainty=0.0,
    units="m/s²",
    description="Standard acceleration of gravity"
)

# Standard atmosphere
atm = PhysicalConstant(
    value=101325.0,
    uncertainty=0.0,
    units="Pa",
    description="Standard atmosphere"
)

# Absolute zero in Celsius
T_0 = PhysicalConstant(
    value=-273.15,
    uncertainty=0.0,
    units="°C",
    description="Absolute zero"
)


# Atomic mass unit
u = PhysicalConstant(
    value=nist.ATOMIC_MASS_UNIT,
    uncertainty=5.0e-37,
    units="kg",
    description="Atomic mass unit"
)

# Faraday constant
F = PhysicalConstant(
    value=nist.ELEMENTARY_CHARGE * nist.AVOGADRO_CONSTANT,
    uncertainty=0.0,
    units="C/mol",
    description="Faraday constant"
)


def get_all_constants() -> dict:
    """Return dictionary of all physical constants."""
    return {
        'c': c,
        'h': h,
        'hbar': hbar,
        'k_B': k_B,
        'G': G,
        'e': e,
        'm_e': m_e,
        'm_p': m_p,
        'm_n': m_n,
        'N_A': N_A,
        'R': R,
        'sigma': sigma,
        'epsilon_0': epsilon_0,
        'mu_0': mu_0,
        'alpha': alpha,
        'R_inf': R_inf,
        'a_0': a_0,
        'g_0': g_0,
        'atm': atm,
        'T_0': T_0,
        'u': u,
        'F': F,
    }


if __name__ == "__main__":
    print("Fundamental Physical Constants (NIST CODATA 2022 via nist_constants.py)")
    print("=" * 80)
    for name, const in get_all_constants().items():
        print(f"{name:8} = {const.value:15.6e} ± {const.uncertainty:10.3e} {const.units:15} # {const.description}")
