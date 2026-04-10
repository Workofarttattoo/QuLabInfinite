"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

PHYSICAL CHEMISTRY LAB
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.constants import k, Avogadro, g, c, h, e, pi
from typing import List

@dataclass
class PhysicalConstants:
    boltzmann: float = field(init=False)
    avogadro: float = field(init=False)
    gravity: float = field(init=False)
    speed_of_light: float = field(init=False)
    planck_constant: float = field(init=False)
    elementary_charge: float = field(init=False)

    def __post_init__(self):
        self.boltzmann = k
        self.avogadro = Avogadro
        self.gravity = g
        self.speed_of_light = c
        self.planck_constant = h
        self.elementary_charge = e

@dataclass
class IdealGasLaw:
    constants: PhysicalConstants
    r_gas: float = field(init=False)

    def __post_init__(self):
        self.r_gas = self.constants.k * self.constants.Avogadro

    def ideal_gas_equation(self, p: np.ndarray, v: np.ndarray, n: int, t: float) -> np.ndarray:
        return (p * v - n * self.r_gas * t)

@dataclass
class KineticTheoryOfGases:
    constants: PhysicalConstants

    def root_mean_square_velocity(self, temperature: float, mass: float) -> float:
        return np.sqrt((3 * self.constants.k * temperature / mass))

@dataclass
class QuantumChemistry:
    constants: PhysicalConstants

    def bohr_radius(self) -> float:
        return ((4 * pi * e**2) / (m_electron * c**2))**(1/2)

    @property
    def m_electron(self):
        # Assuming the mass of an electron in atomic units for simplicity
        return 9.10938356e-31

@dataclass
class Thermodynamics:
    constants: PhysicalConstants
    
    def ideal_carnot_efficiency(self, t_hot: float, t_cold: float) -> float:
        return (t_hot - t_cold) / t_hot

def run_demo():
    print("Running Physical Chemistry Lab Demo...")
    
    p = np.array([1.0e5], dtype=np.float64)
    v = np.array([10.0], dtype=np.float64)
    n = 1
    t = 300.0

    const = PhysicalConstants()
    gas_law = IdealGasLaw(constants=const)

    print(f"Ideal Gas Law Test: {gas_law.ideal_gas_equation(p, v, n, t)}")

    ktg = KineticTheoryOfGases(constants=const)
    temperature = 300
    mass = 28e-3 / const.avogadro
    rms_velocity = ktg.root_mean_square_velocity(temperature, mass)
    
    print(f"Root Mean Square Velocity Test: {rms_velocity}")

    qc = QuantumChemistry(constants=const)
    bohr_radius = qc.bohr_radius()

    print(rf"Bohr Radius Test: \({bohr_radius:.6e}\) meters")

    thermodynamics = Thermodynamics(constants=const)

    t_hot, t_cold = 500.0, 300.0
    efficiency = thermodynamics.ideal_carnot_efficiency(t_hot, t_cold)
    
    print(f"Ideal Carnot Efficiency Test: {efficiency:.2f}")

if __name__ == '__main__':
    run_demo()