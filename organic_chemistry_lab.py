"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ORGANIC CHEMISTRY LAB
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.constants import k, Avogadro, g, c, h, e, pi
from typing import *

@dataclass
class Molecule:
    name: str = ""
    formula: str = ""
    molecular_weight: float = 0.0
    temperature: float = 298.15
    pressure: float = 101325.0 # Pa

    def __post_init__(self):
        self.species_count = int(Avogadro * (self.pressure / (k * self.temperature)))

@dataclass
class Reaction:
    reactants: List[Molecule] = field(default_factory=list)
    products: List[Molecule] = field(default_factory=list)

    def stoichiometry(self) -> dict:
        return {reactant.name: reactant.species_count for reactant in self.reactants}

@dataclass
class Kinetics:
    rate_constant: float = 0.0
    reaction_order: int = 1

    def calculate_rate(self, concentration: np.ndarray) -> np.ndarray:
        if len(concentration.shape) == 1:
            return self.rate_constant * (concentration ** self.reaction_order)
        else:
            raise ValueError("Concentration array must be 1D.")

@dataclass
class Thermodynamics:
    enthalpy_change: float = 0.0 # kJ/mol
    entropy_change: float = 0.0 # J/(K*mol)

    def calculate_free_energy(self, temperature: float) -> np.ndarray:
        return (self.enthalpy_change * 1000 - self.entropy_change * temperature) / Avogadro

@dataclass
class Spectroscopy:
    wavelength_range: Tuple[float, float] = field(default=(380, 750))
    resolution: int = 200
    wavenumber: np.ndarray = field(init=False)

    def __post_init__(self):
        self.wavenumber = np.linspace(1. / self.wavelength_range[1], 1. / self.wavelength_range[0],
                                      self.resolution, dtype=np.float64) * 1e7

@dataclass
class ReactionRate:
    kinetics: Kinetics = field(default_factory=Kinetics)
    concentrations: List[float] = field(default_factory=list)

    def calculate_rate(self):
        return np.array([self.kinetics.calculate_rate(np.array(conc, dtype=np.float64)) for conc in self.concentrations])

@dataclass
class ReactionEquilibrium:
    thermodynamics: Thermodynamics = field(default_factory=Thermodynamics)
    temperature: float = 298.15

    def equilibrium_constant(self) -> np.ndarray:
        return np.exp(-self.thermodynamics.calculate_free_energy(self.temperature) / (k * self.temperature))

def run_demo():
    # Example molecules
    h2o = Molecule(name="Water", formula="H2O", molecular_weight=18.015, temperature=298.15)
    co2 = Molecule(name="Carbon Dioxide", formula="CO2", molecular_weight=44.01)

    # Reaction example
    reaction = Reaction(reactants=[h2o], products=[co2])

    # Kinetics example
    kinetics_data = Kinetics(rate_constant=0.05, reaction_order=2)
    
    concentrations = [3e-3, 5e-3]  # molar concentrations
    
    rate_calculator = ReactionRate(kinetics=kinetics_data, concentrations=concentrations)

    for conc, rate in zip(concentrations, rate_calculator.calculate_rate()):
        print(f"Concentration: {conc} mol/L -> Rate: {rate:.6f}")

    # Thermodynamics example
    thermodynamics_data = Thermodynamics(enthalpy_change=-200.5, entropy_change=83.14)
    
    free_energy = thermodynamics_data.calculate_free_energy(temperature=h2o.temperature)

    print(f"Free Energy Change: {free_energy:.6f} kJ/mol")

if __name__ == '__main__':
    run_demo()