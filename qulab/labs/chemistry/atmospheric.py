"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ATMOSPHERIC CHEMISTRY LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass
from scipy.constants import physical_constants

@dataclass
class Molecule:
    name: str
    molecular_weight: float  # in g/mol
    concentration: float     # in molecules/cm^3
    cross_section: float     # in cm^2

@dataclass
class GasMixtures:
    temperature: float       # in Kelvin
    pressure: float          # in Pa
    molecules: list[Molecule]

def calculate_mole_fraction(molecules: list[Molecule], total_concentration: float) -> np.ndarray[float]:
    mole_fractions = []
    for molecule in molecules:
        mole_fraction = (molecule.concentration * molecule.molecular_weight / 1000.0) / total_concentration
        mole_fractions.append(mole_fraction)
    return np.array(mole_fractions, dtype=np.float64)

def calculate_total_mass_density(gas_mixtures: GasMixtures) -> float:
    total_mole_fraction = sum(m.concentration * m.molecular_weight for m in gas_mixtures.molecules) / 1000.0
    total_concentration = gas_mixtures.pressure / (physical_constants['Boltzmann constant'][-3] * gas_mixtures.temperature)
    return total_concentration * total_mole_fraction

def calculate_absorption_coefficient(gas_mixtures: GasMixtures) -> np.ndarray[float]:
    absorption_coefficients = []
    for molecule in gas_mixtures.molecules:
        absorption_coefficient = molecule.concentration * molecule.cross_section
        absorption_coefficients.append(absorption_coefficient)
    return np.array(absorption_coefficients, dtype=np.float64)

class AtmosphericChemistryLab:
    def __init__(self, temperature: float, pressure: float, molecules: list[Molecule]):
        self.gas_mixtures = GasMixtures(temperature=temperature, pressure=pressure, molecules=molecules)
    
    def run_analysis(self):
        total_concentration = self.gas_mixtures.pressure / (physical_constants['Boltzmann constant'][-3] * self.gas_mixtures.temperature)
        
        mole_fractions = calculate_mole_fraction(self.gas_mixtures.molecules, total_concentration)
        total_mass_density = calculate_total_mass_density(self.gas_mixtures)
        
        absorption_coefficients = calculate_absorption_coefficient(self.gas_mixtures)
        
        return mole_fractions, total_mass_density, absorption_coefficients

def run_demo():
    no2 = Molecule(name="NO2", molecular_weight=46.01, concentration=5e18, cross_section=3e-19)
    o3 = Molecule(name="O3", molecular_weight=47.998, concentration=2e18, cross_section=5e-20)
    
    temperature = 298.15
    pressure = 101325.0
    
    lab = AtmosphericChemistryLab(temperature=temperature, pressure=pressure, molecules=[no2, o3])
    mole_fractions, total_mass_density, absorption_coefficients = lab.run_analysis()
    
    print(f"Mole Fractions: {mole_fractions}")
    print(f"Total Mass Density: {total_mass_density:.6f} kg/m^3")
    print(f"Absorption Coefficients: {absorption_coefficients}")

if __name__ == '__main__':
    run_demo()