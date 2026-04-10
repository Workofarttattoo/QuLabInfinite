"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

CHEMICAL ENGINEERING LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.constants import pi, gas_constant

@dataclass
class ChemicalSystem:
    temperature: float  # in Kelvin
    pressure: float     # in Pascal
    composition: dict   # molar fractions of components
    volume: float       # in cubic meters
    flow_rate: float    # in moles per second

    def __post_init__(self):
        self.total_moles = sum(self.composition.values())
        self.components = list(self.composition.keys())

    def calculate_partial_pressures(self) -> dict:
        partial_pressures = {component: self.pressure * frac for component, frac in self.composition.items()}
        return partial_pressures

    def isothermal_expansion(self, new_volume: float) -> 'ChemicalSystem':
        """
        Simulate an isothermal expansion of the system to a new volume.
        
        Args:
            new_volume (float): The new volume for the system in cubic meters.

        Returns:
            ChemicalSystem: A new instance representing the expanded state.
        """
        new_pressure = self.pressure * (self.volume / new_volume)
        return ChemicalSystem(temperature=self.temperature, pressure=new_pressure,
                              composition=self.composition, volume=new_volume, flow_rate=self.flow_rate)

    def adiabatic_expansion(self, new_volume: float) -> 'ChemicalSystem':
        """
        Simulate an adiabatic expansion of the system to a new volume.
        
        Args:
            new_volume (float): The new volume for the system in cubic meters.

        Returns:
            ChemicalSystem: A new instance representing the expanded state.
        """
        k = 1.4  # example value for gamma or Cp/Cv ratio
        p_ratio = np.power(new_volume / self.volume, -k)
        new_pressure = self.pressure * p_ratio
        return ChemicalSystem(temperature=self.temperature * p_ratio**(k-1), pressure=new_pressure,
                              composition=self.composition, volume=new_volume, flow_rate=self.flow_rate)

    def calculate_mole_fractions(self):
        mole_fractions = {component: moles / self.total_moles for component, moles in self.composition.items()}
        return mole_fractions

def run_demo():
    demo_system = ChemicalSystem(temperature=300, pressure=(1e5), composition={'H2': 0.5, 'O2': 0.5}, volume=1,
                                 flow_rate=0.1)
    
    print("Partial Pressures:", demo_system.calculate_partial_pressures())
    expanded_iso = demo_system.isothermal_expansion(new_volume=2)
    print("\nIsothermal Expansion Partial Pressures:", expanded_iso.calculate_partial_pressures())

if __name__ == '__main__':
    run_demo()