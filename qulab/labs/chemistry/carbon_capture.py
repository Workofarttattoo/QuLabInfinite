"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

CARBON CAPTURE LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.constants import gas_constant
from typing import List

@dataclass
class GasMixture:
    composition: dict  # Composition of gases in mole fractions (e.g., {"CO2": 0.15, "N2": 0.84})
    temperature: float = 298.15  # K - default room temperature
    pressure: float = 101325.0  # Pa - standard atmospheric pressure

@dataclass
class CarbonCapturePlant:
    gas_mixtures: List[GasMixture]
    efficiency: float = field(default=0.9, metadata={'help': 'Fraction of CO2 captured'})

    def __post_init__(self):
        self.calculate_total_co2()

    def calculate_total_co2(self):
        total_co2_moles = np.sum([gm.composition['CO2'] * gm.pressure / (gas_constant * gm.temperature)
                                  for gm in self.gas_mixtures])
        self.total_co2_mass = total_co2_moles * gas_constant * self.gas_mixtures[0].temperature / 1e5

    def capture_co2(self):
        captured_co2_moles = self.efficiency * np.sum([gm.composition['CO2'] * gm.pressure / (gas_constant * gm.temperature)
                                                       for gm in self.gas_mixtures])
        return captured_co2_moles * gas_constant * self.gas_mixtures[0].temperature / 1e5

    def calculate_capture_cost(self, price_per_kg: float):
        """Calculate the cost of capturing CO2 based on efficiency and market prices."""
        total_captured_mass = self.capture_co2()
        return total_captured_mass * price_per_kg

@dataclass
class AdsorbentMaterial:
    name: str  # Name of adsorbent material (e.g., "Activated Carbon")
    capacity_co2: float  # Capacity for CO2 in kg/m^3
    density: float = field(default=0.5, metadata={'help': 'Density of the solid adsorbent'})

@dataclass
class AdsorptionColumn:
    length: float  # Length of column in meters
    diameter: float  # Diameter of column in meters
    adsorbents: List[AdsorbentMaterial]

    def calculate_adsorption_volume(self, total_mass_co2):
        volume = total_mass_co2 / np.array([ad.capacity_co2 * ad.density for ad in self.adsorbents]).sum()
        return volume

def run_demo():
    # Example setup
    gas_mixture_1 = GasMixture({'CO2': 0.15, 'N2': 0.84}, temperature=300)
    gas_mixture_2 = GasMixture({'CO2': 0.10, 'O2': 0.89}, temperature=300)

    plant = CarbonCapturePlant([gas_mixture_1, gas_mixture_2])
    print(f"Total CO2 Mass: {plant.total_co2_mass:.4f} kg")

    # Example adsorbent
    activated_carbon = AdsorbentMaterial('Activated Carbon', capacity_co2=0.15)
    
    column = AdsorptionColumn(length=2, diameter=0.3, adsorbents=[activated_carbon])
    captured_volume = column.calculate_adsorption_volume(plant.capture_co2())
    print(f"Captured CO2 volume: {captured_volume:.4f} m^3")

    # Cost example
    price_per_kg_co2 = 50  # Example market price per kg of CO2 capture cost
    total_cost = plant.calculate_capture_cost(price_per_kg=price_per_kg_co2)
    print(f"Total Capture Cost: {total_cost:.4f} €")

if __name__ == '__main__':
    run_demo()