"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

CARBON CAPTURE LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.constants import gas_constant
from typing import List, Dict, Any, Optional
from qulab.core.base_lab import BaseLab, register_lab

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

@register_lab(
    name="carbon_capture",
    category="chemistry",
    description="Carbon Capture and Adsorption Simulation Lab",
    version="1.0.0",
    tags=("carbon-capture", "environment", "adsorption")
)
class CarbonCaptureLab(BaseLab):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def run_experiment(self, experiment_spec: Dict[str, Any]) -> Dict[str, Any]:
        self._track_experiment()

        mixtures_data = experiment_spec.get("gas_mixtures", [])
        gas_mixtures = [GasMixture(**m) for m in mixtures_data]
        efficiency = experiment_spec.get("efficiency", 0.9)

        plant = CarbonCapturePlant(gas_mixtures, efficiency=efficiency)

        adsorbent_data = experiment_spec.get("adsorbent", {})
        adsorbent = AdsorbentMaterial(**adsorbent_data)

        column = AdsorptionColumn(
            length=experiment_spec.get("column_length", 2.0),
            diameter=experiment_spec.get("column_diameter", 0.3),
            adsorbents=[adsorbent]
        )

        captured_mass = plant.capture_co2()
        captured_volume = column.calculate_adsorption_volume(captured_mass)

        return {
            "total_co2_mass_kg": float(plant.total_co2_mass),
            "captured_co2_mass_kg": float(captured_mass),
            "captured_volume_m3": float(captured_volume),
            "efficiency": float(efficiency)
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "lab": "CarbonCaptureLab",
            "experiment_count": self._experiment_count,
            "uptime": self.uptime_seconds
        }

def run_demo():
    lab = CarbonCaptureLab()
    spec = {
        "gas_mixtures": [
            {'composition': {'CO2': 0.15, 'N2': 0.84}, 'temperature': 300},
            {'composition': {'CO2': 0.10, 'O2': 0.89}, 'temperature': 300}
        ],
        "adsorbent": {'name': 'Activated Carbon', 'capacity_co2': 0.15},
        "efficiency": 0.9
    }
    results = lab.run_experiment(spec)
    print(f"Results: {results}")

if __name__ == '__main__':
    run_demo()
