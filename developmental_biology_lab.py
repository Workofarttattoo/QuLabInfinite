"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

DEVELOPMENTAL BIOLOGY LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.constants import pi, Boltzmann

@dataclass
class DevelopmentalBiology:
    cell_volume: float  # in cubic micrometers (um^3)
    temperature: float = 310.15  # in Kelvin (K), default to 37°C body temp
    diffusion_coefficient: float = 2e-11  # in square meters per second (m^2/s)
    
    def __post_init__(self):
        self.cell_radius = ((3 * self.cell_volume) / (4 * pi)) ** (1/3)

    def concentration_profile(self, distance_from_nucleus: float, time_elapsed: float) -> np.ndarray:
        """Simulates diffusion of a substance within the cell over time."""
        assert distance_from_nucleus <= self.cell_radius

        # Convert um to meters
        radius_meters = self.cell_radius * 1e-6
        distance_from_nucleus_meters = distance_from_nucleus * 1e-6
        
        # Calculate concentration profile using Fick's second law (simplified)
        D = self.diffusion_coefficient
        t = time_elapsed
        r_squared_over_4Dt = (distance_from_nucleus_meters**2) / (4 * D * t)

        # Concentration is zero at the surface of the sphere for simplicity
        concentration_profile = np.exp(-r_squared_over_4Dt)
        
        return concentration_profile
    
    def osmotic_pressure(self, solute_concentration: float) -> float:
        """Calculates osmotic pressure using Van't Hoff's equation."""
        R = Boltzmann * 1e23  # convert to J/(mol·K) for consistency with scipy.constants
        V_psi = (R * self.temperature * solute_concentration)
        
        return V_psi
    
    def cell_division_time(self, growth_rate: float) -> float:
        """Estimates time required for a single cell division."""
        # Growth rate is in um^3/min
        volume_increased_per_minute = growth_rate
        
        # Time to double the volume (assuming exponential growth)
        original_volume = self.cell_volume
        target_volume = 2 * original_volume

        time_for_division = np.log(target_volume / original_volume) / np.log(2) / (volume_increased_per_minute / original_volume)
        
        return time_for_division
    
    def cell_cycle_phases(self, phase_durations: dict) -> None:
        """Simulate the phases of a cell cycle."""
        phases = ['G1', 'S', 'G2', 'M']
        total_duration = sum(phase_durations.values())
        
        print(f"Cell Cycle Simulation at {self.temperature}K")
        for phase in phases:
            duration = phase_durations.get(phase, 0)
            if duration == 0:
                continue
            print(f"{phase}: Duration: {duration:.2f} minutes (Proportion of cell cycle: {100*duration/total_duration:.2f}%)")


def run_demo():
    db_lab = DevelopmentalBiology(cell_volume=4500)  # in cubic micrometers
    print("Cell Radius:", db_lab.cell_radius, "um")
    
    concentration_profile = db_lab.concentration_profile(distance_from_nucleus=10, time_elapsed=30)
    print(f"Concentration profile at 10 um from nucleus after 30 minutes: {concentration_profile:.4f}")
    
    osmotic_pressure_value = db_lab.osmotic_pressure(solute_concentration=2)  # units are in mol/L (M), hypothetical concentration
    print(f"Osmotic pressure with solute concentration of 2 M at {db_lab.temperature}K: {osmotic_pressure_value:.4f} Pa")
    
    division_time = db_lab.cell_division_time(growth_rate=10)  # volume increase rate in cubic micrometers per minute
    print(f"Estimated time for cell division with growth rate of 10 um^3/min: {division_time:.2f} minutes")

    phases_and_durations = {'G1': 90, 'S': 60, 'G2': 50, 'M': 40}
    db_lab.cell_cycle_phases(phases_and_durations)

if __name__ == '__main__':
    run_demo()