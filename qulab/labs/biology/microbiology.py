"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

MICROBIOLOGY LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.constants import Boltzmann, Avogadro

@dataclass
class MicrobiologyExperiment:
    temperature: float = 37.0  # Default human body temp in Celsius
    volume_ml: float = 100.0   # Volume of medium in mL
    initial_density_cfuml: float = 1e6  # Initial bacterial density (cfu/mL)
    
    def __post_init__(self):
        self.temperature_kelvin = self.temperature + 273.15

    @property
    def time_steps(self) -> int:
        return 100
    
    @property
    def growth_rate(self) -> float:
        # Bacterial doubling time in hours (E. coli: ~20 minutes)
        doubling_time_hours = 0.33
        return np.log(2) / (doubling_time_hours * 3600)
    
    def bacterial_growth_simulation(self):
        initial_bacteria_count = self.initial_density_cfuml * self.volume_ml
        growth_over_time = np.zeros((self.time_steps, ), dtype=np.float64)

        for t in range(1, self.time_steps + 1):
            growth_rate_t = self.growth_rate / (1 + np.exp(-Boltzmann * self.temperature_kelvin))
            bacteria_count_t = initial_bacteria_count * np.exp(growth_rate_t * t)
            growth_over_time[t - 1] = bacteria_count_t
            
        return growth_over_time
    
    def cell_volume_calculation(self, bacteria_density: float):
        # Assuming spherical shape and a typical radius for E. coli (0.5 um)
        cell_radius_um = 0.5
        volume_of_one_cell_ul = (4/3) * np.pi * (cell_radius_um / 1000)**3
        
        return bacteria_density * volume_of_one_cell_ul
    
    def calculate_concentration(self, time_index: int):
        bacteria_count_t = self.bacterial_growth_simulation()[time_index]
        
        if bacteria_count_t > 0:
            concentration_mgml = bacteria_count_t / (self.volume_ml) * self.cell_volume_calculation(bacteria_density=self.initial_density_cfuml)
        else:
            concentration_mgml = np.nan
        
        return concentration_mgml
    
    def simulate_phenotypic_variation(self, mutation_rate: float):
        # Simplified simulation of genetic variation
        mutated_population_ratio = (1 + mutation_rate)**self.time_steps
        total_bacteria_count = self.bacterial_growth_simulation()[-1]
        
        mutated_population = total_bacteria_count / mutated_population_ratio
        
        return mutated_population

@dataclass
class CultureMedium:
    volume_ml: float = 100.0
    ph_initial: float = 7.2
    salinity_ppm: int = 350
    
    def __post_init__(self):
        self.ph_over_time = np.zeros((MicrobiologyExperiment().time_steps, ), dtype=np.float64)
    
    def simulate_ph_change(self, bacteria_count: float, ph_consumption_rate_per_bacteria: float=1e-9, ph_reproduction_rate_per_ml: float=2.5e-8):
        self.ph_over_time[0] = self.ph_initial
        
        for t in range(1, len(self.ph_over_time)):
            # Simulate PH change due to bacterial activity
            ph_change_due_to_bacteria = -bacteria_count[t-1] * ph_consumption_rate_per_bacteria
            ph_change_due_to_growth = bacteria_count[t-1] / self.volume_ml * ph_reproduction_rate_per_ml
            
            self.ph_over_time[t] = max(min(self.ph_initial + ph_change_due_to_bacteria + ph_change_due_to_growth, 14), 0)
        
        return self.ph_over_time

def run_demo():
    experiment = MicrobiologyExperiment()
    
    # Simulate bacterial growth
    bacteria_count_over_time = experiment.bacterial_growth_simulation()
    print(f"Bacterial counts over time (cfu/mL): {bacteria_count_over_time}")
    
    # Calculate concentrations at each point
    concentration_over_time = np.array([experiment.calculate_concentration(t) for t in range(experiment.time_steps)])
    print(f"Concentrations over time (mg/mL): {concentration_over_time}")
    
    medium = CultureMedium(volume_ml=100.0)
    
    # Simulate PH change
    ph_changes_over_time = medium.simulate_ph_change(bacteria_count=bacteria_count_over_time)
    print(f"PH changes over time: {ph_changes_over_time}")

if __name__ == '__main__':
    run_demo()