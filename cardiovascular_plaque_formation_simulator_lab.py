"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

CARDIOVASCULAR PLAQUE FORMATION SIMULATOR
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List
from scipy.constants import pi

# Constants and configuration
@dataclass
class PlaqueFormationConfig:
    sim_duration: float = 10.0  # Duration of simulation in years
    dt: float = 0.01            # Time step in years
    initial_plaque_size: float = 0.0  # Initial size of plaque in mm^2
    max_plaque_growth_rate: float = 0.1  # Maximum growth rate in mm^2/year
    avg_cholesterol_level: float = 5  # Average cholesterol level in mg/dL
    cholesterol_deposit_rate: float = 0.03  # Cholesterol deposit rate in mm^2/mg/dL/year
    inflammation_factor: float = 1.2  # Inflammation factor affecting growth

@dataclass
class PlaqueFormationResult:
    time_points: List[float] = field(default_factory=list)
    plaque_sizes: List[float] = field(default_factory=list)

class CardiovascularPlaqueSimulator:
    def __init__(self, config: PlaqueFormationConfig):
        self.config = config
        self.results = PlaqueFormationResult()
        
    def simulate(self):
        t = 0
        plaque_size = self.config.initial_plaque_size
        
        while t <= self.config.sim_duration:
            growth_rate = min(plaque_size * self.config.inflammation_factor, self.config.max_plaque_growth_rate)
            
            # Calculate cholesterol deposit based on average cholesterol level and deposit rate
            cholesterol_deposit = self.config.cholesterol_level(t) * self.config.cholesterol_deposit_rate
            
            # Update plaque size
            plaque_size += growth_rate + cholesterol_deposit
            
            # Append results to time_points and plaque_sizes lists
            self.results.time_points.append(t)
            self.results.plaque_sizes.append(plaque_size)
            
            t += self.config.dt
        
    def cholesterol_level(self, t):
        return self.config.avg_cholesterol_level * (1.0 + 0.2 * np.sin(2 * pi * t / self.config.sim_duration))
        
    def get_results(self) -> PlaqueFormationResult:
        return self.results

def run_demo():
    config = PlaqueFormationConfig(
        sim_duration=5,
        dt=0.01,
        initial_plaque_size=0.1,
        max_plaque_growth_rate=0.2,
        avg_cholesterol_level=6,
        cholesterol_deposit_rate=0.04
    )
    
    simulator = CardiovascularPlaqueSimulator(config)
    simulator.simulate()
    
    results = simulator.get_results()
    print(f"Time points: {results.time_points}")
    print(f"Plaque sizes (mm^2): {results.plaque_sizes}")

if __name__ == '__main__':
    run_demo()