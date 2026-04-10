"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

IMMUNOLOGY LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.constants import pi, N_A  # Avogadro's number

@dataclass
class ImmuneCell:
    cell_type: str
    surface_area: float = 0.0  # in square micrometers
    volume: float = 0.0  # in cubic micrometers
    concentration_receptors: int = 0  # receptors per cell

@dataclass
class Pathogen:
    type: str
    size_diameter: float = 0.1  # average diameter of pathogens, in micrometers
    surface_charge: float = 0.0  # charge on the surface of pathogen
    concentration: int = 1  # number of pathogens per unit volume

@dataclass
class ImmuneResponse:
    immune_cells: list[ImmuneCell] = field(default_factory=list)
    pathogens: Pathogen = None
    duration_minutes: float = 60.0  # in minutes
    simulation_timestep_seconds: float = 1.0  # in seconds

    def initialize_simulation(self):
        self.time_array_seconds = np.arange(0, self.duration_minutes * 60 + 1, self.simulation_timestep_seconds)

    def compute_contact_rate(self) -> float:
        surface_area_cells_total = sum(cell.surface_area for cell in self.immune_cells)
        volume_pathogens = (self.pathogens.size_diameter ** 3) / 6 * np.pi
        concentration_pathogens_per_volume = N_A * self.pathogens.concentration * (1e-24)**3
        return surface_area_cells_total * concentration_pathogens_per_volume

    def simulate_immune_response(self):
        contact_rate = self.compute_contact_rate()
        self.contact_event_probabilities = np.random.rand(len(self.time_array_seconds))
        self.contact_events_detected = [prob < contact_rate for prob in self.contact_event_probabilities]

    def run_simulation(self) -> None:
        self.initialize_simulation()
        self.simulate_immune_response()

def run_demo():
    cell1 = ImmuneCell(cell_type='T Cell', surface_area=80, volume=5432.0)
    pathogen1 = Pathogen(type="Virus", size_diameter=0.1, concentration=1e7)

    immune_response = ImmuneResponse(immune_cells=[cell1], pathogens=pathogen1, duration_minutes=60.0,
                                      simulation_timestep_seconds=5.0)
    
    immune_response.run_simulation()
    print(f"Total contacts detected: {sum(immune_response.contact_events_detected)}")

if __name__ == '__main__':
    run_demo()