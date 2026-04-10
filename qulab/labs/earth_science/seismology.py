"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

SEISMOLOGY LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.constants import g

@dataclass
class SeismologyLab:
    """
    A class for seismological simulations using NumPy.
    
    Attributes:
        grid_size: The size of the grid (N x N) where seismic waves propagate.
        wave_speed: Speed of wave propagation in m/s.
        time_step: Time step for numerical integration.
        duration: Total simulation duration.
        source_location: Location of the earthquake source on the grid.
        receiver_locations: Locations to record seismograms.
    """
    
    grid_size: int = 100
    wave_speed: float = 346.0  # Speed of sound in air (m/s)
    time_step: float = 0.01
    duration: float = 2.0
    source_location: tuple[int, int] = field(default_factory=lambda: (50, 50))
    receiver_locations: list[tuple[int, int]] = field(
        default_factory=lambda: [(49, 49), (51, 51)]
    )

    def __post_init__(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
        self.time_steps = int(self.duration / self.time_step)

    def generate_seismograms(self) -> None:
        """
        Simulate seismic waves and record seismograms at receiver locations.
        """
        for t in range(self.time_steps):
            if t % 100 == 0:  # Every 100th step
                self.generate_wave()
            for x, y in self.receiver_locations:
                print(f"Receiver ({x}, {y}) seismogram at time {t * self.time_step:.2f}s")

    def generate_wave(self) -> None:
        """
        Generate a wave pulse centered at the source location.
        """
        center_x, center_y = self.source_location
        dist_squared = (np.arange(self.grid_size)[:, np.newaxis] - center_x)**2 + \
                       (np.arange(self.grid_size)[np.newaxis, :] - center_y)**2
        wave_pulse = 1 / (1 + dist_squared)
        wave_pulse *= np.exp(-self.time_step * self.wave_speed**2)
        self.grid += wave_pulse

    def run_simulation(self) -> None:
        """
        Run the entire simulation and record seismograms.
        """
        print("Running Seismic Simulation...")
        self.generate_seismograms()

def run_demo() -> None:
    lab = SeismologyLab()
    lab.run_simulation()

if __name__ == '__main__':
    run_demo()