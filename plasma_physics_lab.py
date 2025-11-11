"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

PLASMA PHYSICS LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.constants import epsilon_0, mu_0, Boltzmann
import typing

@dataclass
class PlasmaParameters:
    electron_density: float = 1e20  # [m^-3]
    ion_density: float = 1e20      # [m^-3], assume singly charged ions for simplicity
    temperature_electron: float = 1e4     # [Kelvin]
    temperature_ion: float = 1e3         # [Kelvin], lower than electron temp typically
    magnetic_field_strength: float = 0.5 # [Tesla]
    plasma_frequency_ions: float = field(init=False)
    
    def __post_init__(self):
        self.plasma_frequency_ions = np.sqrt(self.ion_density * (mu_0**2 / (epsilon_0 * Boltzmann * self.temperature_electron)))

@dataclass
class PlasmaWave:
    plasma_params: PlasmaParameters
    wave_number: float  # [m^-1]
    frequency: float = field(init=False)

    def __post_init__(self):
        omega_pe = np.sqrt(self.plasma_params.ion_density / (epsilon_0 * self.plasma_params.temperature_electron))
        self.frequency = omega_pe * np.sqrt(1 - ((mu_0*self.plasma_params.magnetic_field_strength)/(4*omega_pe**2))**2)

@dataclass
class PlasmaSimulation:
    plasma_wave: PlasmaWave
    time_steps: int  # number of simulation steps, integer
    time_step_size: float  # [s]
    
    def simulate(self):
        times = np.linspace(0.0, self.time_steps * self.time_step_size, num=self.time_steps + 1, dtype=np.float64)
        frequencies = np.array([self.plasma_wave.frequency for _ in range(len(times))], dtype=np.float64)

        return times, frequencies

def run_demo():
    params = PlasmaParameters()
    wave = PlasmaWave(plasma_params=params, wave_number=0.2)
    
    sim = PlasmaSimulation(plasma_wave=wave, time_steps=100, time_step_size=5e-9)
    times, freqs = sim.simulate()

    print(f"Times: {times}")
    print(f"Frequencies: {freqs}")

if __name__ == '__main__':
    run_demo()