"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

CATALYSIS LAB
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.constants import k, Avogadro, g, c, h, e, pi
from typing import Any

# Constants and configuration
K_BOLTS = k
AVOGADRO = Avogadro
GRAVITY = g
SPEED_OF_LIGHT = c
PLANCK_CONSTANT = h
ELEMENTARY_CHARGE = e

@dataclass
class CatalysisLab:
    temperature: float = field(default=298.15, metadata={"description": "Temperature in Kelvin"})
    pressure: float = field(default=101325, metadata={"description": "Pressure in Pa"})
    volume: np.ndarray = field(default_factory=lambda: np.array([1.0], dtype=np.float64), metadata={"description": "Volume of system in m^3"})
    concentration: np.ndarray = field(default_factory=lambda: np.array([1e-3, 2e-3]), metadata={"description": "Concentration of reactants in mol/m^3"})
    rate_constants: dict[str, float] = field(default_factory=lambda: {"k1": 0.05, "k2": 0.1}, metadata={"description": "Rate constants for reactions"})
    
    def __post_init__(self):
        self.pressure /= AVOGADRO
        self.volume = np.array(self.volume, dtype=np.float64)
        self.concentration = np.array(self.concentration, dtype=np.float64)

    def calculate_chemical_potential(self) -> np.ndarray:
        return K_BOLTS * self.temperature * np.log(self.concentration / (self.pressure / AVOGADRO))

    def calculate_reaction_rate(self) -> np.ndarray:
        return self.rate_constants["k1"] * np.sum(np.prod(self.concentration ** -1, axis=0)) + self.rate_constants["k2"]

    def simulate_catalysis_process(self, time_steps: int = 1000, dt: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0, time_steps * dt, time_steps)
        concentration_t = np.zeros((time_steps,) + self.concentration.shape, dtype=np.float64)

        for i in range(time_steps):
            reaction_rate = self.calculate_reaction_rate()
            d_conc_dt = -reaction_rate
            concentration_t[i] = self.concentration + d_conc_dt * dt

        return t, concentration_t
    
    def print_summary(self) -> None:
        chemical_potential = self.calculate_chemical_potential()
        reaction_rate = self.calculate_reaction_rate()

        print(f"Temperature: {self.temperature} K")
        print(f"Pressure (scaled): {self.pressure / AVOGADRO:.2e}")
        print(f"Volume: {self.volume} m^3")
        print(f"Initial Concentration: {self.concentration} mol/m^3")
        print(f"Rate constants: k1={self.rate_constants['k1']}, k2={self.rate_constants['k2']} s^-1")
        print(f"Chemical Potential: {chemical_potential:.4e}")
        print(f"Reaction Rate: {reaction_rate:.6f}")

def run_demo() -> None:
    lab = CatalysisLab()
    lab.print_summary()

    t, concentration_t = lab.simulate_catalysis_process(time_steps=10, dt=0.05)

    import matplotlib.pyplot as plt
    for i in range(lab.concentration.shape[0]):
        plt.plot(t, concentration_t[:, i], label=f"Concentration of Species {i}")
    
    plt.xlabel(r'Time (s)')
    plt.ylabel(r'Concentration ($mol/m^3$)')
    plt.title('Simulated Catalysis Process')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    run_demo()