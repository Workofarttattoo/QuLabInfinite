from dataclasses import dataclass, field
import numpy as np
from scipy.constants import k, Avogadro, g, c, h, e, pi
from typing import TypeVar, Generic

T = TypeVar('T', bound=np.ndarray)

@dataclass
class ThermodynamicsLab:
    temperature: float = 298.15
    pressure: float = field(default=101325., metadata={"units": "Pa"})
    volume: float = field(default=1.0, metadata={"units": "m^3"})
    mass: float = field(default=1.0, metadata={"units": "kg"})
    specific_heat_capacity: dict[str, float] = field(
        default_factory=lambda: {"Cp": 29.1, "Cv": 20.8}, metadata={"units": "J/(K mol)"}
    )
    gas_constant: float = k * Avogadro
    boltzmann_constant: float = k

    def __post_init__(self):
        self.internal_energy_change(self.volume)
        self.enthalpy_change()

    def internal_energy_change(self, volume: T) -> np.ndarray:
        return (self.mass / self.gas_constant) * volume

    def enthalpy_change(self) -> float:
        return self.temperature * self.pressure - self.gas_constant * self.temperature**2 / self.volume

    def entropy_change(self, heat_transfer: T) -> np.ndarray:
        return np.log(1 + (heat_transfer / (self.mass * self.boltzmann_constant)))

    def isothermal_compression_work(self, volume_initial: float, volume_final: float) -> float:
        return self.pressure * (volume_initial - volume_final)

    def adiabatic_expansion_work(self, pressure_initial: float, volume_initial: float, gamma: float = 1.4) -> float:
        return ((pressure_initial * volume_initial) / (gamma - 1)) * (1 - (volume_initial/self.volume)**(gamma-1))

    def specific_volume(self) -> np.ndarray:
        return self.mass / self.pressure

    def heat_capacity_ratio(self) -> float:
        return self.specific_heat_capacity["Cp"] / self.specific_heat_capacity["Cv"]

def run_demo():
    lab = ThermodynamicsLab()
    volume_change = np.array([0.5, 1.0], dtype=np.float64)
    print(f"Internal Energy Change: {lab.internal_energy_change(volume_change)}")
    print(f"Enthalpy Change: {lab.enthalpy_change()}")
    heat_transfer = np.array([-25, -30], dtype=np.float64)
    print(f"Entropy Change: {lab.entropy_change(heat_transfer)}")
    work_iso_compression = lab.isothermal_compression_work(volume_initial=1.5, volume_final=1.2)
    print(f"Isothermal Compression Work: {work_iso_compression}")
    work_adi_expansion = lab.adibatic_expansion_work(pressure_initial=90000., volume_initial=1.3)
    print(f"Adiabatic Expansion Work: {work_adi_expansion}")

if __name__ == '__main__':
    run_demo()