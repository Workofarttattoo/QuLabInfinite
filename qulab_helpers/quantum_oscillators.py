"""Quantum harmonic oscillator helper utilities."""

from dataclasses import dataclass
from typing import Dict
from scipy.constants import hbar, e, pi


@dataclass
class QuantumHarmonicOscillator:
    """Analytical harmonic oscillator used for quick validations."""

    mass_kg: float
    frequency_hz: float
    label: str = "quantum_oscillator"

    @property
    def angular_frequency(self) -> float:
        return 2.0 * pi * self.frequency_hz

    def zero_point_energy_joules(self) -> float:
        return 0.5 * hbar * self.angular_frequency

    def zero_point_energy_ev(self) -> float:
        return self.zero_point_energy_joules() / e

    def describe(self) -> Dict[str, float]:
        return {
            "label": self.label,
            "mass_kg": self.mass_kg,
            "frequency_hz": self.frequency_hz,
            "zero_point_energy_j": self.zero_point_energy_joules(),
            "zero_point_energy_eV": self.zero_point_energy_ev(),
        }
