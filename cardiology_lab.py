"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

CARDIOLOGY LAB
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.constants import k, Avogadro, g, c, h, e, pi, physical_constants

@dataclass
class CardiologyLab:
    heart_rate: float = 70.0
    systolic_bp: float = 120.0
    diastolic_bp: float = 80.0
    body_mass_index: float = 23.5
    age: int = 40
    height_cm: float = 175.0
    weight_kg: float = 70.0

    def __post_init__(self):
        self._bmi = None
        self._heart_beats_per_minute = np.array([self.heart_rate], dtype=np.float64)
        self._blood_pressure = np.array([self.systolic_bp, self.diastolic_bp], dtype=np.float64)
        
    @property
    def bmi(self) -> float:
        if not self._bmi:
            self._bmi = self.weight_kg / (self.height_cm/100)**2
        return self._bmi
    
    @property
    def heart_beats_per_minute(self) -> np.ndarray:
        return self._heart_beats_per_minute
    
    @property
    def blood_pressure(self) -> np.ndarray:
        return self._blood_pressure

    def calculate_blood_volume(self) -> float:
        """Estimate total blood volume from body mass index."""
        bmi_adjustment = 0.07 if self.bmi > 25 else -0.03
        estimated_weight_in_liters = (self.weight_kg * 0.08 + bmi_adjustment)
        return estimated_weight_in_liters
    
    def cardiac_output(self) -> float:
        """Calculate cardiac output using the Fick principle."""
        cohr = self.heart_beats_per_minute[0] / 60
        stroke_volume = (self.calculate_blood_volume() * pi / (cohr * Avogadro)) ** (1/3)
        return cohr * stroke_volume
    
    def pressure_gradient(self) -> np.ndarray:
        """Calculate pressure gradient between systolic and diastolic blood pressures."""
        return self.blood_pressure[0] - self.blood_pressure[1]
    
    def electrocardiogram(self, time: float = 1.0, frequency: int = 500):
        """Simulate an ECG signal over a period of time with given sampling rate."""
        t = np.linspace(0, time, frequency * time)
        ecg_signal = (t**2 + np.sin(t*frequency*pi) - t*np.cos(t))
        return ecg_signal
    
    def arterial_wall_stress(self):
        """Calculate the stress on the arterial wall using Laplace's law."""
        delta_p = self.pressure_gradient()
        r_aortic_root = 0.6 # cm
        return (delta_p * r_aortic_root) / np.array([1e-4, -1e-4], dtype=np.float64)
    
    def pulse_wave_velocity(self):
        """Calculate the velocity of a pulse wave in arteries."""
        arterial_elasticity = 0.3 # example value for elasticity
        blood_density = 1050 # kg/m^3
        return np.sqrt((self.pressure_gradient() / (2 * pi * self.height_cm)) 
                        + (pi**2 * Avogadro * e**2) / ((h*g)*blood_density))

@dataclass
class CardiologyLabDemo:
    lab: CardiologyLab = field(default_factory=CardiologyLab)

def run_demo():
    demo_lab = CardiologyLabDemo()
    print(f"Blood Pressure (Systolic, Diastolic): {demo_lab.lab.blood_pressure}")
    print(f"Heart Rate (beats per minute): {demo_lab.lab.heart_beats_per_minute[0]}")
    print(f"BMI: {demo_lab.lab.bmi:.2f}")
    
    blood_volume = demo_lab.lab.calculate_blood_volume()
    cardiac_output = demo_lab.lab.cardiac_output()
    stress = demo_lab.lab.arterial_wall_stress()
    pulse_velocity = demo_lab.lab.pulse_wave_velocity()

    print(f"Estimated Blood Volume: {blood_volume} liters")
    print(f"Heart's Cardiac Output: {cardiac_output:.2f} L/min")
    print(f"Arterial Wall Stress (Laplace): {stress[0]:.2e}, {-stress[1]:.2e}")
    print(f"Pulse Wave Velocity: {pulse_velocity[0]:.2f} m/s, {-pulse_velocity[1]:.2f} m/s")
    
if __name__ == '__main__':
    run_demo()