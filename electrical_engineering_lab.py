"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ELECTRICAL ENGINEERING LAB
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.constants import k, Avogadro, g, c, h, e, pi
from typing import List

# Constants and configuration
VOLTAGE_RANGES = [1.0, 5.0, 12.0, 24.0]
CURRENT_RANGES = [0.1, 1.0, 5.0, 10.0]

@dataclass
class CircuitElement:
    name: str
    voltage_rating: float
    current_rating: float

@dataclass
class Resistor(CircuitElement):
    resistance: float
    power_rating: float = field(default=2.5)

@dataclass
class Capacitor(CircuitElement):
    capacitance: float
    energy_storage: float = field(default=0.1)

@dataclass
class Inductor(CircuitElement):
    inductance: float

@dataclass
class VoltageSource:
    voltage: float
    internal_resistance: float = field(default=0.0)

@dataclass
class CurrentSource:
    current: float
    internal_inductance: float = field(default=1e-3)

@dataclass
class Battery(VoltageSource):
    capacity: float

@dataclass
class Transformer(CircuitElement):
    turns_ratio: float
    primary_voltage: float

@dataclass
class Diode:
    forward_voltage_drop: float = field(default=0.7)
    reverse_leakage_current: float = field(default=1e-9)

@dataclass
class Transistor:
    hfe: float  # Current gain
    saturation_voltage: float = field(default=0.2)
    cutoff_current: float = field(default=5e-6)

def calculate_resistance(voltage, current):
    return voltage / current

def calculate_power(resistance, current):
    return np.square(current) * resistance

@dataclass
class ElectricalLab:
    components: List[CircuitElement] = field(default_factory=list)
    
    def add_component(self, component):
        self.components.append(component)

    def simulate_circuit(self):
        total_voltage_drop = 0.0
        total_power_consumption = np.zeros_like(VOLTAGE_RANGES, dtype=np.float64)
        
        for component in self.components:
            if isinstance(component, Resistor):
                voltage_drop = calculate_resistance(component.resistance, component.current_rating)
                power_consumption = calculate_power(component.resistance, component.current_rating)
                
                total_voltage_drop += voltage_drop
                total_power_consumption += power_consumption
            
            elif isinstance(component, Capacitor):
                charge_time = 1.0 / (2 * pi * np.sqrt(component.capacitance * component.inductance))
            
            elif isinstance(component, Inductor):
                energy_storage = 0.5 * (component.current_rating ** 2) * component.inductance
            
            elif isinstance(component, VoltageSource):
                total_voltage_drop += component.voltage - calculate_resistance(component.internal_resistance, component.current)
            
            elif isinstance(component, CurrentSource):
                voltage_drop = calculate_resistance(component.internal_inductance, component.current)
                
        print(f"Total voltage drop: {total_voltage_drop}")
        print(f"Total power consumption: {total_power_consumption}")

def run_demo():
    lab = ElectricalLab()

    resistor = Resistor("R1", 5.0, 2.0, resistance=1000, power_rating=5)
    capacitor = Capacitor("C1", 5.0, 1.0, capacitance=1e-6, energy_storage=0.1)
    inductor = Inductor("L1", 5.0, 2.0, inductance=1e-3)
    voltage_source = VoltageSource(12.0, internal_resistance=0.01)
    current_source = CurrentSource(1.0, internal_inductance=1e-4)

    lab.add_component(resistor)
    lab.add_component(capacitor)
    lab.add_component(inductor)
    lab.add_component(voltage_source)
    lab.add_component(current_source)

    lab.simulate_circuit()

if __name__ == '__main__':
    run_demo()