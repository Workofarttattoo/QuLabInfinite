"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

CONTROL SYSTEMS LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List

# Constants and configuration
kB = 1.380649e-23  # Boltzmann constant in J/K (scipy.constants.k)
N_A = 6.02214076e23  # Avogadro's number (scipy.constants.Avogadro)
g = 9.80665  # Gravity in m/s^2 (scipy.constants.g)
c = 299792458.0  # Speed of light in vacuum in m/s (scipy.constants.c)
h = 6.62607015e-34  # Planck constant in J*s (scipy.constants.h)
e = 1.60217663e-19  # Elementary charge in C (scipy.constants.e)

@dataclass
class TransferFunction:
    num: np.ndarray
    den: np.ndarray

@dataclass
class ControlSystem:
    name: str
    transfer_function: TransferFunction
    system_matrix: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    input_vector: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    output_vector: np.ndarray = field(init=False)

    def __post_init__(self):
        self.output_vector = np.zeros_like(self.input_vector)  # Initialize with zeros

    @staticmethod
    def compute_response(system_matrix, input_vector, time_points):
        return np.dot(system_matrix, input_vector)

@dataclass
class PIController:
    Kp: float
    Ki: float

    def calculate_output(self, error_sum, current_error, dt):
        return self.Kp * current_error + self.Ki * dt * error_sum


def run_demo():
    # Example control system setup
    tf = TransferFunction(np.array([1.0], dtype=np.float64), np.array([1.0, 2.0, 3.0], dtype=np.float64))
    system_matrix = np.array([[1.0, 0.5]], dtype=np.float64)
    input_vector = np.array([1.0, -1.0], dtype=np.float64)

    control_system = ControlSystem("Example System", tf, system_matrix, input_vector)
    
    # Compute response
    time_points = np.linspace(0, 10, num=100)  # Time points for response computation
    output_response = ControlSystem.compute_response(control_system.system_matrix, control_system.input_vector, time_points)
    print(f"Output Response: {output_response}")

    # PI Controller Example
    pi_controller = PIController(Kp=2.5, Ki=0.1)
    error_sum = 0.0
    current_error = -1.0
    dt = 0.1  # Time step for discrete control
    
    output = pi_controller.calculate_output(error_sum, current_error, dt)
    print(f"Controller Output: {output}")

if __name__ == '__main__':
    run_demo()