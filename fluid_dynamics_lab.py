"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

FLUID DYNAMICS LAB
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy import constants

@dataclass
class FluidDynamics:
    viscosity: float = 1e-3  # kg/(m*s)
    density_fluid: float = 1000.0  # kg/m^3 (water at room temp)
    gravity: float = constants.g
    height: float = field(default=1.0, metadata=dict(unit="m"))
    length_scale: float = field(default=1e-2, metadata=dict(unit="m"))
    time_scale: float = field(default=np.sqrt(constants.pi * constants.e), metadata=dict(unit="s"))
    
    def __post_init__(self):
        self.reynolds_number = (self.density_fluid * self.length_scale / self.time_scale) * self.viscosity
        self.weber_number = (self.density_fluid * np.power(self.length_scale, 2) * self.gravity) / self.viscosity ** 2
    
    def calculate_velocity_profile(self, y: np.ndarray):
        u_max = constants.c / 100.0  # Arbitrary upper velocity limit for demonstration
        return u_max * (1 - np.square(y/self.height))
    
    def stream_function(self, x: np.ndarray, y: np.ndarray):
        return self.density_fluid * (self.length_scale ** 2) * constants.g / (2 * self.viscosity) * (y - self.height)
    
    def velocity_potential(self, x: np.ndarray, y: np.ndarray):
        u = self.calculate_velocity_profile(y)
        return u * x
    
    def pressure_differential(self, dpdx: float):
        """Calculates the pressure differential given a change in pressure over distance"""
        return -self.density_fluid * dpdx

@dataclass
class FluidChannel:
    fluid_dynamics: FluidDynamics = field(default_factory=FluidDynamics)
    
    def __post_init__(self):
        self.width_channel = 0.1  # m, assuming a typical channel width
    
    def calculate_flow_rate(self) -> float:
        u_max = constants.c / 1000.0
        return 0.5 * u_max * np.pi * self.fluid_dynamics.height ** 2 / self.width_channel
    
def run_demo():
    fd = FluidDynamics(density_fluid=873, height=0.5)
    
    y = np.linspace(0, fd.height, 100, dtype=np.float64)
    u_profile = fd.calculate_velocity_profile(y)

    x = np.array([0.1], dtype=np.float64)
    y = np.array([0.2, 0.3], dtype=np.float64)
    psi_values = fd.stream_function(x, y)
    phi_values = fd.velocity_potential(x, y)
    
    channel_flow = FluidChannel(fd)
    q_rate = channel_flow.calculate_flow_rate()
    
    print(f"Reynolds Number: {fd.reynolds_number:.3f}")
    print(f"Weber Number: {fd.weber_number:.3f}")
    print(f"Velocity Profile at Height 0.5m: {u_profile[-1]:.4f} m/s")
    print(f"Stream Function Values (psi): {psi_values}")
    print(f"Velocity Potential Values (phi): {phi_values}")
    print(f"Flow Rate through Channel: {q_rate:.3f} m^3/s")

if __name__ == '__main__':
    run_demo()