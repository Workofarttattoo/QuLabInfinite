import numpy as np
from dataclasses import dataclass, field
from scipy.constants import g, pi

@dataclass
class Rocket:
    mass: float = 0.0  # kg
    thrust: float = 0.0  # N
    drag_coefficient: float = 0.0  # dimensionless
    area: float = 0.0  # m^2
    initial_altitude: float = 0.0  # m
    initial_velocity: float = 0.0  # m/s

    def simulate_trajectory(self, time_steps: int) -> np.ndarray:
        """Simulates rocket trajectory."""
        dt = 1.0 / time_steps  # seconds
        t = np.linspace(0, 1, time_steps + 1, dtype=float)
        
        altitude = np.zeros_like(t, dtype=np.float64)
        velocity = np.zeros_like(t, dtype=np.float64)

        altitude[0] = self.initial_altitude
        velocity[0] = self.initial_velocity

        for i in range(1, time_steps + 1):
            acceleration_gravity = -g
            acceleration_thrust = self.thrust / self.mass
            drag_force = 0.5 * (1.225) * self.drag_coefficient * self.area * velocity[i-1]**2
            acceleration_drag = drag_force / self.mass

            net_acceleration = acceleration_gravity + acceleration_thrust + acceleration_drag
            velocity[i] = velocity[i - 1] + net_acceleration * dt
            altitude[i] = altitude[i - 1] + velocity[i - 1] * dt
        
        return np.column_stack((t, altitude, velocity))

@dataclass
class Propellant:
    name: str = ""
    specific_impulse: float = 0.0  # s
    density: float = 0.0  # kg/m^3

    def thrust(self, mass_flow_rate: float) -> float:
        """Calculates thrust given a specific mass flow rate."""
        return mass_flow_rate * self.specific_impulse * g

@dataclass
class Aerodynamics:
    name: str = ""
    drag_coefficient: float = 0.0

def run_demo():
    propellant = Propellant(name="RP-1", specific_impulse=285, density=960)
    aerodynamics = Aerodynamics(name="Nose Cone", drag_coefficient=0.3)
    
    rocket = Rocket(mass=700, thrust=propellant.thrust(1), drag_coefficient=aerodynamics.drag_coefficient,
                    area=np.pi * (0.5 ** 2), initial_velocity=0, initial_altitude=0)

    trajectory_data = rocket.simulate_trajectory(time_steps=100)
    
    print(f"Rocket Trajectory Simulation\n{'-'*48}")
    print("Time Steps: ", len(trajectory_data))
    for t in trajectory_data:
        print(f"t={t[0]:6.2f} s, Altitude={t[1]:7.2f} m, Velocity={t[2]:5.2f} m/s")

if __name__ == '__main__':
    run_demo()