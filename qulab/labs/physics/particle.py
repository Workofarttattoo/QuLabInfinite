"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

PARTICLE PHYSICS LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import TypeVar
import scipy.constants

T = TypeVar('T')

# Constants and configuration
K_BOLTS = scipy.constants.k  # Boltzmann constant (J/K)
AVOGADRO = scipy.constants.Avogadro  # Avogadro's number
GRAVITY = scipy.constants.g  # gravitational acceleration (m/s^2)
SPEED_LIGHT = scipy.constants.c  # speed of light in vacuum (m/s)
PLANCK = scipy.constants.h  # Planck constant (J*s)
ELEM_CHARGE = scipy.constants.e  # elementary charge (Coulomb)

@dataclass
class Particle:
    mass: float
    charge: int
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    acceleration: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.acceleration = np.zeros((3,), dtype=np.float64)

@dataclass
class ParticleSystem:
    particles: list[Particle]
    time_step: float
    
    def update(self, force_function):
        for particle in self.particles:
            force = force_function(particle)
            acceleration = force / (particle.mass * AVOGADRO)  # acceleration due to force
            particle.acceleration += acceleration
            
            velocity_change = particle.acceleration * self.time_step
            particle.velocity += velocity_change
            
            position_change = particle.velocity * self.time_step
            particle.position += position_change

def electric_force(particle):
    """Compute electric force on a particle."""
    field_strength = 1.0  # dummy value, replace with real computation
    direction_vector = np.array([1., 0., 0.])
    return ELEM_CHARGE * particle.charge * field_strength * direction_vector

def magnetic_force(particle):
    """Compute magnetic force on a particle."""
    B_field_strength = 1.0  # dummy value, replace with real computation
    direction_vector_B = np.array([0., 1., 0.])
    
    v_cross_B = np.cross(particle.velocity, direction_vector_B)
    return ELEM_CHARGE * particle.charge * (v_cross_B * B_field_strength)

def gravity_force(particle):
    """Compute gravitational force on a particle."""
    G = scipy.constants.G
    mass_planet = 1.0e24  # mass of Earth in kg (dummy value)
    
    direction_vector_G = -particle.position / np.linalg.norm(particle.position, ord=2)  # towards the center
    
    return GRAVITY * particle.mass * mass_planet * direction_vector_G

def run_demo():
    p1 = Particle(
        mass=9.1093837e-31,  # electron mass
        charge=-1,
        position=np.array([0., 0., 0.], dtype=np.float64),
        velocity=np.array([1.e5, 0., 0.], dtype=np.float64)
    )
    
    p2 = Particle(
        mass=1.6726219e-27,  # proton mass
        charge=1,
        position=np.array([0., 0., 0.], dtype=np.float64),
        velocity=np.array([-5.e3, 0., 0.], dtype=np.float64)
    )
    
    system = ParticleSystem(
        particles=[p1, p2],
        time_step=1e-9
    )
    
    for _ in range(10):  # run simulation steps
        force_function = electric_force  # change to magnetic_force or gravity_force as desired
        system.update(force_function)
        
        print(f"Positions: {system.particles[0].position}, {system.particles[1].position}")
    
if __name__ == '__main__':
    run_demo()