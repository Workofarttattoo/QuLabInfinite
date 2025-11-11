"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ASTROPHYSICS LAB
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.constants import k, Avogadro, g, c, h, e, pi, physical_constants, G
from typing import *

@dataclass
class AstrophysicsConstants:
    boltzmann: float = k
    avogadro: float = Avogadro
    gravity: float = g
    speed_of_light: float = c
    planck_constant: float = h
    elementary_charge: float = e

@dataclass
class CelestialBody:
    mass: np.ndarray
    radius: np.ndarray
    temperature: np.ndarray
    luminosity: Optional[np.ndarray] = None
    
    @property
    def surface_gravity(self) -> np.ndarray:
        return (G * self.mass) / (self.radius**2)
    
    @property
    def density(self) -> np.ndarray:
        return self.mass / ((4/3) * pi * self.radius**3)
    
    @staticmethod
    def blackbody_luminosity(temperature: np.ndarray, radius: np.ndarray) -> np.ndarray:
        sigma = physical_constants['Stefan-Boltzmann constant'][0]
        luminosity = 4*pi*radius**2*sigma*temperature**4
        return luminosity
    
    @property
    def blackbody_luminosities(self) -> np.ndarray:
        if self.luminosity is None:
            self.luminosity = CelestialBody.blackbody_luminosity(self.temperature, self.radius)
        return self.luminosity

@dataclass
class Star(CelestialBody):
    age: Optional[np.ndarray] = None
    
    def main_sequence_lifetime(self) -> np.ndarray:
        if self.age is None or len(self.age) == 0:
            raise ValueError('Age must be provided for the star.')
        return (self.mass / self.luminosity)**(3/2)

@dataclass
class Galaxy:
    stars: List[CelestialBody] = field(default_factory=list)
    
    def total_mass(self) -> np.ndarray:
        masses = [star.mass for star in self.stars]
        return np.sum(masses, axis=0)
    
    def average_luminosity(self) -> np.ndarray:
        luminosities = [star.luminosity if hasattr(star, 'luminosity') else None for star in self.stars]
        non_null_luminosities = [lum for lum in luminosities if lum is not None]
        return np.mean(non_null_luminosities, axis=0) if len(non_null_luminosities) > 0 else np.zeros_like(masses)
    
    def __str__(self):
        total_mass = self.total_mass()
        avg_lum = self.average_luminosity()
        return f"Galaxy Summary:\n- Total Mass: {total_mass}\n- Average Luminosity: {avg_lum}"

constants = AstrophysicsConstants()

def run_demo():
    star1_mass = np.array([1.0, 2.0], dtype=np.float64)
    star1_radius = np.array([7e8, 9e8], dtype=np.float64)
    star1_temp = np.array([5772, 3500], dtype=np.float64)
    
    star1 = Star(mass=star1_mass, radius=star1_radius, temperature=star1_temp)

    galaxy = Galaxy(stars=[star1])

    print(f"Star 1: {star1}")
    print(f"Galaxy Summary:\n{galaxy}")

if __name__ == '__main__':
    run_demo()