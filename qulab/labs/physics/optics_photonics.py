"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

OPTICS AND PHOTONICS LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
import scipy.constants as const

@dataclass
class OpticsLab:
    wavelength: float = 589.3e-9  # Sodium D-line
    refractive_index: float = 1.0
    material_absorption_coefficient: float = 0.0
    light_intensity: float = 1.0
    angle_of_incidence: float = 0.0

    def __post_init__(self):
        self.angle_of_incidence_rad = np.radians(self.angle.of.incidence)

    @property
    def wavelength_nm(self) -> np.ndarray:
        return np.array([self.wavelength * 1e9], dtype=np.float64)

    @property
    def refractive_index_array(self) -> np.ndarray:
        return np.full_like(self.wavelength_nm, self.refractive_index, dtype=np.float64)

    @property
    def material_absorption_coefficient_array(self) -> np.ndarray:
        return np.full_like(self.wavelength_nm, self.material_absorption_coefficient, dtype=np.float64)

    @property
    def light_intensity_array(self) -> np.ndarray:
        return np.array([self.light_intensity], dtype=np.float64)

    def angle_of_refraction(self) -> np.ndarray:
        return np.arcsin(np.sin(self.angle_of_incidence_rad) / self.refractive_index_array)

    def fresnel_coefficients(self, mode: str = 's') -> np.ndarray:
        if mode == 's':
            numerator = (self.refractive_index - 1j * self.material_absorption_coefficient) ** 2 \
                        * np.cos(self.angle_of_refraction()) - self.refractive_index_array
            denominator = (self.refractive_index + 1j * self.material_absorption_coefficient) ** 2 \
                          * np.cos(self.angle_of_incidence_rad) + self.refractive_index_array

        elif mode == 'p':
            numerator = (self.refractive_index - 1j * self.material_absorption_coefficient) ** 2 \
                        * np.cos(self.angle_of_refraction()) - self.refractive_index
            denominator = (self.refractive_index + 1j * self.material_absorption_coefficient) ** 2 \
                          * np.cos(self.angle_of_incidence_rad) + self.refractive_index_array

        else:
            raise ValueError("Mode must be 's' or 'p'.")
        
        return numerator / denominator
    
    def reflectance(self, mode: str = 's') -> np.ndarray:
        fresnel_coefficient = self.fresnel_coefficients(mode)
        return np.abs(fresnel_coefficient) ** 2

def run_demo() -> None:
    lab = OpticsLab(wavelength=589.3e-9, refractive_index=1.5, material_absorption_coefficient=0.01,
                    light_intensity=1.0, angle_of_incidence=30)
    
    r_s = lab.reflectance(mode='s')
    print(f"Reflectance (mode 'S'): {r_s[0]:.6f}")
    
    r_p = lab.reflectance(mode='p')
    print(f"Reflectance (mode 'P'): {r_p[0]:.6f}")

if __name__ == '__main__':
    run_demo()