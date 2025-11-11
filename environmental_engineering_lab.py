"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ENVIRONMENTAL ENGINEERING LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.constants import k, Avogadro, g, c, h, e, pi
from typing import List

@dataclass
class EnvironmentalEngineering:
    temperature: float = 298.15  # default value in Kelvin
    pressure: float = 101325     # default value in Pa
    humidity: float = 0.5        # relative humidity, fraction of max vapor capacity
    altitude: float = 0          # altitude above sea level in meters

    def __post_init__(self):
        self.water_vapor_pressure = self._calculate_water_vapor_pressure()
        self.saturated_vapor_pressure = self._calculate_saturated_vapor_pressure()

    @property
    def dew_point(self) -> float:
        return self._calculate_dew_point()

    @property
    def absolute_humidity(self) -> np.ndarray:
        return self._calculate_absolute_humidity()

    def _calculate_water_vapor_pressure(self) -> float:
        # Calculate water vapor pressure using the relative humidity and saturated vapor pressure at given temperature.
        saturated_vapor_press = self._calculate_saturated_vapor_pressure()
        return self.humidity * saturated_vapor_press

    def _calculate_saturated_vapor_pressure(self) -> float:
        # Antoine equation for water vapor pressure in Pa
        A, B, C = 8.07131, 1730.63, 233.426
        return np.exp(A - (B / (self.temperature + C)))

    def _calculate_dew_point(self) -> float:
        # Calculate dew point using iterations to solve for the temperature at which the vapor pressure equals the water vapor pressure.
        Tv = self.temperature  # Initial guess
        delta_Tv = 1           # Initialize change in temperature

        while abs(delta_Tv) > 1e-6:  # Stop when change is very small
            Tv_new = self._calculate_dew_point_temperature(Tv)
            delta_Tv = Tv - Tv_new
            Tv = Tv_new

        return Tv_new

    def _calculate_dew_point_temperature(self, Td_guess) -> float:
        # Antoine equation for vapor pressure at the guessed temperature.
        A, B, C = 8.07131, 1730.63, 233.426
        return A - (B / (Td_guess + C))

    def _calculate_absolute_humidity(self) -> np.ndarray:
        # Absolute humidity is the mass of water vapor per unit volume.
        molecular_weight_water = 18.015e-3  # kg/mol
        density_air = self._density_of_air()

        return (self.water_vapor_pressure * Avogadro * e) / (molecular_weight_water * density_air)

    def _density_of_air(self) -> float:
        # Ideal gas law for dry air with standard molar volume at 273.15 K and P = 100 kPa.
        R_dry_air = 8.314 / 29  # J/(K*mol), assuming average molecular weight of 29 g/mol
        return (self.pressure * Avogadro) / (R_dry_air * self.temperature)

    @staticmethod
    def _convert_to_celsius(K):
        return K - 273.15

def run_demo():
    ee = EnvironmentalEngineering(temperature=280, pressure=90000, humidity=0.6)
    print(f"Dew Point: {EnvironmentalEngineering._convert_to_celsius(ee.dew_point):.2f} °C")
    print(f"Absolute Humidity: {ee.absolute_humidity:.4e} kg/m³")

if __name__ == '__main__':
    run_demo()