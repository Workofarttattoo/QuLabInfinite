"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

HYDROLOGY LAB
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.constants import k, Avogadro, g, c, h, e, pi

@dataclass
class Hydrology:
    constants: dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.constants['k'] = k
        self.constants['Avogadro'] = Avogadro
        self.constants['g'] = g
        
    def vapor_pressure(self, temperature: np.ndarray) -> np.ndarray:
        """Calculate vapor pressure using the Clausius-Clapeyron equation."""
        return 10.0**(7.5 - (2345 / (temperature + 273.15)))
    
    def relative_humidity(self, p_vapor: np.ndarray, p_saturation: np.ndarray) -> np.ndarray:
        """Calculate the relative humidity from vapor pressure and saturation."""
        return p_vapor / p_saturation
    
    def dew_point_temperature(self, p_vapor: np.ndarray) -> np.ndarray:
        """Dew point temperature calculation from vapor pressure."""
        return (2345 * 10**((np.log10(p_vapor) - 7.5))) / (-2345 + 10 **(np.log10(p_vapor)))
    
    def saturation_vapor_pressure(self, temperature: np.ndarray) -> np.ndarray:
        """Calculate the saturation vapor pressure at a given temperature."""
        return self.vapor_pressure(temperature)
    
    def evaporation_rate(self, wind_speed: np.ndarray, relative_humidity: np.ndarray,
                         air_temperature: np.ndarray, water_vapor_concentration: np.ndarray) -> np.ndarray:
        """Calculate the evaporation rate from surface to air."""
        # Simplified Hargreaves model
        return 0.216 * (air_temperature - dew_point_temperature(water_vapor_concentration)) ** (3/4) \
               * wind_speed
    
    def infiltration(self, rainfall: np.ndarray, soil_water_storage: np.ndarray,
                     saturation_deficit: np.ndarray) -> np.ndarray:
        """Calculate the infiltration capacity of the soil."""
        return np.minimum(rainfall - saturation_deficit, soil_water_storage)
    
    def runoff_depth(self, rainfall: np.ndarray, infiltration_rate: np.ndarray) -> np.ndarray:
        """Determine the depth of surface water runoff from an area."""
        return rainfall - infiltration_rate
    
def run_demo():
    hydrology = Hydrology()
    temperatures = np.array([0., 10., 25.], dtype=np.float64)
    wind_speeds = np.array([1., 3., 5.], dtype=np.float64)
    relative_humidities = np.array([0.5, 0.7, 0.9], dtype=np.float64)
    
    vapor_pressures = hydrology.vapor_pressure(temperatures)
    saturation_vaps = hydrology.saturation_vapor_pressure(temperatures)
    rel_hums = hydrology.relative_humidity(vapor_pressures, saturation_vaps)
    dew_point_temps = hydrology.dew_point_temperature(vapor_pressures)
    evap_rates = hydrology.evaporation_rate(wind_speeds, relative_humidities,
                                             temperatures, vapor_pressures)
    
    print("Vapor Pressures:", vapor_pressures)
    print("Saturation Vapor Pressures:", saturation_vaps)
    print("Relative Humidity:", rel_hums)
    print("Dew Point Temperatures:", dew_point_temps)
    print("Evaporation Rates:", evap_rates)

if __name__ == '__main__':
    run_demo()