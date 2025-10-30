# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Temperature Control System
Range: -273.15°C to 10,000°C with ±0.001 K precision
"""

import numpy as np
from typing import Tuple, Optional, Callable
import threading


class TemperatureControl:
    """
    High-precision temperature control system with thermal gradients,
    radiative heating, and cryogenic capabilities.
    """

    # Physical constants
    ABSOLUTE_ZERO = -273.15  # °C
    STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)

    def __init__(self, precision: float = 0.001):
        """
        Initialize temperature control system.

        Args:
            precision: Temperature precision in Kelvin (default: 0.001 K)
        """
        self.precision = precision
        self._lock = threading.RLock()

        # Temperature state
        self._base_temperature = 298.15  # K (25°C)
        self._gradient_field = None  # 3D temperature gradient field
        self._heat_sources = []  # List of (position, power) tuples
        self._heat_sinks = []  # List of (position, power) tuples
        self._radiative_enabled = False
        self._radiation_temperature = 5778  # K (solar equivalent)

        # Time-dependent heating profiles
        self._heating_profile = None  # Callable: time -> temperature

    def set_temperature(self, temperature: float, unit: str = "C") -> None:
        """
        Set uniform base temperature.

        Args:
            temperature: Temperature value
            unit: Temperature unit ("C", "K", "F")

        Raises:
            ValueError: If temperature is below absolute zero
        """
        with self._lock:
            temp_k = self._convert_to_kelvin(temperature, unit)

            if temp_k < 0:
                raise ValueError(f"Temperature {temp_k} K is below absolute zero")

            if temp_k > 10273.15:  # 10,000°C
                raise ValueError(f"Temperature {temp_k} K exceeds maximum 10,273.15 K")

            # Quantize to precision
            self._base_temperature = self._quantize(temp_k)

    def get_temperature(self, position: Optional[Tuple[float, float, float]] = None,
                       unit: str = "C") -> float:
        """
        Get temperature at a specific position or base temperature.

        Args:
            position: (x, y, z) coordinates in meters (None for base temperature)
            unit: Temperature unit for return value

        Returns:
            Temperature at position with specified precision
        """
        with self._lock:
            if position is None:
                temp_k = self._base_temperature
            else:
                temp_k = self._calculate_local_temperature(position)

            return self._convert_from_kelvin(temp_k, unit)

    def set_gradient(self, gradient_vector: Tuple[float, float, float],
                     unit: str = "K/m") -> None:
        """
        Set linear temperature gradient.

        Args:
            gradient_vector: (dT/dx, dT/dy, dT/dz) in K/m
            unit: Gradient unit (default: K/m)
        """
        with self._lock:
            self._gradient_field = np.array(gradient_vector, dtype=np.float64)

    def add_heat_source(self, position: Tuple[float, float, float],
                       power: float, radius: float = 0.01) -> int:
        """
        Add a point or volumetric heat source.

        Args:
            position: (x, y, z) coordinates in meters
            power: Heat power in Watts
            radius: Effective radius in meters for heat distribution

        Returns:
            Source ID for later removal
        """
        with self._lock:
            source_id = len(self._heat_sources)
            self._heat_sources.append({
                'id': source_id,
                'position': np.array(position, dtype=np.float64),
                'power': power,
                'radius': radius
            })
            return source_id

    def add_heat_sink(self, position: Tuple[float, float, float],
                     power: float, radius: float = 0.01) -> int:
        """
        Add a heat sink (cooling source).

        Args:
            position: (x, y, z) coordinates in meters
            power: Cooling power in Watts (positive value)
            radius: Effective radius in meters

        Returns:
            Sink ID for later removal
        """
        with self._lock:
            sink_id = len(self._heat_sinks)
            self._heat_sinks.append({
                'id': sink_id,
                'position': np.array(position, dtype=np.float64),
                'power': power,
                'radius': radius
            })
            return sink_id

    def enable_radiative_heating(self, blackbody_temperature: float = 5778,
                                emissivity: float = 1.0,
                                direction: Tuple[float, float, float] = (0, 0, -1)) -> None:
        """
        Enable radiative heating from a blackbody source (e.g., solar radiation).

        Args:
            blackbody_temperature: Source temperature in Kelvin (default: 5778 K for Sun)
            emissivity: Surface emissivity (0-1)
            direction: Normalized direction vector of radiation
        """
        with self._lock:
            self._radiative_enabled = True
            self._radiation_temperature = blackbody_temperature
            self._emissivity = emissivity
            self._radiation_direction = np.array(direction, dtype=np.float64)
            self._radiation_direction /= np.linalg.norm(self._radiation_direction)

    def disable_radiative_heating(self) -> None:
        """Disable radiative heating."""
        with self._lock:
            self._radiative_enabled = False

    def set_heating_profile(self, profile_func: Callable[[float], float]) -> None:
        """
        Set time-dependent heating profile.

        Args:
            profile_func: Function taking time (seconds) and returning temperature (K)
        """
        with self._lock:
            self._heating_profile = profile_func

    def set_cryogenic_mode(self, temperature_k: float,
                          cooldown_rate: float = 1.0) -> None:
        """
        Configure cryogenic cooling mode.

        Args:
            temperature_k: Target temperature in Kelvin
            cooldown_rate: Cooling rate in K/s
        """
        with self._lock:
            if temperature_k < 0:
                raise ValueError("Temperature must be above absolute zero")

            self._base_temperature = self._quantize(temperature_k)
            self._cooldown_rate = cooldown_rate

    def calculate_thermal_flux(self, position: Tuple[float, float, float],
                              normal_vector: Tuple[float, float, float]) -> float:
        """
        Calculate thermal flux at a surface element.

        Args:
            position: (x, y, z) coordinates in meters
            normal_vector: Surface normal vector (outward)

        Returns:
            Heat flux in W/m²
        """
        with self._lock:
            flux = 0.0

            # Conductive flux from gradient
            if self._gradient_field is not None:
                normal = np.array(normal_vector, dtype=np.float64)
                normal /= np.linalg.norm(normal)
                flux += np.dot(self._gradient_field, normal)

            # Radiative flux
            if self._radiative_enabled:
                temp_local = self._calculate_local_temperature(position)
                flux += (self._emissivity * self.STEFAN_BOLTZMANN *
                        (self._radiation_temperature**4 - temp_local**4))

            return flux

    def update(self, dt: float) -> None:
        """
        Update temperature state for time step.

        Args:
            dt: Time step in seconds
        """
        with self._lock:
            # Update time-dependent heating profile
            if self._heating_profile is not None:
                # This would require tracking simulation time
                pass

    def _calculate_local_temperature(self, position: Tuple[float, float, float]) -> float:
        """
        Calculate temperature at a specific position including all effects.

        Args:
            position: (x, y, z) coordinates

        Returns:
            Temperature in Kelvin
        """
        pos = np.array(position, dtype=np.float64)
        temp = self._base_temperature

        # Add gradient contribution
        if self._gradient_field is not None:
            temp += np.dot(self._gradient_field, pos)

        # Add heat source contributions
        for source in self._heat_sources:
            r = np.linalg.norm(pos - source['position'])
            if r < source['radius']:
                # Simplified heat distribution (Gaussian-like)
                contribution = source['power'] / (4 * np.pi * source['radius']**2)
                contribution *= np.exp(-r**2 / (2 * (source['radius']/3)**2))
                temp += contribution * 0.01  # Scaling factor for reasonable temps

        # Subtract heat sink contributions
        for sink in self._heat_sinks:
            r = np.linalg.norm(pos - sink['position'])
            if r < sink['radius']:
                contribution = sink['power'] / (4 * np.pi * sink['radius']**2)
                contribution *= np.exp(-r**2 / (2 * (sink['radius']/3)**2))
                temp -= contribution * 0.01

        # Ensure temperature is above absolute zero
        temp = max(0.0, temp)

        return self._quantize(temp)

    def _convert_to_kelvin(self, temperature: float, unit: str) -> float:
        """Convert temperature to Kelvin."""
        unit = unit.upper()
        if unit == "K":
            return temperature
        elif unit == "C":
            return temperature + 273.15
        elif unit == "F":
            return (temperature - 32) * 5/9 + 273.15
        else:
            raise ValueError(f"Unknown temperature unit: {unit}")

    def _convert_from_kelvin(self, temperature_k: float, unit: str) -> float:
        """Convert temperature from Kelvin to specified unit."""
        unit = unit.upper()
        if unit == "K":
            return temperature_k
        elif unit == "C":
            return temperature_k - 273.15
        elif unit == "F":
            return (temperature_k - 273.15) * 9/5 + 32
        else:
            raise ValueError(f"Unknown temperature unit: {unit}")

    def _quantize(self, value: float) -> float:
        """Quantize value to specified precision."""
        return np.round(value / self.precision) * self.precision

    def get_state(self) -> dict:
        """
        Get complete temperature state.

        Returns:
            Dictionary with all temperature parameters
        """
        with self._lock:
            return {
                'base_temperature_K': self._base_temperature,
                'base_temperature_C': self._convert_from_kelvin(self._base_temperature, "C"),
                'precision_K': self.precision,
                'gradient_field': self._gradient_field.tolist() if self._gradient_field is not None else None,
                'num_heat_sources': len(self._heat_sources),
                'num_heat_sinks': len(self._heat_sinks),
                'radiative_enabled': self._radiative_enabled,
                'radiation_temperature_K': self._radiation_temperature if self._radiative_enabled else None,
            }

    def reset(self) -> None:
        """Reset temperature control to default state."""
        with self._lock:
            self._base_temperature = 298.15  # K (25°C)
            self._gradient_field = None
            self._heat_sources = []
            self._heat_sinks = []
            self._radiative_enabled = False
            self._heating_profile = None
