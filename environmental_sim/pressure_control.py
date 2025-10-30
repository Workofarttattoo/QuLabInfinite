# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Pressure Control System
Range: 0 to 1,000,000 bar with ±0.01% precision
"""

import numpy as np
from typing import Optional, Tuple
import threading


class PressureControl:
    """
    High-precision pressure control system with vacuum, supercritical fluids,
    and shock wave capabilities.
    """

    # Physical constants
    ATMOSPHERIC_PRESSURE = 1.01325  # bar
    PASCAL_PER_BAR = 1e5  # Pa/bar

    def __init__(self, precision_percent: float = 0.01):
        """
        Initialize pressure control system.

        Args:
            precision_percent: Relative precision in percent (default: 0.01%)
        """
        self.precision_percent = precision_percent
        self._lock = threading.RLock()

        # Pressure state
        self._base_pressure = self.ATMOSPHERIC_PRESSURE  # bar
        self._pressure_field = None  # 3D pressure field
        self._pressure_gradients = []  # Pressure gradient regions
        self._shock_waves = []  # Active shock waves

        # Vacuum parameters
        self._vacuum_level = None  # Pressure in torr for vacuum mode
        self._is_vacuum_mode = False

    def set_pressure(self, pressure: float, unit: str = "bar") -> None:
        """
        Set uniform base pressure.

        Args:
            pressure: Pressure value
            unit: Pressure unit ("bar", "Pa", "atm", "psi", "torr", "GPa")

        Raises:
            ValueError: If pressure is negative or exceeds maximum
        """
        with self._lock:
            pressure_bar = self._convert_to_bar(pressure, unit)

            if pressure_bar < 0:
                raise ValueError(f"Pressure cannot be negative: {pressure_bar} bar")

            if pressure_bar > 1e6:  # 1,000,000 bar
                raise ValueError(f"Pressure {pressure_bar} bar exceeds maximum 1,000,000 bar")

            # Quantize to precision
            self._base_pressure = self._quantize(pressure_bar)
            self._is_vacuum_mode = pressure_bar < 1e-3  # Below 1 mbar is vacuum

    def get_pressure(self, position: Optional[Tuple[float, float, float]] = None,
                    unit: str = "bar") -> float:
        """
        Get pressure at a specific position or base pressure.

        Args:
            position: (x, y, z) coordinates in meters (None for base pressure)
            unit: Pressure unit for return value

        Returns:
            Pressure at position with specified precision
        """
        with self._lock:
            if position is None:
                pressure_bar = self._base_pressure
            else:
                pressure_bar = self._calculate_local_pressure(position)

            return self._convert_from_bar(pressure_bar, unit)

    def set_vacuum_level(self, pressure: float, unit: str = "torr") -> None:
        """
        Set vacuum level with high precision.

        Args:
            pressure: Vacuum pressure (common units: torr, Pa)
            unit: Pressure unit
        """
        with self._lock:
            pressure_bar = self._convert_to_bar(pressure, unit)

            if pressure_bar >= 1.0:
                raise ValueError(f"Vacuum pressure must be < 1 bar, got {pressure_bar} bar")

            self._base_pressure = self._quantize(pressure_bar)
            self._is_vacuum_mode = True
            self._vacuum_level = self._convert_from_bar(pressure_bar, "torr")

    def set_pressure_gradient(self, direction: Tuple[float, float, float],
                             gradient: float, unit: str = "bar/m") -> None:
        """
        Set pressure gradient (e.g., hydrostatic pressure).

        Args:
            direction: Direction vector (will be normalized)
            gradient: Pressure change per meter
            unit: Gradient unit (default: bar/m)
        """
        with self._lock:
            direction_norm = np.array(direction, dtype=np.float64)
            direction_norm /= np.linalg.norm(direction_norm)

            # Convert gradient to bar/m
            if unit == "Pa/m":
                gradient_bar = gradient / self.PASCAL_PER_BAR
            elif unit == "bar/m":
                gradient_bar = gradient
            else:
                raise ValueError(f"Unknown gradient unit: {unit}")

            self._pressure_gradients.append({
                'direction': direction_norm,
                'gradient': gradient_bar
            })

    def add_shock_wave(self, origin: Tuple[float, float, float],
                      peak_pressure: float, velocity: float,
                      start_time: float = 0.0) -> int:
        """
        Add a propagating shock wave.

        Args:
            origin: (x, y, z) origin point
            peak_pressure: Peak overpressure in bar
            velocity: Shock wave velocity in m/s
            start_time: Wave start time in seconds

        Returns:
            Shock wave ID
        """
        with self._lock:
            wave_id = len(self._shock_waves)
            self._shock_waves.append({
                'id': wave_id,
                'origin': np.array(origin, dtype=np.float64),
                'peak_pressure': peak_pressure,
                'velocity': velocity,
                'start_time': start_time,
                'current_radius': 0.0
            })
            return wave_id

    def is_supercritical(self, temperature_k: float, substance: str = "CO2") -> bool:
        """
        Check if conditions are supercritical for given substance.

        Args:
            temperature_k: Temperature in Kelvin
            substance: Substance name (default: CO2)

        Returns:
            True if in supercritical regime
        """
        # Critical points for common substances
        critical_points = {
            'CO2': {'T': 304.13, 'P': 73.8},  # K, bar
            'H2O': {'T': 647.1, 'P': 220.6},
            'N2': {'T': 126.2, 'P': 34.0},
            'O2': {'T': 154.6, 'P': 50.4},
            'He': {'T': 5.2, 'P': 2.27},
        }

        if substance not in critical_points:
            raise ValueError(f"Unknown substance: {substance}")

        crit = critical_points[substance]
        return (temperature_k > crit['T'] and self._base_pressure > crit['P'])

    def calculate_hydrostatic_pressure(self, depth: float, fluid_density: float = 1000,
                                      gravity: float = 9.81) -> float:
        """
        Calculate hydrostatic pressure at depth.

        Args:
            depth: Depth in meters (positive downward)
            fluid_density: Fluid density in kg/m³ (default: water)
            gravity: Gravitational acceleration in m/s² (default: Earth)

        Returns:
            Total pressure in bar (atmospheric + hydrostatic)
        """
        # P = P_atm + ρgh
        hydrostatic_pa = fluid_density * gravity * depth
        hydrostatic_bar = hydrostatic_pa / self.PASCAL_PER_BAR

        with self._lock:
            total_pressure = self._base_pressure + hydrostatic_bar
            return self._quantize(total_pressure)

    def calculate_compression_ratio(self, reference_pressure: float = None,
                                   unit: str = "bar") -> float:
        """
        Calculate compression ratio relative to reference pressure.

        Args:
            reference_pressure: Reference pressure (default: atmospheric)
            unit: Unit for reference pressure

        Returns:
            Compression ratio (dimensionless)
        """
        if reference_pressure is None:
            reference_pressure = self.ATMOSPHERIC_PRESSURE
        else:
            reference_pressure = self._convert_to_bar(reference_pressure, unit)

        with self._lock:
            return self._base_pressure / reference_pressure

    def update(self, dt: float, current_time: float = 0.0) -> None:
        """
        Update pressure state for time step (e.g., shock wave propagation).

        Args:
            dt: Time step in seconds
            current_time: Current simulation time in seconds
        """
        with self._lock:
            # Update shock wave radii
            for wave in self._shock_waves:
                if current_time >= wave['start_time']:
                    wave['current_radius'] += wave['velocity'] * dt

    def _calculate_local_pressure(self, position: Tuple[float, float, float]) -> float:
        """
        Calculate pressure at a specific position including all effects.

        Args:
            position: (x, y, z) coordinates

        Returns:
            Pressure in bar
        """
        pos = np.array(position, dtype=np.float64)
        pressure = self._base_pressure

        # Add gradient contributions
        for gradient in self._pressure_gradients:
            distance = np.dot(pos, gradient['direction'])
            pressure += gradient['gradient'] * distance

        # Add shock wave contributions
        for wave in self._shock_waves:
            r = np.linalg.norm(pos - wave['origin'])
            if abs(r - wave['current_radius']) < 1.0:  # Within shock front
                # Simplified shock profile (exponential decay)
                pressure += wave['peak_pressure'] * np.exp(-(r - wave['current_radius'])**2)

        # Ensure non-negative pressure
        pressure = max(0.0, pressure)

        return self._quantize(pressure)

    def _convert_to_bar(self, pressure: float, unit: str) -> float:
        """Convert pressure to bar."""
        unit = unit.lower()
        if unit == "bar":
            return pressure
        elif unit == "pa":
            return pressure / self.PASCAL_PER_BAR
        elif unit == "atm":
            return pressure * 1.01325
        elif unit == "psi":
            return pressure * 0.0689476
        elif unit == "torr":
            return pressure * 0.00133322
        elif unit == "gpa":
            return pressure * 10000
        elif unit == "mpa":
            return pressure * 10
        else:
            raise ValueError(f"Unknown pressure unit: {unit}")

    def _convert_from_bar(self, pressure_bar: float, unit: str) -> float:
        """Convert pressure from bar to specified unit."""
        unit = unit.lower()
        if unit == "bar":
            return pressure_bar
        elif unit == "pa":
            return pressure_bar * self.PASCAL_PER_BAR
        elif unit == "atm":
            return pressure_bar / 1.01325
        elif unit == "psi":
            return pressure_bar / 0.0689476
        elif unit == "torr":
            return pressure_bar / 0.00133322
        elif unit == "gpa":
            return pressure_bar / 10000
        elif unit == "mpa":
            return pressure_bar / 10
        else:
            raise ValueError(f"Unknown pressure unit: {unit}")

    def _quantize(self, value: float) -> float:
        """Quantize value to specified relative precision."""
        if value == 0:
            return 0.0
        # Relative precision
        precision = abs(value) * (self.precision_percent / 100.0)
        return np.round(value / precision) * precision

    def get_state(self) -> dict:
        """
        Get complete pressure state.

        Returns:
            Dictionary with all pressure parameters
        """
        with self._lock:
            return {
                'base_pressure_bar': self._base_pressure,
                'base_pressure_Pa': self._convert_from_bar(self._base_pressure, "Pa"),
                'base_pressure_atm': self._convert_from_bar(self._base_pressure, "atm"),
                'base_pressure_torr': self._convert_from_bar(self._base_pressure, "torr"),
                'precision_percent': self.precision_percent,
                'is_vacuum_mode': self._is_vacuum_mode,
                'vacuum_level_torr': self._vacuum_level if self._is_vacuum_mode else None,
                'num_pressure_gradients': len(self._pressure_gradients),
                'num_shock_waves': len(self._shock_waves),
            }

    def reset(self) -> None:
        """Reset pressure control to default state."""
        with self._lock:
            self._base_pressure = self.ATMOSPHERIC_PRESSURE
            self._pressure_field = None
            self._pressure_gradients = []
            self._shock_waves = []
            self._vacuum_level = None
            self._is_vacuum_mode = False
