# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Mechanical Forces System
Gravity (0g to 100g), centrifugal forces, vibration, acoustic waves, stress/strain fields
"""

import numpy as np
from typing import Tuple, Optional, Callable
import threading


class MechanicalForces:
    """
    Comprehensive mechanical forces system with gravity fields,
    vibration, acoustic waves, and stress/strain analysis.
    """

    # Physical constants
    EARTH_GRAVITY = 9.80665  # m/s²

    def __init__(self):
        """Initialize mechanical forces system."""
        self._lock = threading.RLock()

        # Gravity
        self._gravity_magnitude = self.EARTH_GRAVITY  # m/s²
        self._gravity_direction = np.array([0, 0, -1], dtype=np.float64)  # Downward
        self._gravity_field = None  # Non-uniform gravity field

        # Centrifugal forces
        self._rotation_axis = None
        self._rotation_center = None
        self._angular_velocity = 0.0  # rad/s

        # Vibration
        self._vibrations = []  # List of vibration sources

        # Acoustic waves
        self._acoustic_sources = []

        # Stress/strain fields
        self._applied_stress = None
        self._strain_field = None

    def set_gravity(self, g_factor: float = 1.0,
                   direction: Tuple[float, float, float] = (0, 0, -1)) -> None:
        """
        Set uniform gravity field.

        Args:
            g_factor: Gravity factor (1.0 = Earth gravity, 0.0 = microgravity)
            direction: Normalized direction vector (default: downward)

        Raises:
            ValueError: If g_factor is out of range
        """
        with self._lock:
            if not (0 <= g_factor <= 100):
                raise ValueError("Gravity factor must be 0-100 (0g to 100g)")

            self._gravity_magnitude = g_factor * self.EARTH_GRAVITY

            # Normalize direction
            dir_array = np.array(direction, dtype=np.float64)
            dir_norm = np.linalg.norm(dir_array)
            if dir_norm > 0:
                self._gravity_direction = dir_array / dir_norm
            else:
                raise ValueError("Direction vector cannot be zero")

    def get_gravity(self, position: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
        """
        Get gravity vector at position.

        Args:
            position: (x, y, z) coordinates (None for uniform field)

        Returns:
            Gravity acceleration vector (m/s²)
        """
        with self._lock:
            if position is not None and self._gravity_field is not None:
                # Non-uniform field (e.g., near massive object)
                return self._gravity_field(position)
            else:
                # Uniform field
                return self._gravity_magnitude * self._gravity_direction

    def set_gravity_field(self, field_function: Callable) -> None:
        """
        Set non-uniform gravity field.

        Args:
            field_function: Function taking (x,y,z) and returning gravity vector
        """
        with self._lock:
            self._gravity_field = field_function

    def set_rotation(self, angular_velocity: float,
                    axis: Tuple[float, float, float] = (0, 0, 1),
                    center: Tuple[float, float, float] = (0, 0, 0)) -> None:
        """
        Set rotation for centrifugal force calculation.

        Args:
            angular_velocity: Rotation rate in rad/s
            axis: Rotation axis (normalized)
            center: Center of rotation
        """
        with self._lock:
            self._angular_velocity = angular_velocity

            # Normalize axis
            axis_array = np.array(axis, dtype=np.float64)
            axis_norm = np.linalg.norm(axis_array)
            if axis_norm > 0:
                self._rotation_axis = axis_array / axis_norm
            else:
                raise ValueError("Rotation axis cannot be zero")

            self._rotation_center = np.array(center, dtype=np.float64)

    def get_centrifugal_force(self, position: Tuple[float, float, float],
                             mass: float = 1.0) -> np.ndarray:
        """
        Calculate centrifugal force at position.

        Args:
            position: (x, y, z) coordinates
            mass: Mass in kg (default: 1.0)

        Returns:
            Centrifugal force vector in Newtons
        """
        with self._lock:
            if self._rotation_axis is None or self._angular_velocity == 0:
                return np.zeros(3, dtype=np.float64)

            # Position relative to rotation center
            pos = np.array(position, dtype=np.float64) - self._rotation_center

            # Distance from rotation axis
            r_perp = pos - np.dot(pos, self._rotation_axis) * self._rotation_axis
            r_mag = np.linalg.norm(r_perp)

            if r_mag == 0:
                return np.zeros(3, dtype=np.float64)

            # Centrifugal force: F = m * ω² * r (outward)
            force_mag = mass * self._angular_velocity**2 * r_mag
            force_dir = r_perp / r_mag

            return force_mag * force_dir

    def add_vibration(self, vibration_type: str, frequency: float,
                     amplitude: float, direction: Tuple[float, float, float] = (0, 0, 1),
                     phase: float = 0.0) -> int:
        """
        Add vibration source.

        Args:
            vibration_type: "sinusoidal", "random", or "shock"
            frequency: Frequency in Hz (ignored for random)
            amplitude: Amplitude in meters
            direction: Vibration direction (normalized)
            phase: Phase offset in radians

        Returns:
            Vibration source ID
        """
        with self._lock:
            # Normalize direction
            dir_array = np.array(direction, dtype=np.float64)
            dir_norm = np.linalg.norm(dir_array)
            if dir_norm > 0:
                direction_normalized = dir_array / dir_norm
            else:
                raise ValueError("Direction vector cannot be zero")

            vib_id = len(self._vibrations)
            self._vibrations.append({
                'id': vib_id,
                'type': vibration_type,
                'frequency': frequency,
                'amplitude': amplitude,
                'direction': direction_normalized,
                'phase': phase,
                'random_state': np.random.RandomState(vib_id)  # For random vibration
            })
            return vib_id

    def get_vibration_displacement(self, time: float,
                                  vibration_id: Optional[int] = None) -> np.ndarray:
        """
        Get vibration displacement at time.

        Args:
            time: Time in seconds
            vibration_id: Specific vibration ID (None for sum of all)

        Returns:
            Displacement vector in meters
        """
        with self._lock:
            displacement = np.zeros(3, dtype=np.float64)

            vibrations = [self._vibrations[vibration_id]] if vibration_id is not None else self._vibrations

            for vib in vibrations:
                if vib['type'] == 'sinusoidal':
                    # x(t) = A * sin(2πft + φ)
                    disp_mag = vib['amplitude'] * np.sin(2 * np.pi * vib['frequency'] * time + vib['phase'])
                    displacement += disp_mag * vib['direction']

                elif vib['type'] == 'random':
                    # Random vibration (white noise)
                    disp_mag = vib['random_state'].normal(0, vib['amplitude'])
                    displacement += disp_mag * vib['direction']

                elif vib['type'] == 'shock':
                    # Shock pulse (half-sine)
                    pulse_duration = 1.0 / vib['frequency']
                    if 0 <= time <= pulse_duration:
                        disp_mag = vib['amplitude'] * np.sin(np.pi * time / pulse_duration)
                        displacement += disp_mag * vib['direction']

            return displacement

    def get_vibration_acceleration(self, time: float,
                                   vibration_id: Optional[int] = None) -> np.ndarray:
        """
        Get vibration acceleration at time.

        Args:
            time: Time in seconds
            vibration_id: Specific vibration ID (None for sum of all)

        Returns:
            Acceleration vector in m/s²
        """
        with self._lock:
            acceleration = np.zeros(3, dtype=np.float64)

            vibrations = [self._vibrations[vibration_id]] if vibration_id is not None else self._vibrations

            for vib in vibrations:
                if vib['type'] == 'sinusoidal':
                    # a(t) = -A * (2πf)² * sin(2πft + φ)
                    omega = 2 * np.pi * vib['frequency']
                    accel_mag = -vib['amplitude'] * omega**2 * np.sin(omega * time + vib['phase'])
                    acceleration += accel_mag * vib['direction']

                elif vib['type'] == 'random':
                    # Random acceleration
                    accel_mag = vib['random_state'].normal(0, vib['amplitude'] * (2 * np.pi * 100)**2)
                    acceleration += accel_mag * vib['direction']

                elif vib['type'] == 'shock':
                    # Shock acceleration
                    pulse_duration = 1.0 / vib['frequency']
                    if 0 <= time <= pulse_duration:
                        accel_mag = vib['amplitude'] * (np.pi / pulse_duration)**2 * np.sin(np.pi * time / pulse_duration)
                        acceleration += accel_mag * vib['direction']

            return acceleration

    def add_acoustic_wave(self, frequency: float, amplitude: float,
                         origin: Tuple[float, float, float],
                         speed: float = 343.0) -> int:
        """
        Add acoustic wave source.

        Args:
            frequency: Frequency in Hz
            amplitude: Pressure amplitude in Pa
            origin: Wave origin (x, y, z)
            speed: Speed of sound in m/s (default: 343 m/s in air)

        Returns:
            Acoustic source ID
        """
        with self._lock:
            source_id = len(self._acoustic_sources)
            self._acoustic_sources.append({
                'id': source_id,
                'frequency': frequency,
                'amplitude': amplitude,
                'origin': np.array(origin, dtype=np.float64),
                'speed': speed,
                'wavelength': speed / frequency
            })
            return source_id

    def get_acoustic_pressure(self, position: Tuple[float, float, float],
                             time: float) -> float:
        """
        Calculate acoustic pressure at position and time.

        Args:
            position: (x, y, z) coordinates
            time: Time in seconds

        Returns:
            Pressure amplitude in Pa
        """
        with self._lock:
            pressure = 0.0
            pos = np.array(position, dtype=np.float64)

            for source in self._acoustic_sources:
                # Distance from source
                r = np.linalg.norm(pos - source['origin'])

                # Wave number
                k = 2 * np.pi / source['wavelength']

                # Spherical wave: p(r,t) = (A/r) * sin(kr - ωt)
                omega = 2 * np.pi * source['frequency']
                if r > 0:
                    pressure += (source['amplitude'] / r) * np.sin(k * r - omega * time)

            return pressure

    def apply_stress(self, stress_tensor: np.ndarray) -> None:
        """
        Apply stress field (3x3 stress tensor).

        Args:
            stress_tensor: 3x3 stress tensor in Pa
        """
        with self._lock:
            if stress_tensor.shape != (3, 3):
                raise ValueError("Stress tensor must be 3x3")

            self._applied_stress = stress_tensor.copy()

    def calculate_strain(self, elastic_modulus: float, poisson_ratio: float) -> np.ndarray:
        """
        Calculate strain from applied stress (linear elasticity).

        Args:
            elastic_modulus: Young's modulus in Pa
            poisson_ratio: Poisson's ratio (dimensionless)

        Returns:
            3x3 strain tensor
        """
        with self._lock:
            if self._applied_stress is None:
                return np.zeros((3, 3), dtype=np.float64)

            # Simplified isotropic linear elasticity
            # ε = (1/E) * [(1+ν)*σ - ν*tr(σ)*I]
            E = elastic_modulus
            nu = poisson_ratio

            trace = np.trace(self._applied_stress)
            strain = ((1 + nu) / E) * self._applied_stress - (nu / E) * trace * np.eye(3)

            self._strain_field = strain
            return strain

    def get_total_force(self, position: Tuple[float, float, float],
                       mass: float, time: float = 0.0) -> np.ndarray:
        """
        Get total force at position (gravity + centrifugal + vibration).

        Args:
            position: (x, y, z) coordinates
            mass: Mass in kg
            time: Time in seconds (for vibration)

        Returns:
            Total force vector in Newtons
        """
        with self._lock:
            # Gravity force
            g = self.get_gravity(position)
            force = mass * g

            # Centrifugal force
            force += self.get_centrifugal_force(position, mass)

            # Vibration force (F = ma)
            accel = self.get_vibration_acceleration(time)
            force += mass * accel

            return force

    def get_state(self) -> dict:
        """
        Get complete mechanical forces state.

        Returns:
            Dictionary with all force parameters
        """
        with self._lock:
            return {
                'gravity_magnitude_m_s2': self._gravity_magnitude,
                'gravity_direction': self._gravity_direction.tolist(),
                'gravity_g_factor': self._gravity_magnitude / self.EARTH_GRAVITY,
                'angular_velocity_rad_s': self._angular_velocity,
                'rotation_axis': self._rotation_axis.tolist() if self._rotation_axis is not None else None,
                'num_vibrations': len(self._vibrations),
                'num_acoustic_sources': len(self._acoustic_sources),
                'has_applied_stress': self._applied_stress is not None,
            }

    def reset(self) -> None:
        """Reset mechanical forces to default state."""
        with self._lock:
            self._gravity_magnitude = self.EARTH_GRAVITY
            self._gravity_direction = np.array([0, 0, -1], dtype=np.float64)
            self._gravity_field = None
            self._rotation_axis = None
            self._rotation_center = None
            self._angular_velocity = 0.0
            self._vibrations = []
            self._acoustic_sources = []
            self._applied_stress = None
            self._strain_field = None
