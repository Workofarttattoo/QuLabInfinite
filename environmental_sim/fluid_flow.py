# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Fluid Flow System
Wind, turbulence, boundary layers, vortices, CFD integration
"""

import numpy as np
from typing import Tuple, Optional, Callable
import threading


class FluidFlow:
    """
    Comprehensive fluid flow system with wind, turbulence,
    boundary layers, and vortex modeling.
    """

    def __init__(self):
        """Initialize fluid flow system."""
        self._lock = threading.RLock()

        # Wind field
        self._wind_velocity = np.zeros(3, dtype=np.float64)  # m/s
        self._wind_profile = None  # Height-dependent wind profile

        # Turbulence
        self._turbulence_intensity = 0.0  # 0-1 (0=laminar, 1=highly turbulent)
        self._turbulence_length_scale = 1.0  # meters

        # Flow regime
        self._reynolds_number = 0.0
        self._flow_regime = "laminar"  # laminar, transitional, turbulent

        # Boundary layers
        self._boundary_layers = []

        # Vortices
        self._vortices = []

        # Fluid properties
        self._fluid_density = 1.225  # kg/m³ (air at STP)
        self._dynamic_viscosity = 1.81e-5  # Pa·s (air at 20°C)

    def set_wind(self, velocity: Tuple[float, float, float],
                unit: str = "m/s") -> None:
        """
        Set uniform wind velocity.

        Args:
            velocity: (vx, vy, vz) velocity vector
            unit: Velocity unit ("m/s", "mph", "km/h", "knots")
        """
        with self._lock:
            vel_array = np.array(velocity, dtype=np.float64)

            # Convert to m/s
            if unit == "m/s":
                vel_m_s = vel_array
            elif unit == "mph":
                vel_m_s = vel_array * 0.44704
            elif unit == "km/h":
                vel_m_s = vel_array / 3.6
            elif unit == "knots":
                vel_m_s = vel_array * 0.514444
            else:
                raise ValueError(f"Unknown velocity unit: {unit}")

            # Check maximum wind speed
            speed = np.linalg.norm(vel_m_s)
            if speed > 500 * 0.44704:  # 500 mph
                raise ValueError(f"Wind speed {speed} m/s exceeds maximum 223 m/s (500 mph)")

            self._wind_velocity = vel_m_s

    def get_wind(self, position: Optional[Tuple[float, float, float]] = None,
                unit: str = "m/s") -> np.ndarray:
        """
        Get wind velocity at position.

        Args:
            position: (x, y, z) coordinates (None for base wind)
            unit: Velocity unit for return value

        Returns:
            Wind velocity vector
        """
        with self._lock:
            if position is not None and self._wind_profile is not None:
                # Height-dependent wind profile
                velocity = self._wind_profile(position)
            else:
                velocity = self._wind_velocity.copy()

            # Add turbulence
            if self._turbulence_intensity > 0:
                turbulence = self._generate_turbulence()
                velocity += turbulence

            # Convert units
            if unit == "m/s":
                return velocity
            elif unit == "mph":
                return velocity / 0.44704
            elif unit == "km/h":
                return velocity * 3.6
            elif unit == "knots":
                return velocity / 0.514444
            else:
                raise ValueError(f"Unknown velocity unit: {unit}")

    def set_wind_profile(self, profile_func: Callable) -> None:
        """
        Set height-dependent wind profile.

        Args:
            profile_func: Function taking (x,y,z) and returning velocity vector
        """
        with self._lock:
            self._wind_profile = profile_func

    def set_power_law_profile(self, reference_velocity: float,
                             reference_height: float,
                             exponent: float = 0.143) -> None:
        """
        Set power law wind profile: v(z) = v_ref * (z/z_ref)^α

        Args:
            reference_velocity: Wind speed at reference height (m/s)
            reference_height: Reference height (m)
            exponent: Power law exponent (default: 0.143 for open terrain)
        """
        with self._lock:
            def power_law_profile(position):
                z = position[2]  # Height
                if z <= 0:
                    return np.zeros(3)

                speed = reference_velocity * (z / reference_height) ** exponent
                # Assume wind in x-direction
                direction = self._wind_velocity / (np.linalg.norm(self._wind_velocity) + 1e-10)
                return speed * direction

            self._wind_profile = power_law_profile

    def set_turbulence(self, intensity: float, length_scale: float = 1.0) -> None:
        """
        Set turbulence parameters.

        Args:
            intensity: Turbulence intensity (0-1, 0=laminar, 1=highly turbulent)
            length_scale: Characteristic length scale in meters

        Raises:
            ValueError: If intensity is out of range
        """
        with self._lock:
            if not (0 <= intensity <= 1):
                raise ValueError("Turbulence intensity must be 0-1")

            self._turbulence_intensity = intensity
            self._turbulence_length_scale = length_scale

    def calculate_reynolds_number(self, characteristic_length: float) -> float:
        """
        Calculate Reynolds number for flow.

        Args:
            characteristic_length: Characteristic length scale (m)

        Returns:
            Reynolds number (dimensionless)
        """
        with self._lock:
            speed = np.linalg.norm(self._wind_velocity)
            self._reynolds_number = (self._fluid_density * speed * characteristic_length /
                                    self._dynamic_viscosity)

            # Classify flow regime
            if self._reynolds_number < 2300:
                self._flow_regime = "laminar"
            elif self._reynolds_number < 4000:
                self._flow_regime = "transitional"
            else:
                self._flow_regime = "turbulent"

            return self._reynolds_number

    def get_flow_regime(self) -> str:
        """
        Get flow regime classification.

        Returns:
            "laminar", "transitional", or "turbulent"
        """
        with self._lock:
            return self._flow_regime

    def add_boundary_layer(self, surface_position: Tuple[float, float, float],
                          surface_normal: Tuple[float, float, float],
                          thickness: float) -> int:
        """
        Add boundary layer near surface.

        Args:
            surface_position: Point on surface
            surface_normal: Surface normal vector (normalized)
            thickness: Boundary layer thickness (m)

        Returns:
            Boundary layer ID
        """
        with self._lock:
            # Normalize normal
            normal = np.array(surface_normal, dtype=np.float64)
            normal /= np.linalg.norm(normal)

            layer_id = len(self._boundary_layers)
            self._boundary_layers.append({
                'id': layer_id,
                'position': np.array(surface_position, dtype=np.float64),
                'normal': normal,
                'thickness': thickness
            })
            return layer_id

    def add_vortex(self, center: Tuple[float, float, float],
                  axis: Tuple[float, float, float],
                  circulation: float, core_radius: float = 0.1) -> int:
        """
        Add vortex structure.

        Args:
            center: Vortex center (x, y, z)
            axis: Vortex axis direction (normalized)
            circulation: Circulation strength (m²/s)
            core_radius: Vortex core radius (m)

        Returns:
            Vortex ID
        """
        with self._lock:
            # Normalize axis
            axis_array = np.array(axis, dtype=np.float64)
            axis_norm = np.linalg.norm(axis_array)
            if axis_norm > 0:
                axis_normalized = axis_array / axis_norm
            else:
                raise ValueError("Vortex axis cannot be zero")

            vortex_id = len(self._vortices)
            self._vortices.append({
                'id': vortex_id,
                'center': np.array(center, dtype=np.float64),
                'axis': axis_normalized,
                'circulation': circulation,
                'core_radius': core_radius
            })
            return vortex_id

    def get_vortex_velocity(self, position: Tuple[float, float, float],
                           vortex_id: Optional[int] = None) -> np.ndarray:
        """
        Get velocity induced by vortex(es) at position.

        Args:
            position: (x, y, z) coordinates
            vortex_id: Specific vortex ID (None for all vortices)

        Returns:
            Velocity vector induced by vortex
        """
        with self._lock:
            velocity = np.zeros(3, dtype=np.float64)
            pos = np.array(position, dtype=np.float64)

            vortices = [self._vortices[vortex_id]] if vortex_id is not None else self._vortices

            for vortex in vortices:
                # Position relative to vortex center
                r_vec = pos - vortex['center']

                # Distance from vortex axis
                r_parallel = np.dot(r_vec, vortex['axis']) * vortex['axis']
                r_perp = r_vec - r_parallel
                r = np.linalg.norm(r_perp)

                if r > 1e-10:
                    # Rankine vortex model
                    if r < vortex['core_radius']:
                        # Solid body rotation inside core
                        v_theta = vortex['circulation'] * r / (2 * np.pi * vortex['core_radius']**2)
                    else:
                        # Potential flow outside core
                        v_theta = vortex['circulation'] / (2 * np.pi * r)

                    # Tangential velocity direction
                    tangent = np.cross(vortex['axis'], r_perp / r)
                    velocity += v_theta * tangent

            return velocity

    def calculate_drag_force(self, velocity: Tuple[float, float, float],
                            drag_coefficient: float, reference_area: float) -> np.ndarray:
        """
        Calculate drag force on object.

        Args:
            velocity: Object velocity relative to fluid (m/s)
            drag_coefficient: Drag coefficient (dimensionless)
            reference_area: Reference area (m²)

        Returns:
            Drag force vector (N)
        """
        with self._lock:
            vel = np.array(velocity, dtype=np.float64)
            speed = np.linalg.norm(vel)

            if speed < 1e-10:
                return np.zeros(3, dtype=np.float64)

            # Drag force: F_d = 0.5 * ρ * v² * C_d * A
            force_mag = 0.5 * self._fluid_density * speed**2 * drag_coefficient * reference_area
            force_dir = -vel / speed  # Opposite to velocity

            return force_mag * force_dir

    def set_fluid_properties(self, density: float, dynamic_viscosity: float) -> None:
        """
        Set fluid properties.

        Args:
            density: Fluid density (kg/m³)
            dynamic_viscosity: Dynamic viscosity (Pa·s)
        """
        with self._lock:
            self._fluid_density = density
            self._dynamic_viscosity = dynamic_viscosity

    def _generate_turbulence(self) -> np.ndarray:
        """
        Generate random turbulent fluctuations.

        Returns:
            Turbulent velocity fluctuation vector
        """
        # Simplified turbulence model (Gaussian random field)
        base_speed = np.linalg.norm(self._wind_velocity)
        turbulent_velocity = np.random.normal(0, self._turbulence_intensity * base_speed, 3)

        return turbulent_velocity

    def get_state(self) -> dict:
        """
        Get complete fluid flow state.

        Returns:
            Dictionary with all flow parameters
        """
        with self._lock:
            return {
                'wind_velocity_m_s': self._wind_velocity.tolist(),
                'wind_speed_m_s': float(np.linalg.norm(self._wind_velocity)),
                'wind_speed_mph': float(np.linalg.norm(self._wind_velocity) / 0.44704),
                'turbulence_intensity': self._turbulence_intensity,
                'turbulence_length_scale_m': self._turbulence_length_scale,
                'reynolds_number': self._reynolds_number,
                'flow_regime': self._flow_regime,
                'fluid_density_kg_m3': self._fluid_density,
                'dynamic_viscosity_Pa_s': self._dynamic_viscosity,
                'num_boundary_layers': len(self._boundary_layers),
                'num_vortices': len(self._vortices),
            }

    def reset(self) -> None:
        """Reset fluid flow to default state."""
        with self._lock:
            self._wind_velocity = np.zeros(3, dtype=np.float64)
            self._wind_profile = None
            self._turbulence_intensity = 0.0
            self._turbulence_length_scale = 1.0
            self._reynolds_number = 0.0
            self._flow_regime = "laminar"
            self._boundary_layers = []
            self._vortices = []
            self._fluid_density = 1.225
            self._dynamic_viscosity = 1.81e-5
