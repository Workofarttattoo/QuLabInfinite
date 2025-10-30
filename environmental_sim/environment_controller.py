# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Environment Controller - Master Controller
Thread-safe, real-time updates, coordinates all environmental subsystems
"""

import numpy as np
import threading
import time
import math
from typing import Dict, Optional, Tuple, Any

from .temperature_control import TemperatureControl
from .pressure_control import PressureControl
from .atmosphere_control import AtmosphereControl
from .mechanical_forces import MechanicalForces
from .fluid_flow import FluidFlow
from .radiation_environment import RadiationEnvironment
from .multi_physics_coupling import MultiPhysicsCoupling


SECONDS_PER_YEAR = 365.25 * 24 * 3600.0


class EnvironmentController:
    """
    Master controller coordinating all environmental subsystems.
    Thread-safe with real-time updates and multi-physics coupling.
    """

    def __init__(self, update_rate: float = 100.0):
        """
        Initialize environment controller.

        Args:
            update_rate: Update rate in Hz (default: 100 Hz)
        """
        self._lock = threading.RLock()

        # Subsystems
        self.temperature = TemperatureControl()
        self.pressure = PressureControl()
        self.atmosphere = AtmosphereControl()
        self.mechanics = MechanicalForces()
        self.fluid = FluidFlow()
        self.radiation = RadiationEnvironment()
        self.coupling = MultiPhysicsCoupling()

        # Update control
        self._update_rate = update_rate
        self._dt = 1.0 / update_rate  # Time step in seconds
        self._simulation_time = 0.0
        self._is_running = False
        self._update_thread = None

        # Corrosion tracking
        self._corrosion_state: Dict[str, Dict[str, Any]] = {}

        # State history (for time-dependent simulations)
        self._history = []
        self._max_history_length = 10000  # Store up to 10,000 timesteps

    @staticmethod
    def _normalize_material_id(material_id: str) -> str:
        """Normalize material identifiers for consistent corrosion tracking."""
        return material_id.strip().lower()

    def _get_or_create_corrosion_entry(self, material_id: str) -> Dict[str, Any]:
        """Fetch corrosion entry for material, creating if missing."""
        key = self._normalize_material_id(material_id)
        entry = self._corrosion_state.get(key)
        if entry is None:
            entry = {
                'material_id': material_id,
                'baseline_rate_mm_per_year': None,
                'active_multiplier': 1.0,
                'adjusted_rate_mm_per_year': None,
                'cumulative_loss_mm': 0.0,
                'total_exposure_hours': 0.0,
                'sources': [],
                'metadata': {},
            }
            self._corrosion_state[key] = entry
        elif not entry.get('material_id'):
            entry['material_id'] = material_id
        return entry

    @staticmethod
    def _clone_corrosion_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        """Create a copy of the corrosion entry safe for external use."""
        clone = dict(entry)
        clone['sources'] = [dict(source) for source in entry.get('sources', [])]
        clone['metadata'] = dict(entry.get('metadata', {}))
        return clone

    def _recalculate_corrosion_rate(self, entry: Dict[str, Any]) -> None:
        """Recalculate adjusted corrosion rate for a material."""
        baseline = entry.get('baseline_rate_mm_per_year')
        multiplier = entry.get('active_multiplier', 1.0)
        if baseline is None:
            entry['adjusted_rate_mm_per_year'] = None
        else:
            entry['adjusted_rate_mm_per_year'] = max(baseline, 0.0) * max(multiplier, 0.0)

    def _update_corrosion_progress(self, dt: float) -> None:
        """Integrate corrosion damage over time."""
        if dt <= 0:
            return

        for entry in self._corrosion_state.values():
            rate = entry.get('adjusted_rate_mm_per_year')
            if rate is None or rate <= 0:
                continue

            increment = rate * (dt / SECONDS_PER_YEAR)
            if not math.isfinite(increment):
                continue
            entry['cumulative_loss_mm'] += increment

    def set_corrosion_baseline(
        self,
        material_id: str,
        rate_mm_per_year: Optional[float],
        *,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register baseline corrosion rate for a material."""
        with self._lock:
            entry = self._get_or_create_corrosion_entry(material_id)
            if rate_mm_per_year is None:
                entry['baseline_rate_mm_per_year'] = None
            else:
                entry['baseline_rate_mm_per_year'] = max(float(rate_mm_per_year), 0.0)
            if metadata:
                entry['metadata'].update(metadata)
            self._recalculate_corrosion_rate(entry)

    def record_corrosion_effect(
        self,
        material_id: str,
        multiplier: float,
        exposure_hours: float = 0.0,
        *,
        source: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Apply corrosion multiplier derived from environmental exposure."""
        with self._lock:
            entry = self._get_or_create_corrosion_entry(material_id)
            try:
                entry['active_multiplier'] *= float(multiplier)
            except (TypeError, ValueError):
                pass

            # Prevent negative multipliers
            if entry['active_multiplier'] < 0:
                entry['active_multiplier'] = 0.0

            try:
                entry['total_exposure_hours'] += max(float(exposure_hours), 0.0)
            except (TypeError, ValueError):
                pass

            if source:
                entry['sources'].append(dict(source))
            if metadata:
                entry['metadata'].update(metadata)

            self._recalculate_corrosion_rate(entry)

    def get_corrosion_state(self, material_id: Optional[str] = None) -> Dict[str, Any]:
        """Return corrosion tracking data."""
        with self._lock:
            if material_id is not None:
                entry = self._corrosion_state.get(self._normalize_material_id(material_id))
                return self._clone_corrosion_entry(entry) if entry else {}

            return {
                (entry.get('material_id') or key): self._clone_corrosion_entry(entry)
                for key, entry in self._corrosion_state.items()
            }

    def clear_corrosion_state(self) -> None:
        """Reset corrosion tracking."""
        with self._lock:
            self._corrosion_state = {}
    def start_realtime_updates(self) -> None:
        """Start real-time environmental updates in background thread."""
        with self._lock:
            if self._is_running:
                return

            self._is_running = True
            self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self._update_thread.start()

    def stop_realtime_updates(self) -> None:
        """Stop real-time environmental updates."""
        with self._lock:
            self._is_running = False
            if self._update_thread is not None:
                self._update_thread.join(timeout=1.0)
                self._update_thread = None

    def update(self, dt: Optional[float] = None) -> None:
        """
        Perform single update step for all subsystems.

        Args:
            dt: Time step in seconds (None to use default)
        """
        with self._lock:
            if dt is None:
                dt = self._dt

            # Update simulation time
            self._simulation_time += dt

            # Update individual subsystems
            self.temperature.update(dt)
            self.pressure.update(dt, self._simulation_time)
            # Atmosphere contaminant decay
            self.atmosphere.update_contaminants(dt)

            # Update corrosion accumulation
            self._update_corrosion_progress(dt)

            # Mechanics updated via queries
            # Fluid turbulence can be updated
            # Radiation accumulation happens via specific calls

            # Store state history
            if len(self._history) < self._max_history_length:
                self._history.append(self.get_full_state())
            else:
                # Ring buffer behavior
                self._history[int(self._simulation_time * self._update_rate) % self._max_history_length] = self.get_full_state()

    def _update_loop(self) -> None:
        """Background update loop for real-time simulation."""
        while self._is_running:
            start_time = time.time()

            self.update()

            # Sleep to maintain update rate
            elapsed = time.time() - start_time
            sleep_time = max(0, self._dt - elapsed)
            time.sleep(sleep_time)

    def get_conditions_at_position(self, position: Tuple[float, float, float]) -> Dict[str, Any]:
        """
        Get all environmental conditions at a specific position.

        Args:
            position: (x, y, z) coordinates in meters

        Returns:
            Dictionary with all environmental parameters at position
        """
        with self._lock:
            return {
                'position': position,
                'temperature_C': self.temperature.get_temperature(position, unit="C"),
                'temperature_K': self.temperature.get_temperature(position, unit="K"),
                'pressure_bar': self.pressure.get_pressure(position, unit="bar"),
                'pressure_Pa': self.pressure.get_pressure(position, unit="Pa"),
                'wind_velocity_m_s': self.fluid.get_wind(position, unit="m/s").tolist(),
                'gravity_m_s2': self.mechanics.get_gravity(position).tolist(),
                'em_intensity_W_m2': self.radiation.get_em_intensity(position),
                'ionizing_dose_rate_Sv_h': self.radiation.get_ionizing_dose_rate(position),
                'simulation_time_s': self._simulation_time,
            }

    def get_full_state(self) -> Dict[str, Any]:
        """
        Get complete environmental state of all subsystems.

        Returns:
            Dictionary with all subsystem states
        """
        with self._lock:
            return {
                'simulation_time_s': self._simulation_time,
                'update_rate_Hz': self._update_rate,
                'temperature': self.temperature.get_state(),
                'pressure': self.pressure.get_state(),
                'atmosphere': self.atmosphere.get_state(),
                'mechanics': self.mechanics.get_state(),
                'fluid': self.fluid.get_state(),
                'radiation': self.radiation.get_state(),
                'coupling': self.coupling.get_coupling_state(),
                'corrosion': self.get_corrosion_state(),
            }

    def reset_all(self) -> None:
        """Reset all subsystems to default state."""
        with self._lock:
            self.temperature.reset()
            self.pressure.reset()
            self.atmosphere.reset()
            self.mechanics.reset()
            self.fluid.reset()
            self.radiation.reset()
            self.coupling.reset()

            self._simulation_time = 0.0
            self._history = []
            self._corrosion_state = {}

    def set_preset_environment(self, preset: str) -> None:
        """
        Set a preset environmental configuration.

        Args:
            preset: Preset name ("STP", "vacuum", "LEO", "deep_sea", "arctic", "desert")
        """
        with self._lock:
            self.reset_all()

            if preset == "STP":
                # Standard Temperature and Pressure
                self.temperature.set_temperature(25, unit="C")
                self.pressure.set_pressure(1.01325, unit="bar")
                self.atmosphere.set_standard_atmosphere("air")

            elif preset == "vacuum":
                # High vacuum
                self.temperature.set_temperature(25, unit="C")
                self.pressure.set_vacuum_level(1e-6, unit="torr")
                self.atmosphere.set_composition({'N2': 100.0})  # Trace nitrogen

            elif preset == "LEO":
                # Low Earth Orbit
                self.temperature.set_temperature(-100, unit="C")  # Shade temperature
                self.pressure.set_vacuum_level(1e-7, unit="torr")
                self.mechanics.set_gravity(g_factor=0.0)  # Microgravity
                self.radiation.add_em_radiation("UV", 1360, wavelength=200e-9)  # Solar UV
                self.radiation.add_ionizing_radiation("proton", 0.1, energy=100, origin=(0, 0, 0))

            elif preset == "deep_sea":
                # Deep ocean (1000m depth)
                self.temperature.set_temperature(4, unit="C")
                depth = 1000  # meters
                pressure_bar = self.pressure.calculate_hydrostatic_pressure(depth, fluid_density=1025)
                self.pressure.set_pressure(pressure_bar, unit="bar")
                self.atmosphere.set_composition({'O2': 21, 'N2': 79})  # Dissolved gases

            elif preset == "arctic":
                # Arctic conditions
                self.temperature.set_temperature(-40, unit="C")
                self.pressure.set_pressure(1.0, unit="bar")
                self.atmosphere.set_standard_atmosphere("air")
                self.atmosphere.set_humidity(30)  # Low humidity
                self.fluid.set_wind((20, 0, 0), unit="mph")  # 20 mph wind

            elif preset == "desert":
                # Desert conditions
                self.temperature.set_temperature(45, unit="C")
                self.pressure.set_pressure(0.95, unit="bar")  # Lower pressure at altitude
                self.atmosphere.set_standard_atmosphere("air")
                self.atmosphere.set_humidity(10)  # Very low humidity
                self.radiation.add_em_radiation("UV", 50, wavelength=300e-9)  # High UV

            else:
                raise ValueError(f"Unknown preset: {preset}")

    def calculate_material_stress(self, material_properties: Dict[str, float],
                                 position: Tuple[float, float, float]) -> Dict[str, float]:
        """
        Calculate total stress on material at position considering all environmental factors.

        Args:
            material_properties: Dictionary with 'elastic_modulus', 'thermal_expansion', etc.
            position: (x, y, z) coordinates

        Returns:
            Dictionary with stress components
        """
        with self._lock:
            # Get environmental conditions
            temp = self.temperature.get_temperature(position, unit="K")
            pressure = self.pressure.get_pressure(position, unit="Pa")
            wind = self.fluid.get_wind(position, unit="m/s")

            # Thermal stress
            thermal_stress = self.coupling.calculate_thermal_stress(
                temp,
                material_properties.get('elastic_modulus', 200e9),
                reference_temperature=298.15
            )

            # Pressure stress (hydrostatic)
            pressure_stress = pressure

            # Wind-induced stress (dynamic pressure)
            wind_speed = np.linalg.norm(wind)
            dynamic_pressure = 0.5 * self.fluid._fluid_density * wind_speed**2

            return {
                'thermal_stress_Pa': thermal_stress,
                'pressure_stress_Pa': pressure_stress,
                'dynamic_pressure_Pa': dynamic_pressure,
                'total_stress_Pa': thermal_stress + pressure_stress + dynamic_pressure,
            }

    def get_history(self, max_points: int = 1000) -> list:
        """
        Get state history.

        Args:
            max_points: Maximum number of history points to return

        Returns:
            List of historical states (most recent first)
        """
        with self._lock:
            if len(self._history) <= max_points:
                return list(reversed(self._history))
            else:
                # Return evenly spaced samples
                indices = np.linspace(0, len(self._history) - 1, max_points, dtype=int)
                return [self._history[i] for i in reversed(indices)]

    def get_simulation_time(self) -> float:
        """Get current simulation time in seconds."""
        with self._lock:
            return self._simulation_time

    def set_simulation_time(self, time_seconds: float) -> None:
        """Set simulation time (for synchronization)."""
        with self._lock:
            self._simulation_time = time_seconds

    def __repr__(self) -> str:
        """String representation."""
        with self._lock:
            return (f"EnvironmentController(time={self._simulation_time:.3f}s, "
                   f"rate={self._update_rate}Hz, running={self._is_running})")
