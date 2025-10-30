# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Environmental Simulator - Main API
High-level interface for environmental simulations
"""

from typing import Dict, Tuple, Optional, Any
from .environment_controller import EnvironmentController


class EnvironmentalSimulator:
    """
    High-level API for environmental simulations.
    Provides simplified interface to the full environment controller.
    """

    def __init__(self, update_rate: float = 100.0):
        """
        Initialize environmental simulator.

        Args:
            update_rate: Update rate in Hz for real-time simulations
        """
        self.controller = EnvironmentController(update_rate=update_rate)

    def setup_aerogel_test(self, temperature_c: float = -200,
                           pressure_bar: float = 0.001,
                           wind_mph: float = 30) -> None:
        """
        Example: Setup aerogel test conditions.

        Args:
            temperature_c: Temperature in Celsius
            pressure_bar: Pressure in bar
            wind_mph: Wind speed in mph
        """
        self.controller.temperature.set_temperature(temperature_c, unit="C")
        self.controller.pressure.set_pressure(pressure_bar, unit="bar")
        self.controller.fluid.set_wind((wind_mph * 0.44704, 0, 0), unit="m/s")
        self.controller.atmosphere.set_standard_atmosphere("air")

    def setup_diamond_anvil_cell(self, pressure_gpa: float = 100,
                                 temperature_k: float = 3000) -> None:
        """
        Example: Setup diamond anvil cell conditions (extreme pressure/temperature).

        Args:
            pressure_gpa: Pressure in GPa
            temperature_k: Temperature in Kelvin
        """
        self.controller.temperature.set_temperature(temperature_k, unit="K")
        self.controller.pressure.set_pressure(pressure_gpa, unit="GPa")

    def setup_leo_conditions(self, altitude_km: float = 400) -> None:
        """
        Example: Setup Low Earth Orbit conditions.

        Args:
            altitude_km: Orbital altitude in km
        """
        self.controller.set_preset_environment("LEO")

        # Add thermal cycling (day/night)
        def thermal_cycle(time):
            import numpy as np
            # 90 minute orbit period
            period = 5400  # seconds
            phase = (time % period) / period
            if phase < 0.5:  # Sunlight
                return 120 + 273.15  # 120°C
            else:  # Shadow
                return -100 + 273.15  # -100°C

        self.controller.temperature.set_heating_profile(thermal_cycle)

    def get_conditions(self, position: Tuple[float, float, float] = (0, 0, 0)) -> Dict[str, Any]:
        """
        Get environmental conditions at position.

        Args:
            position: (x, y, z) coordinates in meters

        Returns:
            Dictionary with all environmental conditions
        """
        return self.controller.get_conditions_at_position(position)

    def run_simulation(self, duration: float, time_step: Optional[float] = None) -> list:
        """
        Run simulation for specified duration.

        Args:
            duration: Simulation duration in seconds
            time_step: Time step in seconds (None for default)

        Returns:
            List of state snapshots over time
        """
        if time_step is None:
            time_step = 1.0 / self.controller._update_rate

        num_steps = int(duration / time_step)
        history = []

        for _ in range(num_steps):
            self.controller.update(time_step)
            history.append(self.controller.get_full_state())

        return history

    def start_realtime(self) -> None:
        """Start real-time simulation updates."""
        self.controller.start_realtime_updates()

    def stop_realtime(self) -> None:
        """Stop real-time simulation updates."""
        self.controller.stop_realtime_updates()

    def reset(self) -> None:
        """Reset simulator to default state."""
        self.controller.reset_all()

    def get_state(self) -> Dict[str, Any]:
        """Get full simulator state."""
        return self.controller.get_full_state()

    def __repr__(self) -> str:
        """String representation."""
        return f"EnvironmentalSimulator({self.controller})"


# Convenience functions for common scenarios

def create_aerogel_simulation(temp_c: float = -200, pressure_bar: float = 0.001,
                              wind_mph: float = 30) -> EnvironmentalSimulator:
    """
    Create simulator configured for aerogel testing.

    Args:
        temp_c: Temperature in Celsius
        pressure_bar: Pressure in bar
        wind_mph: Wind speed in mph

    Returns:
        Configured EnvironmentalSimulator
    """
    sim = EnvironmentalSimulator()
    sim.setup_aerogel_test(temp_c, pressure_bar, wind_mph)
    return sim


def create_diamond_anvil_simulation(pressure_gpa: float = 100,
                                   temp_k: float = 3000) -> EnvironmentalSimulator:
    """
    Create simulator configured for diamond anvil cell.

    Args:
        pressure_gpa: Pressure in GPa
        temp_k: Temperature in Kelvin

    Returns:
        Configured EnvironmentalSimulator
    """
    sim = EnvironmentalSimulator()
    sim.setup_diamond_anvil_cell(pressure_gpa, temp_k)
    return sim


def create_leo_simulation(altitude_km: float = 400) -> EnvironmentalSimulator:
    """
    Create simulator configured for Low Earth Orbit.

    Args:
        altitude_km: Orbital altitude in km

    Returns:
        Configured EnvironmentalSimulator
    """
    sim = EnvironmentalSimulator()
    sim.setup_leo_conditions(altitude_km)
    return sim


# Example usage
if __name__ == "__main__":
    print("Environmental Simulator - Example Usage\n")

    # Example 1: Aerogel test
    print("=== Aerogel Test ===")
    sim1 = create_aerogel_simulation(temp_c=-200, pressure_bar=0.001, wind_mph=30)
    conditions1 = sim1.get_conditions()
    print(f"Temperature: {conditions1['temperature_C']:.2f}°C")
    print(f"Pressure: {conditions1['pressure_bar']:.6f} bar")
    print(f"Wind speed: {conditions1['wind_velocity_m_s'][0]:.2f} m/s")
    print()

    # Example 2: Diamond anvil cell
    print("=== Diamond Anvil Cell ===")
    sim2 = create_diamond_anvil_simulation(pressure_gpa=100, temp_k=3000)
    conditions2 = sim2.get_conditions()
    print(f"Temperature: {conditions2['temperature_K']:.2f} K")
    print(f"Pressure: {conditions2['pressure_Pa']:.2e} Pa")
    print()

    # Example 3: LEO conditions
    print("=== Low Earth Orbit ===")
    sim3 = create_leo_simulation(altitude_km=400)
    conditions3 = sim3.get_conditions()
    print(f"Temperature: {conditions3['temperature_C']:.2f}°C")
    print(f"Pressure: {conditions3['pressure_Pa']:.2e} Pa (vacuum)")
    print(f"Gravity: {conditions3['gravity_m_s2']} m/s²")
    print(f"EM Intensity: {conditions3['em_intensity_W_m2']:.2f} W/m²")
    print()

    print("Environmental Simulator initialized successfully!")
