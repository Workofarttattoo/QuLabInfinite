# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Environmental Simulator - QuLab Infinite
Comprehensive environmental condition modeling with <0.1% error on controlled parameters
"""

from .environmental_sim import (
    EnvironmentalSimulator,
    create_aerogel_simulation,
    create_diamond_anvil_simulation,
    create_leo_simulation,
)
from .environment_controller import EnvironmentController
from .temperature_control import TemperatureControl
from .pressure_control import PressureControl
from .atmosphere_control import AtmosphereControl
from .mechanical_forces import MechanicalForces
from .fluid_flow import FluidFlow
from .radiation_environment import RadiationEnvironment
from .multi_physics_coupling import MultiPhysicsCoupling

__version__ = "1.0.0"
__all__ = [
    "EnvironmentalSimulator",
    "EnvironmentController",
    "TemperatureControl",
    "PressureControl",
    "AtmosphereControl",
    "MechanicalForces",
    "FluidFlow",
    "RadiationEnvironment",
    "MultiPhysicsCoupling",
    "create_aerogel_simulation",
    "create_diamond_anvil_simulation",
    "create_leo_simulation",
]
