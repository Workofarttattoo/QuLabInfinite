#!/usr/bin/env python3
from nist_constants import *

"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

phase_change - Part of Materials Lab
"""

"""
Ice nucleation and crystal growth approximations for QuLab Infinite.

These models are deliberately lightweight; they do not attempt to compete with
computational fluid-dynamics or full molecular simulations.  Instead they
provide order-of-magnitude guidance that can be used by the environmental
tests and by downstream orchestration layers when reasoning about frost
formation on exposed materials.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import math

import numpy as np

from materials_database import MaterialProperties


KB = 1.380649e-23          # Boltzmann constant [J/K]
NA = 6.02214076e23         # Avogadro [1/mol]
WATER_LATENT_HEAT = 333.5e3  # J/kg
ICE_DENSITY = 917.0        # kg/m³
WATER_MOLAR_MASS = 0.018015  # kg/mol


def _supercooling(temperature_k: float, melting_point: float) -> float:
    """Return supercooling ΔT = T_m - T (clamped ≥ 0)."""
    return max(melting_point - temperature_k, 0.0)


@dataclass
class IceNucleationModel:
    """
    Classical nucleation theory inspired approximation.

    Parameters can be tweaked per material by feeding in surface energy or
    contact angle modifiers.  Defaults are tuned for water-based frost on
    engineering substrates.
    """

    material: MaterialProperties
    surface_energy: float = 0.032  # J/m², water on smooth surface
    attempt_frequency: float = 1e11  # 1/s, order of water molecular vibration

    def critical_radius(self, temperature_k: float, melting_point: float) -> float:
        """Return the classical critical radius (in meters)."""
        delta_t = _supercooling(temperature_k, melting_point)
        if delta_t <= 0:
            return np.inf
        delta_g = (ICE_DENSITY * WATER_LATENT_HEAT * delta_t) / melting_point
        return (2.0 * self.surface_energy) / max(delta_g, 1e-12)

    def nucleation_rate(self, temperature_k: float, humidity: float, melting_point: float) -> float:
        """
        Approximate homogeneous nucleation rate [m⁻³·s⁻¹].

        humidity is relative humidity (0-1).  The effective supersaturation is
        assumed to scale linearly with humidity above 0.6.
        """
        delta_t = _supercooling(temperature_k, melting_point)
        if delta_t <= 0 or humidity < 0.4:
            return 0.0

        critical_radius = self.critical_radius(temperature_k, melting_point)
        if not math.isfinite(critical_radius):
            return 0.0

        volume_molecule = WATER_MOLAR_MASS / (ICE_DENSITY * NA)
        delta_g_star = (16 * math.pi * self.surface_energy**3) / (3 * (volume_molecule * KB * temperature_k)**2 + 1e-30)
        supersaturation = max(humidity - 0.6, 0.0) / 0.4  # scale 0-1
        prefactor = self.attempt_frequency * (supersaturation + 1e-6) * 1e20  # coarse scaling
        rate = prefactor * math.exp(-delta_g_star / (KB * temperature_k + 1e-30))
        return float(rate)

    def simulate(self, temperature_k: float, humidity: float, duration_hours: float, melting_point: float) -> Dict[str, float]:
        """Return nucleation metrics over the duration."""
        rate = self.nucleation_rate(temperature_k, humidity, melting_point)
        duration_s = max(duration_hours, 0.0) * 3600.0
        nuclei_per_cm2 = rate * duration_s * 1e-4  # convert m^-3 to cm^-2 assuming 1 mm boundary layer

        return {
            "nucleation_rate_m3s": rate,
            "nuclei_density_cm2": nuclei_per_cm2,
            "critical_radius_nm": self.critical_radius(temperature_k, melting_point) * 1e9,
        }


@dataclass
class IceCrystalGrowthModel:
    """
    Simplified dendritic growth approximation derived from empirical fits.

    Growth velocity is treated as Arrhenius with an activation energy that is
    adjusted based on the substrate's thermal conductivity (better heat removal
    -> faster growth).
    """

    material: MaterialProperties
    activation_energy: float = 45e3  # J/mol
    reference_growth: float = 2e-6   # m/s at ΔT = 5 K

    def growth_velocity(self, temperature_k: float, humidity: float, melting_point: float) -> float:
        delta_t = _supercooling(temperature_k, melting_point)
        if delta_t <= 0:
            return 0.0

        conductivity_factor = max(self.material.thermal_conductivity, 0.05) / 10.0
        humidity_factor = max(humidity, 0.1)
        arrhenius = math.exp(-self.activation_energy / (8.314 * temperature_k))
        velocity = self.reference_growth * (delta_t / 5.0) * conductivity_factor * humidity_factor * arrhenius
        return max(velocity, 0.0)

    def simulate(self, temperature_k: float, humidity: float, duration_hours: float, melting_point: float) -> Dict[str, float]:
        velocity = self.growth_velocity(temperature_k, humidity, melting_point)
        duration_s = max(duration_hours, 0.0) * 3600.0
        thickness = velocity * duration_s

        grain_size = max(5e-6, 2e-5 / max(self.material.confidence, 0.2))

        return {
            "growth_velocity_m_s": velocity,
            "predicted_ice_thickness_mm": thickness * 1e3,
            "estimated_grain_size_um": grain_size * 1e6,
        }


def run_ice_analysis(material: MaterialProperties,
                     temperature_k: float,
                     humidity: float,
                     duration_hours: float) -> Dict[str, float]:
    """
    Convenience helper returning combined nucleation and growth metrics.
    """
    melting_point = material.melting_point if material.melting_point > 0 else 273.15

    nucleation_model = IceNucleationModel(material)
    growth_model = IceCrystalGrowthModel(material)

    nucleation = nucleation_model.simulate(temperature_k, humidity, duration_hours, melting_point)
    growth = growth_model.simulate(temperature_k, humidity, duration_hours, melting_point)

    result = {}
    result.update(nucleation)
    result.update(growth)
    result["supercooling_K"] = _supercooling(temperature_k, melting_point)
    result["relative_humidity"] = humidity

    return result
