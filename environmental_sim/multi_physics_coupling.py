# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Multi-Physics Coupling System
Thermo-mechanical, fluid-structure, electro-thermal, chemo-mechanical coupling
"""

import numpy as np
from typing import Dict, Optional, Callable
import threading


class MultiPhysicsCoupling:
    """
    Multi-physics coupling system for simulating interactions between
    different physical domains.
    """

    def __init__(self):
        """Initialize multi-physics coupling system."""
        self._lock = threading.RLock()

        # Coupling parameters
        self._thermal_expansion_coeff = 0.0  # K⁻¹
        self._stress_temperature_coupling = 0.0  # Pa/K
        self._fluid_structure_coupling = False
        self._electro_thermal_coupling = False

        # Coupling functions (can be overridden)
        self._thermal_stress_func = None
        self._fluid_force_func = None
        self._joule_heating_func = None
        self._chemical_heat_func = None

    def enable_thermo_mechanical_coupling(self, thermal_expansion_coeff: float,
                                         stress_temperature_coupling: float = 0.0) -> None:
        """
        Enable thermo-mechanical coupling (thermal expansion, thermal stress).

        Args:
            thermal_expansion_coeff: Coefficient of thermal expansion (K⁻¹)
            stress_temperature_coupling: Stress dependence on temperature (Pa/K)
        """
        with self._lock:
            self._thermal_expansion_coeff = thermal_expansion_coeff
            self._stress_temperature_coupling = stress_temperature_coupling

    def calculate_thermal_strain(self, temperature_change: float) -> float:
        """
        Calculate thermal strain from temperature change.

        Args:
            temperature_change: Change in temperature (K)

        Returns:
            Thermal strain (dimensionless)
        """
        with self._lock:
            return self._thermal_expansion_coeff * temperature_change

    def calculate_thermal_stress(self, temperature: float,
                                elastic_modulus: float,
                                reference_temperature: float = 298.15) -> float:
        """
        Calculate thermal stress.

        Args:
            temperature: Current temperature (K)
            elastic_modulus: Young's modulus (Pa)
            reference_temperature: Reference temperature (K)

        Returns:
            Thermal stress (Pa)
        """
        with self._lock:
            if self._thermal_stress_func is not None:
                return self._thermal_stress_func(temperature, elastic_modulus, reference_temperature)

            # Default model: σ = E * α * ΔT
            delta_t = temperature - reference_temperature
            thermal_strain = self._thermal_expansion_coeff * delta_t
            stress = elastic_modulus * thermal_strain

            return stress

    def set_thermal_stress_function(self, func: Callable) -> None:
        """
        Set custom thermal stress calculation function.

        Args:
            func: Function taking (temperature, elastic_modulus, ref_temp) -> stress
        """
        with self._lock:
            self._thermal_stress_func = func

    def enable_fluid_structure_coupling(self, enable: bool = True) -> None:
        """
        Enable fluid-structure interaction coupling.

        Args:
            enable: Enable or disable coupling
        """
        with self._lock:
            self._fluid_structure_coupling = enable

    def calculate_fluid_force_on_structure(self, fluid_pressure: float,
                                          fluid_velocity: np.ndarray,
                                          surface_area: float,
                                          surface_normal: np.ndarray) -> np.ndarray:
        """
        Calculate force exerted by fluid on structure.

        Args:
            fluid_pressure: Fluid pressure (Pa)
            fluid_velocity: Fluid velocity vector (m/s)
            surface_area: Surface area (m²)
            surface_normal: Surface normal vector (normalized)

        Returns:
            Force vector (N)
        """
        with self._lock:
            if self._fluid_force_func is not None:
                return self._fluid_force_func(fluid_pressure, fluid_velocity, surface_area, surface_normal)

            # Pressure force (normal to surface)
            pressure_force = fluid_pressure * surface_area * surface_normal

            # Viscous shear force (simplified)
            # F_shear = τ * A, where τ is shear stress
            # For simplicity, assume shear proportional to velocity
            shear_force = 0.001 * fluid_velocity * surface_area  # Simplified

            total_force = pressure_force + shear_force
            return total_force

    def set_fluid_force_function(self, func: Callable) -> None:
        """
        Set custom fluid force calculation function.

        Args:
            func: Function taking (pressure, velocity, area, normal) -> force
        """
        with self._lock:
            self._fluid_force_func = func

    def enable_electro_thermal_coupling(self, enable: bool = True) -> None:
        """
        Enable electro-thermal coupling (Joule heating).

        Args:
            enable: Enable or disable coupling
        """
        with self._lock:
            self._electro_thermal_coupling = enable

    def calculate_joule_heating(self, current: float, resistance: float,
                               volume: float) -> float:
        """
        Calculate Joule heating rate.

        Args:
            current: Electric current (A)
            resistance: Electrical resistance (Ω)
            volume: Volume (m³)

        Returns:
            Volumetric heat generation rate (W/m³)
        """
        with self._lock:
            if self._joule_heating_func is not None:
                return self._joule_heating_func(current, resistance, volume)

            # P = I²R (Joule heating)
            power = current**2 * resistance
            heat_rate = power / volume  # W/m³

            return heat_rate

    def set_joule_heating_function(self, func: Callable) -> None:
        """
        Set custom Joule heating calculation function.

        Args:
            func: Function taking (current, resistance, volume) -> heat_rate
        """
        with self._lock:
            self._joule_heating_func = func

    def calculate_chemical_heat_release(self, reaction_rate: float,
                                       enthalpy_change: float,
                                       volume: float) -> float:
        """
        Calculate heat release from chemical reaction (chemo-thermal coupling).

        Args:
            reaction_rate: Reaction rate (mol/s)
            enthalpy_change: Enthalpy of reaction (J/mol)
            volume: Volume (m³)

        Returns:
            Volumetric heat generation rate (W/m³)
        """
        with self._lock:
            if self._chemical_heat_func is not None:
                return self._chemical_heat_func(reaction_rate, enthalpy_change, volume)

            # Q̇ = r * ΔH (heat release rate)
            power = reaction_rate * enthalpy_change
            heat_rate = power / volume  # W/m³

            return heat_rate

    def set_chemical_heat_function(self, func: Callable) -> None:
        """
        Set custom chemical heat release calculation function.

        Args:
            func: Function taking (reaction_rate, enthalpy, volume) -> heat_rate
        """
        with self._lock:
            self._chemical_heat_func = func

    def calculate_coupled_temperature_change(self, initial_temperature: float,
                                            mechanical_work: float,
                                            chemical_heat: float,
                                            joule_heat: float,
                                            mass: float,
                                            specific_heat: float,
                                            dt: float) -> float:
        """
        Calculate temperature change considering all coupled effects.

        Args:
            initial_temperature: Initial temperature (K)
            mechanical_work: Mechanical work done (J)
            chemical_heat: Chemical heat release (J)
            joule_heat: Joule heating (J)
            mass: Mass (kg)
            specific_heat: Specific heat capacity (J/(kg·K))
            dt: Time step (s)

        Returns:
            New temperature (K)
        """
        with self._lock:
            # Total heat input
            total_heat = mechanical_work + chemical_heat + joule_heat

            # Temperature change: ΔT = Q / (m * c_p)
            delta_t = total_heat / (mass * specific_heat)

            new_temperature = initial_temperature + delta_t

            return new_temperature

    def calculate_piezoresistive_effect(self, base_resistance: float,
                                       stress: float,
                                       gauge_factor: float) -> float:
        """
        Calculate resistance change due to stress (piezoresistive effect).

        Args:
            base_resistance: Base resistance (Ω)
            stress: Applied stress (Pa)
            gauge_factor: Gauge factor (dimensionless)

        Returns:
            New resistance (Ω)
        """
        with self._lock:
            # ΔR/R = GF * (σ/E) for strain gauge
            # Simplified: assume E = 200 GPa
            E = 200e9  # Pa
            strain = stress / E
            relative_change = gauge_factor * strain
            new_resistance = base_resistance * (1 + relative_change)

            return new_resistance

    def calculate_magnetostriction(self, magnetic_field: float,
                                  magnetostrictive_coeff: float) -> float:
        """
        Calculate strain due to magnetic field (magnetostriction).

        Args:
            magnetic_field: Magnetic field strength (T)
            magnetostrictive_coeff: Magnetostrictive coefficient (strain/T²)

        Returns:
            Magnetostrictive strain (dimensionless)
        """
        with self._lock:
            # Simplified: ε = λ * H²
            strain = magnetostrictive_coeff * magnetic_field**2
            return strain

    def get_coupling_state(self) -> dict:
        """
        Get multi-physics coupling state.

        Returns:
            Dictionary with coupling parameters
        """
        with self._lock:
            return {
                'thermo_mechanical_enabled': self._thermal_expansion_coeff != 0.0,
                'thermal_expansion_coeff_K_inv': self._thermal_expansion_coeff,
                'stress_temperature_coupling_Pa_K': self._stress_temperature_coupling,
                'fluid_structure_enabled': self._fluid_structure_coupling,
                'electro_thermal_enabled': self._electro_thermal_coupling,
            }

    def reset(self) -> None:
        """Reset multi-physics coupling to default state."""
        with self._lock:
            self._thermal_expansion_coeff = 0.0
            self._stress_temperature_coupling = 0.0
            self._fluid_structure_coupling = False
            self._electro_thermal_coupling = False
            self._thermal_stress_func = None
            self._fluid_force_func = None
            self._joule_heating_func = None
            self._chemical_heat_func = None
