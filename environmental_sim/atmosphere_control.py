# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Atmosphere Control System
Gas composition, partial pressures, humidity, reactive atmospheres, contamination tracking
"""

import math
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class ContaminantRecord:
    """Tracked contaminant concentration with decay metadata."""
    name: str
    concentration_ppm: float = 0.0
    half_life_hours: Optional[float] = None
    removal_efficiency: float = 0.0  # Fraction removed per hour (0-1)
    total_removed_ppm: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        """Return serializable contaminant profile."""
        return {
            "concentration_ppm": self.concentration_ppm,
            "half_life_hours": self.half_life_hours,
            "removal_efficiency": self.removal_efficiency,
            "total_removed_ppm": self.total_removed_ppm,
        }

    def effective_decay_constant(self) -> float:
        """Combined decay constant in 1/s from half-life and removal efficiency."""
        decay_constant = 0.0
        if self.half_life_hours and self.half_life_hours > 0:
            decay_constant += math.log(2.0) / (self.half_life_hours * 3600.0)

        removal = max(0.0, min(self.removal_efficiency, 0.999999))
        if removal > 0.0:
            # Convert discrete hourly removal efficiency to continuous rate
            decay_constant += -math.log(1.0 - removal) / 3600.0

        return decay_constant


class AtmosphereControl:
    """
    Comprehensive atmosphere control system with gas composition,
    humidity control, and contamination tracking (ppm level).
    """

    # Standard gas compositions
    STANDARD_AIR = {
        'N2': 78.084,
        'O2': 20.946,
        'Ar': 0.934,
        'CO2': 0.041,
    }

    INERT_ATMOSPHERES = {
        'nitrogen': {'N2': 100.0},
        'argon': {'Ar': 100.0},
        'helium': {'He': 100.0},
    }

    def __init__(self):
        """Initialize atmosphere control system."""
        self._lock = threading.RLock()

        # Gas composition (percentage)
        self._composition = self.STANDARD_AIR.copy()

        # Partial pressures (calculated from composition and total pressure)
        self._partial_pressures = {}

        # Humidity
        self._relative_humidity = 50.0  # % RH
        self._dew_point = None  # °C

        # Atmosphere type
        self._atmosphere_type = "air"  # air, inert, oxidizing, reducing

        # Contamination tracking (ppm) with decay metadata
        self._contaminant_records: Dict[str, ContaminantRecord] = {}

        # Trace gases (ppm level)
        self._trace_gases = {}

    @staticmethod
    def _normalize_key(name: str) -> str:
        """Normalize contaminant keys for consistent storage."""
        return name.strip().lower()

    def _get_or_create_contaminant(self, contaminant: str) -> ContaminantRecord:
        """Fetch contaminant record, creating if missing."""
        key = self._normalize_key(contaminant)
        record = self._contaminant_records.get(key)
        if record is None:
            record = ContaminantRecord(name=contaminant.strip())
            self._contaminant_records[key] = record
        elif not record.name:
            record.name = contaminant.strip()
        return record

    def _get_contaminant_record(self, contaminant: str) -> Optional[ContaminantRecord]:
        """Fetch contaminant record without creating."""
        return self._contaminant_records.get(self._normalize_key(contaminant))

    def set_composition(self, composition: Dict[str, float],
                       normalize: bool = True) -> None:
        """
        Set gas composition.

        Args:
            composition: Dictionary of gas:percentage pairs
            normalize: Normalize to 100% if True

        Raises:
            ValueError: If composition is invalid
        """
        with self._lock:
            total = sum(composition.values())

            if normalize:
                if total == 0:
                    raise ValueError("Total composition cannot be zero")
                self._composition = {gas: (frac / total) * 100
                                   for gas, frac in composition.items()}
            else:
                if not (99.9 <= total <= 100.1):
                    raise ValueError(f"Composition must sum to 100%, got {total}%")
                self._composition = composition.copy()

            # Determine atmosphere type
            self._determine_atmosphere_type()

    def get_composition(self, gas: Optional[str] = None) -> float or Dict[str, float]:
        """
        Get gas composition.

        Args:
            gas: Specific gas name (None for full composition)

        Returns:
            Percentage of specified gas or full composition dict
        """
        with self._lock:
            if gas is None:
                return self._composition.copy()
            else:
                return self._composition.get(gas, 0.0)

    def set_standard_atmosphere(self, atmosphere: str) -> None:
        """
        Set to a standard atmosphere.

        Args:
            atmosphere: "air", "nitrogen", "argon", "helium"
        """
        with self._lock:
            if atmosphere == "air":
                self._composition = self.STANDARD_AIR.copy()
                self._atmosphere_type = "air"
            elif atmosphere in self.INERT_ATMOSPHERES:
                self._composition = self.INERT_ATMOSPHERES[atmosphere].copy()
                self._atmosphere_type = "inert"
            else:
                raise ValueError(f"Unknown standard atmosphere: {atmosphere}")

    def set_oxidizing_atmosphere(self, oxygen_percent: float = 100.0) -> None:
        """
        Set oxidizing atmosphere (oxygen-rich).

        Args:
            oxygen_percent: Oxygen percentage (0-100)
        """
        with self._lock:
            if not (0 <= oxygen_percent <= 100):
                raise ValueError("Oxygen percentage must be 0-100")

            remainder = 100 - oxygen_percent
            self._composition = {
                'O2': oxygen_percent,
                'N2': remainder  # Balance with nitrogen
            }
            self._atmosphere_type = "oxidizing"

    def set_reducing_atmosphere(self, hydrogen_percent: float = 5.0,
                               balance_gas: str = "N2") -> None:
        """
        Set reducing atmosphere (hydrogen-containing).

        Args:
            hydrogen_percent: Hydrogen percentage (typically 1-10%)
            balance_gas: Balance gas (N2, Ar, He)
        """
        with self._lock:
            if not (0 < hydrogen_percent < 100):
                raise ValueError("Hydrogen percentage must be 0-100")

            self._composition = {
                'H2': hydrogen_percent,
                balance_gas: 100 - hydrogen_percent
            }
            self._atmosphere_type = "reducing"

    def calculate_partial_pressures(self, total_pressure: float,
                                    unit: str = "bar") -> Dict[str, float]:
        """
        Calculate partial pressures of all gases.

        Args:
            total_pressure: Total pressure
            unit: Pressure unit

        Returns:
            Dictionary of gas:partial_pressure pairs in specified unit
        """
        with self._lock:
            partial_pressures = {}
            for gas, percentage in self._composition.items():
                partial_pressures[gas] = total_pressure * (percentage / 100.0)

            self._partial_pressures = partial_pressures
            return partial_pressures

    def get_partial_pressure(self, gas: str, total_pressure: float,
                            unit: str = "bar") -> float:
        """
        Get partial pressure of a specific gas.

        Args:
            gas: Gas name
            total_pressure: Total pressure
            unit: Pressure unit

        Returns:
            Partial pressure in specified unit
        """
        with self._lock:
            percentage = self._composition.get(gas, 0.0)
            return total_pressure * (percentage / 100.0)

    def set_humidity(self, relative_humidity: float,
                    temperature_c: Optional[float] = None) -> None:
        """
        Set relative humidity.

        Args:
            relative_humidity: Relative humidity (0-100%)
            temperature_c: Temperature in Celsius for dew point calculation

        Raises:
            ValueError: If humidity is out of range
        """
        with self._lock:
            if not (0 <= relative_humidity <= 100):
                raise ValueError("Relative humidity must be 0-100%")

            self._relative_humidity = relative_humidity

            # Calculate dew point if temperature provided
            if temperature_c is not None:
                self._dew_point = self._calculate_dew_point(relative_humidity, temperature_c)

    def get_humidity(self) -> float:
        """Get relative humidity in percent."""
        with self._lock:
            return self._relative_humidity

    def get_dew_point(self) -> Optional[float]:
        """Get dew point in Celsius (None if not calculated)."""
        with self._lock:
            return self._dew_point

    def add_contaminant(
        self,
        contaminant: str,
        concentration_ppm: float,
        *,
        half_life_hours: Optional[float] = None,
        removal_efficiency: Optional[float] = None,
        accumulate: bool = True
    ) -> None:
        """
        Add or update contaminant concentration.

        Args:
            contaminant: Contaminant name
            concentration_ppm: Concentration in parts per million
            half_life_hours: Optional half-life for exponential decay
            removal_efficiency: Optional hourly removal efficiency (0-1)
            accumulate: Accumulate concentration instead of replacing
        """
        with self._lock:
            if concentration_ppm < 0:
                raise ValueError("Concentration cannot be negative")

            record = self._get_or_create_contaminant(contaminant)
            if accumulate:
                record.concentration_ppm += concentration_ppm
            else:
                record.concentration_ppm = concentration_ppm

            if half_life_hours is not None:
                if half_life_hours <= 0:
                    raise ValueError("Half-life must be positive when provided")
                record.half_life_hours = float(half_life_hours)

            if removal_efficiency is not None:
                if not (0.0 <= removal_efficiency <= 1.0):
                    raise ValueError("Removal efficiency must be between 0 and 1")
                record.removal_efficiency = float(removal_efficiency)

    def get_contaminant(self, contaminant: str) -> float:
        """
        Get contaminant concentration.

        Args:
            contaminant: Contaminant name

        Returns:
            Concentration in ppm (0 if not present)
        """
        with self._lock:
            record = self._get_contaminant_record(contaminant)
            return record.concentration_ppm if record else 0.0

    def get_all_contaminants(self) -> Dict[str, float]:
        """Get all contaminants and their concentrations."""
        with self._lock:
            return {record.name or key: record.concentration_ppm
                    for key, record in self._contaminant_records.items()}

    def clear_contaminants(self) -> None:
        """Remove all contaminants."""
        with self._lock:
            self._contaminant_records = {}

    def configure_contaminant_decay(
        self,
        contaminant: str,
        *,
        half_life_hours: Optional[float] = None,
        removal_efficiency: Optional[float] = None
    ) -> None:
        """
        Configure decay characteristics for a contaminant without altering concentration.
        """
        with self._lock:
            record = self._get_or_create_contaminant(contaminant)
            if half_life_hours is not None:
                if half_life_hours <= 0:
                    raise ValueError("Half-life must be positive when provided")
                record.half_life_hours = float(half_life_hours)
            if removal_efficiency is not None:
                if not (0.0 <= removal_efficiency <= 1.0):
                    raise ValueError("Removal efficiency must be between 0 and 1")
                record.removal_efficiency = float(removal_efficiency)

    def update_contaminants(self, delta_time_s: float) -> None:
        """
        Apply exponential decay to contaminants over the provided timestep.
        """
        if delta_time_s <= 0:
            return

        with self._lock:
            for record in self._contaminant_records.values():
                if record.concentration_ppm <= 0:
                    continue

                decay_constant = record.effective_decay_constant()
                if decay_constant <= 0:
                    continue

                factor = math.exp(-decay_constant * delta_time_s)
                factor = max(0.0, min(1.0, factor))

                previous = record.concentration_ppm
                record.concentration_ppm *= factor
                record.total_removed_ppm += previous - record.concentration_ppm

                # Numerical cleanup for very small concentrations
                if record.concentration_ppm < 1e-9:
                    record.total_removed_ppm += record.concentration_ppm
                    record.concentration_ppm = 0.0

    def get_contaminant_profile(self, contaminant: str) -> Dict[str, float]:
        """Return detailed profile for a contaminant."""
        with self._lock:
            record = self._get_contaminant_record(contaminant)
            if record is None:
                return {}
            profile = record.as_dict()
            profile["name"] = record.name
            return profile

    def get_all_contaminant_profiles(self) -> Dict[str, Dict[str, float]]:
        """Return detailed profiles for all contaminants."""
        with self._lock:
            return {
                record.name or key: {**record.as_dict(), "name": record.name or key}
                for key, record in self._contaminant_records.items()
            }

    def add_trace_gas(self, gas: str, concentration_ppm: float) -> None:
        """
        Add trace gas at ppm level.

        Args:
            gas: Gas name
            concentration_ppm: Concentration in parts per million
        """
        with self._lock:
            if concentration_ppm < 0:
                raise ValueError("Concentration cannot be negative")

            self._trace_gases[gas] = concentration_ppm

    def get_atmosphere_type(self) -> str:
        """
        Get atmosphere type classification.

        Returns:
            "air", "inert", "oxidizing", or "reducing"
        """
        with self._lock:
            return self._atmosphere_type

    def is_breathable(self, partial_pressure_o2: float = None) -> bool:
        """
        Check if atmosphere is breathable for humans.

        Args:
            partial_pressure_o2: O2 partial pressure in bar (calculated if None)

        Returns:
            True if breathable (O2 partial pressure in safe range)
        """
        with self._lock:
            if partial_pressure_o2 is None:
                o2_percent = self._composition.get('O2', 0.0)
                # Assume 1 atm total pressure
                partial_pressure_o2 = 1.01325 * (o2_percent / 100.0)

            # Safe range: 0.16 to 0.5 bar O2 partial pressure
            return 0.16 <= partial_pressure_o2 <= 0.5

    def calculate_mean_molecular_weight(self) -> float:
        """
        Calculate mean molecular weight of atmosphere.

        Returns:
            Mean molecular weight in g/mol
        """
        # Molecular weights of common gases (g/mol)
        molecular_weights = {
            'N2': 28.014,
            'O2': 31.998,
            'Ar': 39.948,
            'CO2': 44.01,
            'H2': 2.016,
            'He': 4.003,
            'Ne': 20.180,
            'CH4': 16.043,
            'H2O': 18.015,
        }

        with self._lock:
            total_weight = 0.0
            for gas, percentage in self._composition.items():
                weight = molecular_weights.get(gas, 29.0)  # Default to air
                total_weight += weight * (percentage / 100.0)

            return total_weight

    def _calculate_dew_point(self, rh: float, temp_c: float) -> float:
        """
        Calculate dew point using Magnus formula.

        Args:
            rh: Relative humidity (%)
            temp_c: Temperature (°C)

        Returns:
            Dew point in °C
        """
        # Magnus formula constants
        a = 17.27
        b = 237.7

        alpha = ((a * temp_c) / (b + temp_c)) + np.log(rh / 100.0)
        dew_point = (b * alpha) / (a - alpha)

        return dew_point

    def _determine_atmosphere_type(self) -> None:
        """Automatically determine atmosphere type from composition."""
        o2 = self._composition.get('O2', 0.0)
        h2 = self._composition.get('H2', 0.0)
        n2 = self._composition.get('N2', 0.0)
        ar = self._composition.get('Ar', 0.0)
        he = self._composition.get('He', 0.0)

        # Inert atmosphere (>95% inert gas)
        if (n2 > 95 or ar > 95 or he > 95) and o2 < 1 and h2 < 1:
            self._atmosphere_type = "inert"
        # Oxidizing (high oxygen)
        elif o2 > 50:
            self._atmosphere_type = "oxidizing"
        # Reducing (hydrogen present)
        elif h2 > 0.1:
            self._atmosphere_type = "reducing"
        # Standard air
        elif 19 < o2 < 22 and 77 < n2 < 80:
            self._atmosphere_type = "air"
        else:
            self._atmosphere_type = "custom"

    def get_state(self) -> dict:
        """
        Get complete atmosphere state.

        Returns:
            Dictionary with all atmosphere parameters
        """
        with self._lock:
            return {
                'composition': self._composition.copy(),
                'atmosphere_type': self._atmosphere_type,
                'relative_humidity': self._relative_humidity,
                'dew_point_C': self._dew_point,
                'contaminants_ppm': self.get_all_contaminants(),
                'contaminant_profiles': self.get_all_contaminant_profiles(),
                'trace_gases_ppm': self._trace_gases.copy(),
                'mean_molecular_weight': self.calculate_mean_molecular_weight(),
                'is_breathable': self.is_breathable(),
            }

    def reset(self) -> None:
        """Reset atmosphere control to standard air."""
        with self._lock:
            self._composition = self.STANDARD_AIR.copy()
            self._partial_pressures = {}
            self._relative_humidity = 50.0
            self._dew_point = None
            self._atmosphere_type = "air"
            self._contaminant_records = {}
            self._trace_gases = {}
