# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Radiation Environment System
EM radiation (UV/visible/IR/microwave/radio), ionizing radiation, dose rates, shielding
"""

import numpy as np
from typing import Dict, Optional, Tuple
import threading


class RadiationEnvironment:
    """
    Comprehensive radiation environment system with electromagnetic
    and ionizing radiation sources, dose tracking, and shielding.
    """

    # Physical constants
    SPEED_OF_LIGHT = 2.998e8  # m/s
    PLANCK_CONSTANT = 6.626e-34  # J·s

    def __init__(self):
        """Initialize radiation environment system."""
        self._lock = threading.RLock()

        # Electromagnetic radiation sources
        self._em_sources = []

        # Ionizing radiation sources
        self._ionizing_sources = []

        # Accumulated dose
        self._accumulated_dose = 0.0  # Sieverts (Sv)

        # Shielding materials
        self._shields = []

    def add_em_radiation(self, radiation_type: str, intensity: float,
                        wavelength: Optional[float] = None,
                        frequency: Optional[float] = None,
                        direction: Tuple[float, float, float] = (0, 0, -1)) -> int:
        """
        Add electromagnetic radiation source.

        Args:
            radiation_type: "UV", "visible", "IR", "microwave", "radio"
            intensity: Intensity in W/m²
            wavelength: Wavelength in meters (optional if frequency provided)
            frequency: Frequency in Hz (optional if wavelength provided)
            direction: Direction vector (normalized)

        Returns:
            Radiation source ID
        """
        with self._lock:
            # Calculate wavelength/frequency if only one provided
            if wavelength is None and frequency is None:
                # Use typical wavelength for type
                wavelength_defaults = {
                    'UV': 200e-9,  # 200 nm
                    'visible': 550e-9,  # 550 nm (green)
                    'IR': 10e-6,  # 10 μm
                    'microwave': 0.01,  # 1 cm
                    'radio': 1.0,  # 1 m
                }
                wavelength = wavelength_defaults.get(radiation_type, 550e-9)
                frequency = self.SPEED_OF_LIGHT / wavelength
            elif wavelength is not None and frequency is None:
                frequency = self.SPEED_OF_LIGHT / wavelength
            elif wavelength is None and frequency is not None:
                wavelength = self.SPEED_OF_LIGHT / frequency

            # Normalize direction
            dir_array = np.array(direction, dtype=np.float64)
            dir_norm = np.linalg.norm(dir_array)
            if dir_norm > 0:
                direction_normalized = dir_array / dir_norm
            else:
                raise ValueError("Direction vector cannot be zero")

            # Calculate photon energy
            photon_energy = self.PLANCK_CONSTANT * frequency  # Joules

            source_id = len(self._em_sources)
            self._em_sources.append({
                'id': source_id,
                'type': radiation_type,
                'intensity': intensity,
                'wavelength': wavelength,
                'frequency': frequency,
                'photon_energy': photon_energy,
                'direction': direction_normalized,
            })
            return source_id

    def add_ionizing_radiation(self, radiation_type: str, dose_rate: float,
                              energy: float, origin: Tuple[float, float, float],
                              activity: Optional[float] = None) -> int:
        """
        Add ionizing radiation source.

        Args:
            radiation_type: "X-ray", "gamma", "neutron", "proton", "electron", "alpha"
            dose_rate: Dose rate in Sv/h at 1 meter
            energy: Particle/photon energy in MeV
            origin: Source position (x, y, z)
            activity: Radioactive activity in Bq (optional)

        Returns:
            Radiation source ID
        """
        with self._lock:
            source_id = len(self._ionizing_sources)
            self._ionizing_sources.append({
                'id': source_id,
                'type': radiation_type,
                'dose_rate': dose_rate,
                'energy': energy,
                'origin': np.array(origin, dtype=np.float64),
                'activity': activity,
            })
            return source_id

    def get_em_intensity(self, position: Optional[Tuple[float, float, float]] = None,
                        wavelength_range: Optional[Tuple[float, float]] = None) -> float:
        """
        Get total electromagnetic intensity at position.

        Args:
            position: (x, y, z) coordinates (None for unattenuated)
            wavelength_range: (min, max) wavelength range in meters (None for all)

        Returns:
            Total intensity in W/m²
        """
        with self._lock:
            total_intensity = 0.0

            for source in self._em_sources:
                # Filter by wavelength range
                if wavelength_range is not None:
                    wl_min, wl_max = wavelength_range
                    if not (wl_min <= source['wavelength'] <= wl_max):
                        continue

                intensity = source['intensity']

                # Apply attenuation if position specified
                if position is not None:
                    # Simplified atmospheric attenuation
                    attenuation = self._calculate_em_attenuation(source, position)
                    intensity *= attenuation

                # Apply shielding
                if position is not None:
                    shielding_factor = self._calculate_shielding_factor(source, position)
                    intensity *= shielding_factor

                total_intensity += intensity

            return total_intensity

    def get_ionizing_dose_rate(self, position: Tuple[float, float, float]) -> float:
        """
        Get ionizing radiation dose rate at position.

        Args:
            position: (x, y, z) coordinates

        Returns:
            Dose rate in Sv/h
        """
        with self._lock:
            total_dose_rate = 0.0
            pos = np.array(position, dtype=np.float64)

            for source in self._ionizing_sources:
                # Distance from source
                r = np.linalg.norm(pos - source['origin'])

                if r < 0.01:  # Too close to source
                    continue

                # Inverse square law: dose_rate(r) = dose_rate(1m) / r²
                dose_rate = source['dose_rate'] / (r**2)

                # Apply shielding
                shielding_factor = self._calculate_ionizing_shielding(source, position)
                dose_rate *= shielding_factor

                total_dose_rate += dose_rate

            return total_dose_rate

    def accumulate_dose(self, position: Tuple[float, float, float],
                       duration: float) -> float:
        """
        Accumulate radiation dose over time.

        Args:
            position: (x, y, z) coordinates
            duration: Exposure duration in hours

        Returns:
            Accumulated dose in Sv
        """
        with self._lock:
            dose_rate = self.get_ionizing_dose_rate(position)
            dose = dose_rate * duration
            self._accumulated_dose += dose
            return dose

    def get_accumulated_dose(self) -> float:
        """
        Get total accumulated dose.

        Returns:
            Total dose in Sv
        """
        with self._lock:
            return self._accumulated_dose

    def reset_dose(self) -> None:
        """Reset accumulated dose to zero."""
        with self._lock:
            self._accumulated_dose = 0.0

    def add_shield(self, material: str, thickness: float,
                  position: Tuple[float, float, float],
                  normal: Tuple[float, float, float]) -> int:
        """
        Add shielding material.

        Args:
            material: Material name (e.g., "lead", "concrete", "aluminum")
            thickness: Shield thickness in meters
            position: Shield position (x, y, z)
            normal: Shield normal vector (normalized)

        Returns:
            Shield ID
        """
        with self._lock:
            # Normalize normal
            normal_array = np.array(normal, dtype=np.float64)
            normal_norm = np.linalg.norm(normal_array)
            if normal_norm > 0:
                normal_normalized = normal_array / normal_norm
            else:
                raise ValueError("Normal vector cannot be zero")

            # Material properties (attenuation coefficients)
            attenuation_coefficients = {
                'lead': {'EM': 0.5, 'gamma': 1.2, 'X-ray': 1.5},
                'concrete': {'EM': 0.1, 'gamma': 0.3, 'X-ray': 0.2},
                'aluminum': {'EM': 0.2, 'gamma': 0.15, 'X-ray': 0.3},
                'steel': {'EM': 0.3, 'gamma': 0.5, 'X-ray': 0.6},
                'water': {'EM': 0.05, 'gamma': 0.1, 'X-ray': 0.08, 'neutron': 0.5},
                'polyethylene': {'EM': 0.03, 'neutron': 0.4},
            }

            shield_id = len(self._shields)
            self._shields.append({
                'id': shield_id,
                'material': material,
                'thickness': thickness,
                'position': np.array(position, dtype=np.float64),
                'normal': normal_normalized,
                'attenuation': attenuation_coefficients.get(material, {'EM': 0.1, 'gamma': 0.2}),
            })
            return shield_id

    def calculate_photodegradation(self, material: str,
                                  exposure_time: float) -> float:
        """
        Calculate photodegradation due to UV/radiation exposure.

        Args:
            material: Material name
            exposure_time: Exposure time in hours

        Returns:
            Degradation factor (0-1, 0=no degradation, 1=complete degradation)
        """
        # Simplified photodegradation model
        # Real implementation would use material-specific degradation rates

        with self._lock:
            # Get UV intensity
            uv_intensity = self.get_em_intensity(wavelength_range=(100e-9, 400e-9))

            # Material-specific degradation rates (arbitrary units)
            degradation_rates = {
                'polymer': 0.01,
                'plastic': 0.008,
                'paint': 0.015,
                'rubber': 0.012,
                'fabric': 0.02,
            }

            rate = degradation_rates.get(material, 0.005)
            degradation = 1 - np.exp(-rate * uv_intensity * exposure_time)

            return min(degradation, 1.0)

    def _calculate_em_attenuation(self, source: dict,
                                 position: Tuple[float, float, float]) -> float:
        """
        Calculate electromagnetic attenuation (simplified atmospheric model).

        Args:
            source: EM radiation source dictionary
            position: Target position

        Returns:
            Attenuation factor (0-1)
        """
        # Simplified Beer-Lambert law
        # Real implementation would include atmospheric composition, altitude, etc.

        # Attenuation coefficients (m⁻¹) for different radiation types
        attenuation_per_meter = {
            'UV': 0.001,
            'visible': 0.0001,
            'IR': 0.0005,
            'microwave': 0.00001,
            'radio': 0.000001,
        }

        coeff = attenuation_per_meter.get(source['type'], 0.0001)
        distance = np.linalg.norm(position)

        attenuation = np.exp(-coeff * distance)
        return attenuation

    def _calculate_shielding_factor(self, source: dict,
                                   position: Tuple[float, float, float]) -> float:
        """
        Calculate EM radiation shielding factor.

        Args:
            source: Radiation source dictionary
            position: Target position

        Returns:
            Shielding factor (0-1, 0=complete blocking, 1=no shielding)
        """
        # Simplified shielding calculation
        shielding_factor = 1.0

        for shield in self._shields:
            # Check if shield is between source and position
            # Simplified: just apply attenuation based on thickness
            attenuation_coeff = shield['attenuation'].get('EM', 0.1)
            factor = np.exp(-attenuation_coeff * shield['thickness'])
            shielding_factor *= factor

        return shielding_factor

    def _calculate_ionizing_shielding(self, source: dict,
                                     position: Tuple[float, float, float]) -> float:
        """
        Calculate ionizing radiation shielding factor.

        Args:
            source: Ionizing source dictionary
            position: Target position

        Returns:
            Shielding factor (0-1)
        """
        shielding_factor = 1.0

        for shield in self._shields:
            radiation_type = source['type']
            attenuation_coeff = shield['attenuation'].get(radiation_type, 0.1)
            factor = np.exp(-attenuation_coeff * shield['thickness'])
            shielding_factor *= factor

        return shielding_factor

    def get_state(self) -> dict:
        """
        Get complete radiation environment state.

        Returns:
            Dictionary with all radiation parameters
        """
        with self._lock:
            return {
                'num_em_sources': len(self._em_sources),
                'num_ionizing_sources': len(self._ionizing_sources),
                'total_em_intensity_W_m2': self.get_em_intensity(),
                'accumulated_dose_Sv': self._accumulated_dose,
                'num_shields': len(self._shields),
            }

    def reset(self) -> None:
        """Reset radiation environment to default state."""
        with self._lock:
            self._em_sources = []
            self._ionizing_sources = []
            self._accumulated_dose = 0.0
            self._shields = []
