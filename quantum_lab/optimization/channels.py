"""
Quantum Channel Characterization for Teleportation.

Models realistic quantum channel imperfections:
- Photon loss (fiber attenuation, free space)
- Decoherence (T1, T2 relaxation)
- Noise models (depolarizing, amplitude damping, phase damping)
- Distance-dependent fidelity degradation
- Environmental interference

Used to predict real-world teleportation success rates.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CHANNEL TYPES
# ═══════════════════════════════════════════════════════════════════════════

class ChannelType(Enum):
    """Types of quantum channels for teleportation."""
    FIBER_OPTIC = "fiber_optic"          # Underground or above-ground fiber
    FREE_SPACE = "free_space"            # Line-of-sight (satellite, laser)
    WAVEGUIDE = "waveguide"              # On-chip quantum waveguide
    HYBRID = "hybrid"                    # Mix of fiber and free space


class NoiseModel(Enum):
    """Noise models affecting quantum channels."""
    AMPLITUDE_DAMPING = "amplitude_damping"    # Energy loss
    PHASE_DAMPING = "phase_damping"            # Dephasing
    DEPOLARIZING = "depolarizing"              # Random unitary
    AMPLITUDE_PHASE = "amplitude_phase"        # Combined
    THERMAL = "thermal"                        # Thermal noise


@dataclass
class ChannelCharacteristics:
    """Physical characteristics of a quantum channel."""
    channel_type: ChannelType
    distance_km: float
    noise_model: NoiseModel

    # Loss parameters
    photon_loss_per_km: float = 0.2            # dB/km for fiber
    free_space_loss_coefficient: float = 1.5   # Power law exponent

    # Noise parameters
    amplitude_damping_rate: float = 0.001      # Per microsecond
    phase_damping_rate: float = 0.01           # Per microsecond
    depolarizing_rate: float = 0.002           # Per gate
    thermal_photon_number: float = 0.05        # At room temperature

    # Timing
    quantum_coherence_time_us: float = 100.0   # T2 coherence
    energy_decay_time_us: float = 1000.0       # T1 relaxation
    communication_time_us: float = 1.0         # Time to transmit


@dataclass
class ChannelFidelityAnalysis:
    """Results of channel characterization analysis."""
    channel_type: ChannelType
    distance_km: float
    ideal_fidelity: float = 1.0                # Perfect teleportation
    photon_loss_fidelity: float = 1.0
    noise_fidelity: float = 1.0
    decoherence_fidelity: float = 1.0
    combined_fidelity: float = 1.0

    success_rate_percent: float = 100.0
    limiting_factor: str = "none"              # What causes worst fidelity?
    distance_limit_km: float = float('inf')    # Max distance for >80% fidelity

    fidelity_breakdown: Dict[str, float] = None  # Component contributions
    improvement_opportunities: list = None      # How to improve?


# ═══════════════════════════════════════════════════════════════════════════
# PHOTON LOSS MODELS
# ═══════════════════════════════════════════════════════════════════════════

class PhotonLossModel:
    """Models photon loss in quantum channels."""

    @staticmethod
    def fiber_loss(distance_km: float, attenuation_db_per_km: float = 0.2) -> float:
        """
        Fiber optic loss.

        Standard telecom fiber: ~0.2 dB/km at 1550 nm (C-band)
        Quantum fiber: ~0.3-0.5 dB/km (less optimized)

        Loss in dB = attenuation * distance
        Transmission = 10^(-loss/10)

        Args:
            distance_km: Distance in kilometers
            attenuation_db_per_km: Loss coefficient (dB/km)

        Returns:
            Transmission probability (0-1)
        """
        total_loss_db = attenuation_db_per_km * distance_km
        transmission = 10.0 ** (-total_loss_db / 10.0)
        return transmission

    @staticmethod
    def free_space_loss(
        distance_km: float,
        wavelength_nm: float = 1550.0,
        aperture_diameter_m: float = 0.1
    ) -> float:
        """
        Free space optical loss (satellite, atmospheric).

        Follows inverse square law with additional atmospheric absorption.

        Path loss: L = (4π*R / λ)²

        Args:
            distance_km: Distance in kilometers
            wavelength_nm: Photon wavelength in nanometers
            aperture_diameter_m: Receiver aperture diameter

        Returns:
            Transmission probability (0-1)
        """
        distance_m = distance_km * 1000
        wavelength_m = wavelength_nm * 1e-9

        # Geometric path loss
        path_loss = (4 * np.pi * distance_m / wavelength_m) ** 2
        transmission_geometric = 1.0 / path_loss

        # Diffraction limited receiver
        diffraction_limit = (wavelength_m / (np.pi * aperture_diameter_m / 2)) ** 2
        transmission_diffraction = 1.0 - diffraction_limit

        # Atmospheric absorption (rough model)
        # Clear weather: ~0.5 dB/km, cloudy: ~2 dB/km
        atmospheric_loss_db = 0.5 * distance_km
        transmission_atmospheric = 10.0 ** (-atmospheric_loss_db / 10.0)

        # Combined
        total_transmission = transmission_geometric * transmission_atmospheric

        return min(1.0, total_transmission)

    @staticmethod
    def waveguide_loss(
        distance_mm: float,
        loss_db_per_cm: float = 0.1
    ) -> float:
        """
        On-chip waveguide loss.

        Silicon photonic waveguides: ~0.1 dB/cm typical
        Can be lower (~0.03 dB/cm) with optimization

        Args:
            distance_mm: Distance in millimeters
            loss_db_per_cm: Loss coefficient (dB/cm)

        Returns:
            Transmission probability (0-1)
        """
        distance_cm = distance_mm / 10.0
        total_loss_db = loss_db_per_cm * distance_cm
        transmission = 10.0 ** (-total_loss_db / 10.0)
        return transmission


# ═══════════════════════════════════════════════════════════════════════════
# NOISE MODELS
# ═══════════════════════════════════════════════════════════════════════════

class NoiseModelAnalyzer:
    """Analyzes effects of quantum noise on teleportation fidelity."""

    @staticmethod
    def amplitude_damping(rate: float, time_us: float) -> float:
        """
        Amplitude damping (energy loss).

        Models T1 relaxation: population decays to ground state.
        Fidelity = 1 - rate*time

        Args:
            rate: Damping rate per microsecond
            time_us: Duration in microseconds

        Returns:
            Fidelity reduction (0-1)
        """
        decay = np.exp(-rate * time_us)
        # Energy loss reduces fidelity quadratically
        fidelity_reduction = 1.0 - decay ** 2
        return max(0.0, 1.0 - fidelity_reduction)

    @staticmethod
    def phase_damping(rate: float, time_us: float) -> float:
        """
        Phase damping (dephasing).

        Models T2 relaxation: coherence decays.
        Fidelity = exp(-rate*time)

        Args:
            rate: Dephasing rate per microsecond
            time_us: Duration in microseconds

        Returns:
            Fidelity (0-1)
        """
        dephasing = np.exp(-rate * time_us)
        return dephasing

    @staticmethod
    def depolarizing_noise(
        error_rate: float,
        num_qubits: int = 1,
        num_gates: int = 1
    ) -> float:
        """
        Depolarizing noise (random unitary error).

        Each gate has probability of depolarizing.
        Fidelity = (1 - error_rate)^(num_gates * num_qubits)

        Args:
            error_rate: Error probability per gate
            num_qubits: Number of qubits involved
            num_gates: Number of gates applied

        Returns:
            Fidelity (0-1)
        """
        total_gates = num_gates * num_qubits
        fidelity = (1.0 - error_rate) ** total_gates
        return max(0.0, fidelity)

    @staticmethod
    def thermal_noise(
        temperature_k: float = 300.0,
        photon_energy_meV: float = 0.8  # ~1550 nm
    ) -> float:
        """
        Thermal noise effects.

        At room temperature, thermal photons can introduce noise.
        Thermal photon number ≈ k*T / hf

        Args:
            temperature_k: Temperature in Kelvin
            photon_energy_meV: Photon energy in meV

        Returns:
            Fidelity reduction (0-1)
        """
        boltzmann = 8.617e-5  # meV/K
        thermal_photons = (boltzmann * temperature_k) / photon_energy_meV

        # Thermal photons reduce fidelity
        fidelity_reduction = thermal_photons / (1.0 + thermal_photons)
        return max(0.0, 1.0 - fidelity_reduction)


# ═══════════════════════════════════════════════════════════════════════════
# CHANNEL CHARACTERIZER
# ═══════════════════════════════════════════════════════════════════════════

class ChannelCharacterizer:
    """Characterizes a quantum channel for teleportation."""

    def __init__(self, characteristics: ChannelCharacteristics):
        """Initialize with channel characteristics."""
        self.chars = characteristics

    def analyze_fidelity(self) -> ChannelFidelityAnalysis:
        """
        Analyze channel fidelity degradation.

        Combines all error sources to get final fidelity.
        """
        result = ChannelFidelityAnalysis(
            channel_type=self.chars.channel_type,
            distance_km=self.chars.distance_km,
        )

        # 1. PHOTON LOSS
        if self.chars.channel_type == ChannelType.FIBER_OPTIC:
            loss_fidelity = PhotonLossModel.fiber_loss(
                self.chars.distance_km,
                self.chars.photon_loss_per_km
            )
        elif self.chars.channel_type == ChannelType.FREE_SPACE:
            loss_fidelity = PhotonLossModel.free_space_loss(
                self.chars.distance_km
            )
        else:
            loss_fidelity = 1.0

        result.photon_loss_fidelity = loss_fidelity

        # 2. NOISE
        if self.chars.noise_model == NoiseModel.AMPLITUDE_DAMPING:
            noise_fidelity = NoiseModelAnalyzer.amplitude_damping(
                self.chars.amplitude_damping_rate,
                self.chars.communication_time_us
            )
        elif self.chars.noise_model == NoiseModel.PHASE_DAMPING:
            noise_fidelity = NoiseModelAnalyzer.phase_damping(
                self.chars.phase_damping_rate,
                self.chars.communication_time_us
            )
        elif self.chars.noise_model == NoiseModel.DEPOLARIZING:
            noise_fidelity = NoiseModelAnalyzer.depolarizing_noise(
                self.chars.depolarizing_rate,
                num_qubits=1,
                num_gates=4  # Standard teleportation uses 4 gates
            )
        else:
            noise_fidelity = 1.0

        result.noise_fidelity = noise_fidelity

        # 3. DECOHERENCE
        decoherence_fidelity = NoiseModelAnalyzer.phase_damping(
            1.0 / self.chars.quantum_coherence_time_us,
            self.chars.communication_time_us
        )
        result.decoherence_fidelity = decoherence_fidelity

        # 4. COMBINED
        result.combined_fidelity = (
            result.photon_loss_fidelity *
            result.noise_fidelity *
            result.decoherence_fidelity
        )

        # 5. SUCCESS RATE
        result.success_rate_percent = result.combined_fidelity * 100.0

        # 6. LIMITING FACTOR
        fidelities = {
            "Photon loss": result.photon_loss_fidelity,
            "Noise": result.noise_fidelity,
            "Decoherence": result.decoherence_fidelity,
        }
        result.limiting_factor = min(fidelities, key=fidelities.get)

        # 7. DISTANCE LIMIT (when fidelity drops below 80%)
        result.distance_limit_km = self._calculate_distance_limit()

        # 8. FIDELITY BREAKDOWN
        result.fidelity_breakdown = {
            "Photon loss": 1.0 - result.photon_loss_fidelity,
            "Noise": 1.0 - result.noise_fidelity,
            "Decoherence": 1.0 - result.decoherence_fidelity,
        }

        # 9. IMPROVEMENT OPPORTUNITIES
        result.improvement_opportunities = self._suggest_improvements(result)

        return result

    def _calculate_distance_limit(self, fidelity_threshold: float = 0.80) -> float:
        """Calculate maximum distance for target fidelity."""
        if self.chars.channel_type == ChannelType.FIBER_OPTIC:
            # Exponential loss: F = 10^(-α*d/10)
            # Solve for d: d = -10 * log10(F) / α
            loss_db = -10 * np.log10(fidelity_threshold)
            distance_limit = loss_db / self.chars.photon_loss_per_km
        else:
            distance_limit = float('inf')

        return distance_limit

    def _suggest_improvements(self, result: ChannelFidelityAnalysis) -> list:
        """Suggest improvements to achieve better fidelity."""
        suggestions = []

        if result.photon_loss_fidelity < 0.9:
            suggestions.append("Use quantum repeaters to extend range")
            suggestions.append("Improve fiber quality or use better wavelength")

        if result.noise_fidelity < 0.95:
            suggestions.append("Implement error correction codes")
            suggestions.append("Improve qubit coherence time")

        if result.decoherence_fidelity < 0.98:
            suggestions.append("Speed up protocol execution")
            suggestions.append("Improve quantum memory storage time")

        if not suggestions:
            suggestions.append("Channel performance is excellent")

        return suggestions


# ═══════════════════════════════════════════════════════════════════════════
# CHANNEL SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════

class ChannelScenarios:
    """Pre-configured channel scenarios for common use cases."""

    @staticmethod
    def ideal_laboratory() -> ChannelCharacteristics:
        """Best-case scenario: controlled laboratory environment."""
        return ChannelCharacteristics(
            channel_type=ChannelType.WAVEGUIDE,
            distance_km=0.01,  # On-chip
            noise_model=NoiseModel.AMPLITUDE_DAMPING,
            photon_loss_per_km=0.01,
            amplitude_damping_rate=0.0001,
            phase_damping_rate=0.001,
        )

    @staticmethod
    def metropolitan_fiber() -> ChannelCharacteristics:
        """City-scale: underground fiber, moderate distances."""
        return ChannelCharacteristics(
            channel_type=ChannelType.FIBER_OPTIC,
            distance_km=10.0,
            noise_model=NoiseModel.AMPLITUDE_PHASE,
            photon_loss_per_km=0.2,
            amplitude_damping_rate=0.001,
            phase_damping_rate=0.01,
        )

    @staticmethod
    def long_distance_fiber() -> ChannelCharacteristics:
        """Continental distances: long fiber runs."""
        return ChannelCharacteristics(
            channel_type=ChannelType.FIBER_OPTIC,
            distance_km=100.0,
            noise_model=NoiseModel.AMPLITUDE_PHASE,
            photon_loss_per_km=0.2,
            amplitude_damping_rate=0.002,
            phase_damping_rate=0.02,
        )

    @staticmethod
    def free_space_satellite() -> ChannelCharacteristics:
        """Space: line-of-sight satellite link."""
        return ChannelCharacteristics(
            channel_type=ChannelType.FREE_SPACE,
            distance_km=400.0,  # LEO satellite altitude
            noise_model=NoiseModel.THERMAL,
            photon_loss_per_km=0.0,  # Handled by free_space_loss
            thermal_photon_number=0.1,  # Colder than room temperature
            amplitude_damping_rate=0.0001,
        )

    @staticmethod
    def quantum_internet_node() -> ChannelCharacteristics:
        """Quantum Internet Alliance: optimized for quantum networks."""
        return ChannelCharacteristics(
            channel_type=ChannelType.FIBER_OPTIC,
            distance_km=10.0,
            noise_model=NoiseModel.DEPOLARIZING,
            photon_loss_per_km=0.1,  # Better fiber quality
            depolarizing_rate=0.001,  # Better gates
            quantum_coherence_time_us=1000.0,  # Better qubits
        )
