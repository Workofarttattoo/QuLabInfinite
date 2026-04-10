#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.

Quantum Sensors Module
Magnetometry, gravimetry, atomic clocks

ECH0 Usage:
```python
from quantum_lab import QuantumLabSimulator

lab = QuantumLabSimulator(num_qubits=8)

# Quantum magnetometry
sensitivity = lab.sensors.magnetometry_sensitivity(num_qubits=8)
print(f"Magnetic field sensitivity: {sensitivity:.2e} T/√Hz")

# Gravimeter precision
precision = lab.sensors.gravimetry_precision(interrogation_time=1.0)
print(f"Gravity measurement precision: {precision:.2e} m/s²")

# Atomic clock stability
stability = lab.sensors.atomic_clock_stability(averaging_time=100)
print(f"Clock stability: {stability:.2e} (fractional frequency)")
```
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class SensorType(Enum):
    """Types of quantum sensors"""
    MAGNETOMETER = "magnetometer"
    GRAVIMETER = "gravimeter"
    GYROSCOPE = "gyroscope"
    ATOMIC_CLOCK = "atomic_clock"
    ELECTRIC_FIELD = "electric_field"
    TEMPERATURE = "temperature"


@dataclass
class SensorSpecs:
    """Quantum sensor specifications"""
    sensor_type: SensorType
    sensitivity: float
    dynamic_range: float
    bandwidth: float
    units: str


class QuantumSensors:
    """
    Quantum sensor modeling and simulation.

    Features:
    - Quantum magnetometry (NV centers, SQUIDs)
    - Atom interferometry gravimeters
    - Quantum gyroscopes
    - Atomic clock stability analysis
    """

    def __init__(self, simulator):
        """
        Initialize quantum sensors module.

        Args:
            simulator: QuantumLabSimulator instance
        """
        self.simulator = simulator

        # Physical constants
        self.h = 6.626e-34  # Planck constant (J·s)
        self.hbar = 1.055e-34  # Reduced Planck constant (J·s)
        self.mu_B = 9.274e-24  # Bohr magneton (J/T)
        self.g = 9.81  # Gravitational acceleration (m/s²)
        self.c = 299792458  # Speed of light (m/s)

    def magnetometry_sensitivity(
        self,
        num_qubits: int = 1,
        measurement_time: float = 1.0,
        method: str = 'ramsey'
    ) -> float:
        """
        Compute quantum magnetometry sensitivity.

        Methods:
        - 'ramsey': Ramsey interferometry (single qubit)
        - 'spin_squeezing': Spin-squeezed states (multi-qubit)
        - 'ghz': GHZ states (Heisenberg limit)

        Sensitivity limits:
        - Standard quantum limit (SQL): δB ~ 1/(γ√(NT))
        - Heisenberg limit (HL): δB ~ 1/(γNT)

        Where:
        - γ = gyromagnetic ratio
        - N = number of qubits
        - T = measurement time

        Args:
            num_qubits: Number of qubits used
            measurement_time: Integration time (seconds)
            method: Sensing method

        Returns:
            Magnetic field sensitivity in T/√Hz
        """
        print(f"⚛️  Quantum Magnetometry")
        print(f"   Method: {method}")
        print(f"   Qubits: {num_qubits}")
        print(f"   Integration time: {measurement_time} s")

        # Gyromagnetic ratio (for NV center in diamond: ~28 GHz/T)
        gamma = 28e9 * 2 * np.pi  # rad/(s·T)

        if method == 'ramsey':
            # Standard quantum limit (shot noise)
            # δB = 1/(γ√(NT))
            sensitivity = 1.0 / (gamma * np.sqrt(num_qubits * measurement_time))

            print(f"   Limit: Standard Quantum Limit (SQL)")

        elif method == 'spin_squeezing':
            # Improved by spin squeezing: ~10 dB improvement
            # δB = ξ/(γ√(NT)) where ξ < 1 is squeezing parameter
            squeezing_param = 0.1  # 10 dB squeezing
            sensitivity = squeezing_param / (gamma * np.sqrt(num_qubits * measurement_time))

            print(f"   Limit: Spin-squeezed (ξ = {squeezing_param})")

        elif method == 'ghz':
            # Heisenberg limit with GHZ states
            # δB = 1/(γNT)
            sensitivity = 1.0 / (gamma * num_qubits * measurement_time)

            print(f"   Limit: Heisenberg Limit (GHZ state)")

        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"   Sensitivity: {sensitivity:.2e} T/√Hz")
        print(f"   ({sensitivity * 1e15:.2f} fT/√Hz)")  # femtoTesla

        # Comparison to classical sensors
        classical_sensitivity = 1e-12  # T/√Hz (typical fluxgate magnetometer)
        improvement = classical_sensitivity / sensitivity

        print(f"\n   Classical comparison:")
        print(f"   Fluxgate magnetometer: ~1 pT/√Hz")
        if improvement > 1:
            print(f"   Quantum advantage: {improvement:.1f}×")
        else:
            print(f"   Needs more qubits for quantum advantage")

        return sensitivity

    def gravimetry_precision(
        self,
        interrogation_time: float = 1.0,
        num_atoms: int = 1e6,
        method: str = 'atom_interferometry'
    ) -> float:
        """
        Compute atom interferometry gravimeter precision.

        Precision:
        δg ~ 1/(keff T² √N)

        Where:
        - keff = effective wavevector
        - T = interrogation time
        - N = number of atoms

        Args:
            interrogation_time: Time between pulses (seconds)
            num_atoms: Number of atoms in ensemble
            method: 'atom_interferometry' or 'bloch_oscillations'

        Returns:
            Gravity measurement precision in m/s² (or µGal where 1 µGal = 10⁻⁸ m/s²)
        """
        print(f"⚛️  Quantum Gravimetry")
        print(f"   Method: {method}")
        print(f"   Atoms: {num_atoms:.1e}")
        print(f"   Interrogation time: {interrogation_time} s")

        if method == 'atom_interferometry':
            # Typical: Rb-87 or Cs-133 atom interferometer
            wavelength = 780e-9  # m (Rb D2 line)
            k_eff = 2 * np.pi / wavelength

            # Two-photon recoil (Raman transitions)
            k_eff *= 2

            # Gravity sensitivity
            # Shot-noise limited: δΦ ~ 1/√N
            # Φ = keff g T²
            # δg = δΦ/(keff T²) = 1/(keff T² √N)

            precision = 1.0 / (k_eff * interrogation_time**2 * np.sqrt(num_atoms))

            print(f"   Effective k: {k_eff:.2e} rad/m")
            print(f"   Precision: {precision:.2e} m/s²")
            print(f"   ({precision * 1e8:.2f} µGal)")

            # Comparison to classical
            classical_precision = 1e-9  # m/s² (1 nGal, state-of-art gravimeter)
            if precision < classical_precision:
                improvement = classical_precision / precision
                print(f"   Quantum advantage: {improvement:.1f}×")

        elif method == 'bloch_oscillations':
            # Bloch oscillations in optical lattice
            precision = 1e-10  # m/s² (simplified)
            print(f"   Precision: {precision:.2e} m/s²")

        return precision

    def gyroscope_sensitivity(
        self,
        num_atoms: int = 1e6,
        interrogation_time: float = 1.0,
        area: float = 1e-4
    ) -> float:
        """
        Compute quantum gyroscope (rotation sensor) sensitivity.

        Sagnac effect:
        ΔΦ = (4πA/λc) Ω

        Where:
        - A = enclosed area
        - λ = de Broglie wavelength
        - Ω = rotation rate

        Sensitivity:
        δΩ ~ λc/(4πA T² √N)

        Args:
            num_atoms: Number of atoms
            interrogation_time: Time (seconds)
            area: Enclosed area (m²)

        Returns:
            Rotation rate sensitivity in rad/s/√Hz
        """
        print(f"⚛️  Quantum Gyroscope (Atom Interferometer)")
        print(f"   Atoms: {num_atoms:.1e}")
        print(f"   Area: {area:.2e} m²")
        print(f"   Time: {interrogation_time} s")

        # De Broglie wavelength (Rb-87 at ~1 µK)
        mass = 1.44e-25  # kg (Rb-87)
        velocity = 1e-2  # m/s (cold atoms)
        wavelength = self.h / (mass * velocity)

        # Sensitivity
        sensitivity = (wavelength * self.c) / (4 * np.pi * area * interrogation_time**2 * np.sqrt(num_atoms))

        print(f"   λ_dB: {wavelength:.2e} m")
        print(f"   Sensitivity: {sensitivity:.2e} rad/s/√Hz")
        print(f"   ({sensitivity * 180/np.pi * 3600:.2e} deg/hr/√Hz)")

        # Earth rotation rate for comparison
        earth_rotation = 7.27e-5  # rad/s
        print(f"\n   Earth rotation: {earth_rotation:.2e} rad/s")
        print(f"   Can detect in {(sensitivity/earth_rotation)**2:.2f} s")

        return sensitivity

    def atomic_clock_stability(
        self,
        averaging_time: float = 1.0,
        num_atoms: int = 1e4,
        clock_transition_freq: float = 9.2e9
    ) -> float:
        """
        Compute atomic clock fractional frequency stability.

        Allan deviation:
        σ_y(τ) = 1/(2πν₀ √(N τ))

        Where:
        - ν₀ = clock transition frequency
        - N = number of atoms
        - τ = averaging time

        Args:
            averaging_time: Averaging time (seconds)
            num_atoms: Number of atoms in ensemble
            clock_transition_freq: Clock transition frequency (Hz)

        Returns:
            Fractional frequency stability σ_y
        """
        print(f"⚛️  Atomic Clock Stability")
        print(f"   Clock: Cs-133 (microwave, 9.2 GHz)")
        print(f"   Atoms: {num_atoms:.1e}")
        print(f"   Averaging time: {averaging_time} s")

        # Allan deviation (shot-noise limited)
        stability = 1.0 / (2 * np.pi * clock_transition_freq * np.sqrt(num_atoms * averaging_time))

        print(f"   Fractional stability: {stability:.2e}")
        print(f"   (σ_y = {stability:.2e})")

        # Compare to best clocks
        print(f"\n   Clock comparisons:")
        print(f"   Cs fountain clock: ~1e-16 (at 1 day)")
        print(f"   Optical lattice clock (Sr): ~1e-18 (state-of-art)")

        if stability < 1e-15:
            print(f"   ✅ Research-grade stability")
        elif stability < 1e-13:
            print(f"   ✅ Commercial-grade stability")
        else:
            print(f"   Needs more atoms or longer averaging")

        return stability

    def quantum_radar_cross_section(
        self,
        target_distance: float = 1000.0,
        num_photons: int = 1000
    ) -> float:
        """
        Compute quantum radar (quantum illumination) advantage.

        Quantum illumination: Use entangled photon pairs
        Signal/Idler photons → ~6 dB advantage over classical radar

        Args:
            target_distance: Target range (m)
            num_photons: Number of entangled photon pairs

        Returns:
            Effective radar cross-section enhancement
        """
        print(f"⚛️  Quantum Radar (Quantum Illumination)")
        print(f"   Target range: {target_distance:.0f} m")
        print(f"   Entangled photon pairs: {num_photons}")

        # Quantum advantage in noisy environment
        # SNR_quantum ~ N_S N_I (signal-idler correlations)
        # SNR_classical ~ N_S N_B (signal-background)

        # Simplified: ~6 dB (4×) advantage
        quantum_advantage_db = 6.0
        quantum_advantage_linear = 10**(quantum_advantage_db / 10.0)

        print(f"   Quantum advantage: {quantum_advantage_db:.1f} dB")
        print(f"   ({quantum_advantage_linear:.1f}× better than classical)")

        # Effective cross-section enhancement
        enhancement = quantum_advantage_linear

        print(f"   Effective RCS enhancement: {enhancement:.2f}×")

        return enhancement

    def nitrogen_vacancy_sensing(
        self,
        field_strength: float = 1e-6,
        decoherence_time: float = 1e-3
    ) -> Dict:
        """
        Model NV center in diamond for sensing applications.

        NV centers are excellent for:
        - Magnetometry (nT sensitivity)
        - Thermometry (mK precision)
        - Electric field sensing
        - Pressure sensing

        Args:
            field_strength: External field (T for magnetic)
            decoherence_time: T2 coherence time (seconds)

        Returns:
            Dict with sensing info
        """
        print(f"⚛️  NV Center Sensor (Diamond)")
        print(f"   Field: {field_strength:.2e} T")
        print(f"   T₂: {decoherence_time:.2e} s")

        # NV center spin Hamiltonian
        # H = D S_z² + γ B·S

        D = 2.87e9  # Zero-field splitting (Hz)
        gamma_nv = 28e9  # Gyromagnetic ratio (Hz/T)

        # Energy splitting due to field
        energy_splitting = gamma_nv * field_strength

        print(f"   Zero-field splitting: {D*1e-9:.2f} GHz")
        print(f"   Field-induced splitting: {energy_splitting*1e-6:.2f} MHz")

        # Sensitivity (Ramsey interferometry)
        sensitivity = 1.0 / (gamma_nv * decoherence_time)

        print(f"   Magnetic sensitivity: {sensitivity:.2e} T")
        print(f"   ({sensitivity*1e9:.2f} nT)")

        # Spatial resolution (for imaging)
        spatial_resolution = 10e-9  # m (10 nm, limited by NV depth)

        print(f"   Spatial resolution: {spatial_resolution*1e9:.0f} nm")
        print(f"   ✅ Ideal for nanoscale sensing")

        return {
            "sensitivity_T": sensitivity,
            "decoherence_time_s": decoherence_time,
            "spatial_resolution_m": spatial_resolution,
            "zero_field_splitting_Hz": D,
            "energy_splitting_Hz": energy_splitting
        }

    def quantum_sensing_comparison(self) -> Dict:
        """
        Compare different quantum sensing modalities.

        Returns:
            Dict with comparison data
        """
        print(f"\n{'='*60}")
        print(f"QUANTUM SENSOR COMPARISON")
        print(f"{'='*60}\n")

        comparison = {}

        # Magnetometry
        print("1️⃣  MAGNETOMETRY")
        comparison['magnetometry'] = {
            'squid': {'sensitivity': 1e-18, 'units': 'T/√Hz', 'comment': 'Cryogenic'},
            'nv_center': {'sensitivity': 1e-12, 'units': 'T/√Hz', 'comment': 'Room temp'},
            'spin_squeezing': {'sensitivity': 1e-15, 'units': 'T/√Hz', 'comment': 'Multi-qubit'}
        }
        for method, specs in comparison['magnetometry'].items():
            print(f"   {method}: {specs['sensitivity']:.1e} {specs['units']} ({specs['comment']})")

        # Gravimetry
        print("\n2️⃣  GRAVIMETRY")
        comparison['gravimetry'] = {
            'classical': {'precision': 1e-9, 'units': 'm/s²', 'comment': 'State-of-art'},
            'atom_interferometry': {'precision': 1e-10, 'units': 'm/s²', 'comment': 'Research-grade'}
        }
        for method, specs in comparison['gravimetry'].items():
            print(f"   {method}: {specs['precision']:.1e} {specs['units']} ({specs['comment']})")

        # Clocks
        print("\n3️⃣  ATOMIC CLOCKS")
        comparison['clocks'] = {
            'cs_fountain': {'stability': 1e-16, 'units': 'fractional', 'comment': 'Primary standard'},
            'optical_lattice': {'stability': 1e-18, 'units': 'fractional', 'comment': 'State-of-art'}
        }
        for method, specs in comparison['clocks'].items():
            print(f"   {method}: {specs['stability']:.1e} {specs['units']} ({specs['comment']})")

        print()
        return comparison


# ========== DEMO ==========

if __name__ == "__main__":
    print("\n" + "="*60)
    print("QUANTUM SENSORS MODULE - DEMONSTRATION")
    print("="*60)

    # Mock simulator
    class MockSimulator:
        def __init__(self):
            self.num_qubits = 10

    sim = MockSimulator()
    sensors = QuantumSensors(sim)

    # Test 1: Magnetometry
    print("\n\n1️⃣  QUANTUM MAGNETOMETRY")
    sens_sql = sensors.magnetometry_sensitivity(
        num_qubits=1,
        measurement_time=1.0,
        method='ramsey'
    )

    sens_ghz = sensors.magnetometry_sensitivity(
        num_qubits=10,
        measurement_time=1.0,
        method='ghz'
    )

    # Test 2: Gravimetry
    print("\n\n2️⃣  ATOM INTERFEROMETRY GRAVIMETER")
    precision = sensors.gravimetry_precision(
        interrogation_time=1.0,
        num_atoms=1e6
    )

    # Test 3: Gyroscope
    print("\n\n3️⃣  QUANTUM GYROSCOPE")
    gyro_sens = sensors.gyroscope_sensitivity(
        num_atoms=1e6,
        interrogation_time=1.0,
        area=1e-4
    )

    # Test 4: Atomic clock
    print("\n\n4️⃣  ATOMIC CLOCK STABILITY")
    stability = sensors.atomic_clock_stability(
        averaging_time=100,
        num_atoms=1e4
    )

    # Test 5: NV center
    print("\n\n5️⃣  NV CENTER DIAMOND SENSOR")
    nv_specs = sensors.nitrogen_vacancy_sensing(
        field_strength=1e-6,
        decoherence_time=1e-3
    )

    # Test 6: Comparison
    print("\n\n6️⃣  SENSOR COMPARISON")
    comparison = sensors.quantum_sensing_comparison()

    print("\n\n✅ Quantum sensors module operational!")
