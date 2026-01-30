"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

BIOMEDICAL ENGINEERING LAB
Production-ready biomedical engineering calculations for biomechanics, drug delivery, medical devices, and tissue engineering.
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy import constants, signal, optimize, integrate
from typing import Dict, Tuple, List, Optional, Callable
import warnings

@dataclass
class BiomedicalEngineeringLab:
    """
    Comprehensive biomedical engineering calculations including:
    - Biomechanics calculations
    - Drug delivery modeling
    - Medical device analysis
    - Tissue engineering parameters
    - Biosignal processing
    - Implant design criteria
    """

    # Physiological constants
    body_temperature: float = 310.15  # K (37°C)
    blood_density: float = 1060  # kg/m³
    blood_viscosity: float = 3.5e-3  # Pa·s at 37°C
    water_content: float = 0.6  # Typical body water fraction

    # Biomechanical properties (bone)
    bone_density: float = 1900  # kg/m³
    bone_youngs_modulus: float = 17e9  # Pa (cortical bone)
    bone_yield_strength: float = 130e6  # Pa
    bone_poissons_ratio: float = 0.3

    # Soft tissue properties
    tissue_density: float = 1040  # kg/m³
    tissue_youngs_modulus: float = 10e3  # Pa (varies widely)
    tissue_permeability: float = 1e-14  # m² (for interstitial flow)

    # Universal constants
    gas_constant: float = constants.R  # J/(mol·K)
    avogadro: float = constants.Avogadro
    boltzmann: float = constants.k

    def __post_init__(self):
        """Initialize derived properties"""
        self.diffusion_coefficient_oxygen = 2.1e-9  # m²/s in tissue
        self.diffusion_coefficient_glucose = 6.7e-10  # m²/s in tissue

    def stress_strain_bone(self, stress: float, strain_rate: float = 0.001) -> Dict[str, float]:
        """
        Calculate bone mechanical response with viscoelastic effects

        Args:
            stress: Applied stress in Pa
            strain_rate: Strain rate in 1/s

        Returns:
            Dictionary with strain, safety factor, and viscoelastic modulus
        """
        # Basic elastic strain
        elastic_strain = stress / self.bone_youngs_modulus

        # Viscoelastic correction (simplified)
        viscosity_factor = 1 + 0.1 * np.log10(strain_rate / 0.001)
        effective_modulus = self.bone_youngs_modulus * viscosity_factor

        # Total strain
        total_strain = stress / effective_modulus

        # Safety factor
        safety_factor = self.bone_yield_strength / stress if stress > 0 else float('inf')

        return {
            'elastic_strain': elastic_strain,
            'total_strain': total_strain,
            'effective_modulus': effective_modulus,
            'safety_factor': safety_factor,
            'fracture_risk': 'high' if safety_factor < 2 else 'low'
        }

    def wolff_law_remodeling(self, strain: float, time_days: int) -> float:
        """
        Calculate bone remodeling response using Wolff's Law

        Args:
            strain: Mechanical strain (dimensionless)
            time_days: Time period in days

        Returns:
            Bone density change as fraction
        """
        # Mechanostat theory thresholds
        disuse_threshold = 0.0005
        overload_threshold = 0.003

        if strain < disuse_threshold:
            # Bone resorption
            rate = -0.001  # 0.1% per day loss
        elif strain > overload_threshold:
            # Bone formation
            rate = 0.002  # 0.2% per day gain
        else:
            # Equilibrium zone
            rate = 0

        # Calculate total change
        density_change = rate * time_days

        # Limit to realistic bounds
        return np.clip(density_change, -0.3, 0.5)

    def joint_force_analysis(self, body_weight: float, activity: str = 'walking') -> Dict[str, float]:
        """
        Calculate joint reaction forces for different activities

        Args:
            body_weight: Body weight in N
            activity: Type of activity ('standing', 'walking', 'running', 'jumping')

        Returns:
            Dictionary with joint forces for hip, knee, and ankle
        """
        # Joint force multipliers (times body weight)
        multipliers = {
            'standing': {'hip': 0.3, 'knee': 0.5, 'ankle': 1.0},
            'walking': {'hip': 2.5, 'knee': 2.0, 'ankle': 1.5},
            'running': {'hip': 5.0, 'knee': 7.0, 'ankle': 8.0},
            'jumping': {'hip': 6.0, 'knee': 10.0, 'ankle': 12.0}
        }

        if activity not in multipliers:
            activity = 'standing'

        forces = multipliers[activity]

        return {
            'hip_force': body_weight * forces['hip'],
            'knee_force': body_weight * forces['knee'],
            'ankle_force': body_weight * forces['ankle'],
            'activity': activity
        }

    def muscle_force_length(self, length_ratio: float) -> float:
        """
        Calculate muscle force using force-length relationship

        Args:
            length_ratio: Current length / optimal length

        Returns:
            Force as fraction of maximum (0-1)
        """
        # Gaussian-like force-length curve
        optimal = 1.0
        width = 0.5

        force = np.exp(-((length_ratio - optimal) / width)**2)

        return np.clip(force, 0, 1)

    def hill_muscle_model(self, velocity: float, max_velocity: float,
                         max_force: float) -> float:
        """
        Calculate muscle force using Hill's muscle model

        Args:
            velocity: Contraction velocity in m/s (negative for shortening)
            max_velocity: Maximum shortening velocity in m/s
            max_force: Maximum isometric force in N

        Returns:
            Force in N
        """
        # Hill's equation parameters
        a = 0.25 * max_force  # Shape factor
        b = 0.25 * max_velocity

        if velocity < 0:  # Shortening
            force = max_force * (b - velocity) / (b - velocity/a)
        else:  # Lengthening
            force = max_force * (1.8 - 0.8 / (1 + velocity/max_velocity))

        return np.clip(force, 0, 1.8 * max_force)

    def drug_diffusion_tissue(self, concentration_surface: float,
                            distance: float, time: float,
                            diffusivity: Optional[float] = None) -> float:
        """
        Calculate drug concentration in tissue using Fick's law

        C(x,t) = C0 * erfc(x / (2*√(D*t)))

        Args:
            concentration_surface: Surface concentration in mol/m³
            distance: Distance from surface in meters
            time: Time in seconds
            diffusivity: Diffusion coefficient in m²/s

        Returns:
            Concentration at given distance and time in mol/m³
        """
        if diffusivity is None:
            diffusivity = 1e-10  # Typical for drugs in tissue

        from scipy.special import erfc

        if time <= 0:
            return 0

        argument = distance / (2 * np.sqrt(diffusivity * time))
        concentration = concentration_surface * erfc(argument)

        return concentration

    def compartment_model_pk(self, dose: float, volume: float,
                            clearance: float, time: np.ndarray,
                            ka: Optional[float] = None) -> np.ndarray:
        """
        Calculate drug concentration using one-compartment pharmacokinetic model

        Args:
            dose: Drug dose in mg
            volume: Volume of distribution in L
            clearance: Clearance rate in L/h
            time: Time array in hours
            ka: Absorption rate constant in 1/h (for oral administration)

        Returns:
            Concentration array in mg/L
        """
        # Elimination rate constant
        ke = clearance / volume

        if ka is None:
            # IV bolus
            concentration = (dose / volume) * np.exp(-ke * time)
        else:
            # Oral administration
            if ka != ke:
                concentration = (dose / volume) * (ka / (ka - ke)) * \
                              (np.exp(-ke * time) - np.exp(-ka * time))
            else:
                # Special case when ka = ke
                concentration = (dose / volume) * ka * time * np.exp(-ke * time)

        return concentration

    def drug_release_matrix(self, initial_loading: float, diffusivity: float,
                          matrix_thickness: float, time: np.ndarray) -> np.ndarray:
        """
        Calculate drug release from polymer matrix (Higuchi model)

        M_t = A * √(D * (2C0 - Cs) * Cs * t)

        Args:
            initial_loading: Initial drug concentration in matrix (kg/m³)
            diffusivity: Drug diffusivity in matrix (m²/s)
            matrix_thickness: Matrix thickness in meters
            time: Time array in seconds

        Returns:
            Cumulative drug release array (fraction)
        """
        # Assume drug solubility is 10% of initial loading
        solubility = 0.1 * initial_loading

        # Higuchi equation constant
        k_h = np.sqrt(2 * diffusivity * initial_loading * solubility)

        # Cumulative release
        release = k_h * np.sqrt(time) / matrix_thickness

        # Cap at 100% release
        return np.minimum(release, 1.0)

    def michaelis_menten_kinetics(self, substrate: float, vmax: float,
                                 km: float) -> float:
        """
        Calculate enzyme reaction rate using Michaelis-Menten kinetics

        v = Vmax * [S] / (Km + [S])

        Args:
            substrate: Substrate concentration in mol/L
            vmax: Maximum reaction velocity in mol/(L·s)
            km: Michaelis constant in mol/L

        Returns:
            Reaction rate in mol/(L·s)
        """
        return vmax * substrate / (km + substrate)

    def oxygen_transport_krogh(self, capillary_po2: float, radius: float,
                             consumption_rate: float) -> float:
        """
        Calculate tissue oxygen tension using Krogh cylinder model

        Args:
            capillary_po2: Capillary oxygen partial pressure in Pa
            radius: Distance from capillary center in meters
            consumption_rate: Oxygen consumption rate in mol/(m³·s)

        Returns:
            Tissue oxygen partial pressure in Pa
        """
        # Krogh cylinder parameters
        r_capillary = 5e-6  # m (capillary radius)
        r_tissue = 50e-6  # m (tissue cylinder radius)

        # Henry's law constant for O2 in tissue
        alpha = 1.3e-5  # mol/(m³·Pa)

        # Krogh equation (simplified)
        if radius <= r_capillary:
            return capillary_po2

        po2 = capillary_po2 - (consumption_rate / (4 * self.diffusion_coefficient_oxygen * alpha)) * \
              (radius**2 - r_capillary**2 + 2 * r_tissue**2 * np.log(radius / r_capillary))

        return max(po2, 0)

    def blood_flow_resistance(self, vessel_radius: float, vessel_length: float,
                            viscosity: Optional[float] = None,
                            flow_rate: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate vascular resistance and pressure drop (Poiseuille's law)

        R = 8μL/(πr⁴)

        Args:
            vessel_radius: Vessel radius in meters
            vessel_length: Vessel length in meters
            viscosity: Blood viscosity in Pa·s
            flow_rate: Flow rate in m³/s

        Returns:
            Dictionary with resistance and pressure drop
        """
        if viscosity is None:
            viscosity = self.blood_viscosity

        # Poiseuille resistance
        resistance = 8 * viscosity * vessel_length / (np.pi * vessel_radius**4)

        result = {'resistance': resistance}  # Pa·s/m³

        if flow_rate is not None:
            pressure_drop = flow_rate * resistance
            result['pressure_drop'] = pressure_drop  # Pa
            result['pressure_drop_mmHg'] = pressure_drop / 133.322  # mmHg

        return result

    def wall_shear_stress(self, flow_rate: float, vessel_radius: float,
                        viscosity: Optional[float] = None) -> float:
        """
        Calculate wall shear stress in blood vessel

        τ_wall = 4μQ/(πr³)

        Args:
            flow_rate: Blood flow rate in m³/s
            vessel_radius: Vessel radius in meters
            viscosity: Blood viscosity in Pa·s

        Returns:
            Wall shear stress in Pa
        """
        if viscosity is None:
            viscosity = self.blood_viscosity

        return 4 * viscosity * flow_rate / (np.pi * vessel_radius**3)

    def windkessel_model(self, pressure_in: np.ndarray, resistance: float,
                        compliance: float, time_step: float) -> np.ndarray:
        """
        Calculate arterial pressure using two-element Windkessel model

        Args:
            pressure_in: Input pressure array in Pa
            resistance: Peripheral resistance in Pa·s/m³
            compliance: Arterial compliance in m³/Pa
            time_step: Time step in seconds

        Returns:
            Output pressure array in Pa
        """
        # Time constant
        tau = resistance * compliance

        # Initialize output
        pressure_out = np.zeros_like(pressure_in)
        pressure_out[0] = pressure_in[0]

        # Solve differential equation
        for i in range(1, len(pressure_in)):
            dp_dt = (pressure_in[i] - pressure_out[i-1]) / tau
            pressure_out[i] = pressure_out[i-1] + dp_dt * time_step

        return pressure_out

    def ecg_signal_features(self, ecg_signal: np.ndarray,
                          sampling_rate: float) -> Dict[str, float]:
        """
        Extract features from ECG signal

        Args:
            ecg_signal: ECG signal array in mV
            sampling_rate: Sampling frequency in Hz

        Returns:
            Dictionary with heart rate, QRS duration, etc.
        """
        from scipy.signal import find_peaks

        # Find R peaks (simplified)
        peaks, _ = find_peaks(ecg_signal, height=0.5*np.max(ecg_signal),
                             distance=0.6*sampling_rate)

        if len(peaks) > 1:
            # RR intervals
            rr_intervals = np.diff(peaks) / sampling_rate

            # Heart rate
            heart_rate = 60 / np.mean(rr_intervals)

            # Heart rate variability (SDNN)
            hrv_sdnn = np.std(rr_intervals) * 1000  # ms

        else:
            heart_rate = 0
            hrv_sdnn = 0

        return {
            'heart_rate': heart_rate,  # bpm
            'hrv_sdnn': hrv_sdnn,  # ms
            'num_beats': len(peaks),
            'signal_duration': len(ecg_signal) / sampling_rate  # seconds
        }

    def fourier_transform_biosignal(self, signal: np.ndarray,
                                  sampling_rate: float) -> Dict[str, np.ndarray]:
        """
        Perform FFT on biosignal and extract frequency components

        Args:
            signal: Biosignal array
            sampling_rate: Sampling frequency in Hz

        Returns:
            Dictionary with frequencies and power spectrum
        """
        # Remove DC component
        signal = signal - np.mean(signal)

        # Apply Hamming window
        window = np.hamming(len(signal))
        signal = signal * window

        # FFT
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1/sampling_rate)

        # Power spectrum
        power = np.abs(fft)**2

        # Find dominant frequency
        dominant_freq_idx = np.argmax(power[1:]) + 1  # Skip DC
        dominant_freq = freqs[dominant_freq_idx]

        return {
            'frequencies': freqs,
            'power_spectrum': power,
            'dominant_frequency': dominant_freq,
            'total_power': np.sum(power)
        }

    def filter_biosignal(self, signal: np.ndarray, sampling_rate: float,
                        low_cutoff: Optional[float] = None,
                        high_cutoff: Optional[float] = None,
                        order: int = 4) -> np.ndarray:
        """
        Apply Butterworth filter to biosignal

        Args:
            signal: Input signal array
            sampling_rate: Sampling frequency in Hz
            low_cutoff: High-pass cutoff frequency in Hz
            high_cutoff: Low-pass cutoff frequency in Hz
            order: Filter order

        Returns:
            Filtered signal array
        """
        nyquist = 0.5 * sampling_rate

        if low_cutoff is not None and high_cutoff is not None:
            # Bandpass filter
            sos = signal.butter(order, [low_cutoff/nyquist, high_cutoff/nyquist],
                              btype='band', output='sos')
        elif low_cutoff is not None:
            # High-pass filter
            sos = signal.butter(order, low_cutoff/nyquist,
                              btype='high', output='sos')
        elif high_cutoff is not None:
            # Low-pass filter
            sos = signal.butter(order, high_cutoff/nyquist,
                              btype='low', output='sos')
        else:
            return signal

        return signal.sosfiltfilt(sos, signal)

    def implant_stress_shielding(self, implant_modulus: float,
                               bone_modulus: Optional[float] = None) -> float:
        """
        Calculate stress shielding factor for orthopedic implant

        Args:
            implant_modulus: Implant Young's modulus in Pa
            bone_modulus: Bone Young's modulus in Pa

        Returns:
            Stress shielding factor (0=no shielding, 1=complete shielding)
        """
        if bone_modulus is None:
            bone_modulus = self.bone_youngs_modulus

        # Stress distribution ratio
        stress_ratio = implant_modulus / (implant_modulus + bone_modulus)

        return stress_ratio

    def biocompatibility_index(self, cytotoxicity: float, protein_adsorption: float,
                              cell_adhesion: float, inflammatory_response: float) -> float:
        """
        Calculate overall biocompatibility index (0-100)

        Args:
            cytotoxicity: Cell viability (0-1, 1=no toxicity)
            protein_adsorption: Protein binding (0-1, optimal ~0.5)
            cell_adhesion: Cell attachment (0-1, 1=excellent)
            inflammatory_response: Inflammation (0-1, 0=no inflammation)

        Returns:
            Biocompatibility index (0-100)
        """
        # Weight factors
        weights = {
            'cytotoxicity': 0.4,
            'protein': 0.2,
            'adhesion': 0.25,
            'inflammation': 0.15
        }

        # Calculate protein score (optimal at 0.5)
        protein_score = 1 - 2 * abs(protein_adsorption - 0.5)

        # Calculate inflammation score (inverse)
        inflammation_score = 1 - inflammatory_response

        # Weighted average
        index = (weights['cytotoxicity'] * cytotoxicity +
                weights['protein'] * protein_score +
                weights['adhesion'] * cell_adhesion +
                weights['inflammation'] * inflammation_score)

        return index * 100

    def tissue_scaffold_porosity(self, pore_size: float, scaffold_thickness: float,
                                diffusion_required: float) -> Dict[str, float]:
        """
        Calculate optimal scaffold porosity for tissue engineering

        Args:
            pore_size: Average pore diameter in meters
            scaffold_thickness: Scaffold thickness in meters
            diffusion_required: Required nutrient diffusion rate in m²/s

        Returns:
            Dictionary with porosity, permeability, and cell capacity
        """
        # Optimal porosity for cell growth (empirical)
        optimal_porosity = 0.7 + 0.2 * np.tanh((pore_size - 100e-6) / 50e-6)
        optimal_porosity = np.clip(optimal_porosity, 0.5, 0.95)

        # Kozeny-Carman permeability
        permeability = (optimal_porosity**3 * pore_size**2) / \
                      (180 * (1 - optimal_porosity)**2)

        # Nutrient penetration depth
        penetration_depth = np.sqrt(diffusion_required * scaffold_thickness /
                                  (0.001))  # Assuming consumption rate

        # Cell seeding capacity (cells/m³)
        cell_capacity = optimal_porosity * 1e12  # Assuming cell size ~10μm

        return {
            'porosity': optimal_porosity,
            'permeability': permeability,  # m²
            'penetration_depth': penetration_depth,  # m
            'cell_capacity': cell_capacity,  # cells/m³
            'pore_connectivity': optimal_porosity**2  # Approximate
        }

    def cell_growth_logistic(self, initial_cells: float, growth_rate: float,
                           carrying_capacity: float, time: np.ndarray) -> np.ndarray:
        """
        Model cell population growth using logistic equation

        N(t) = K / (1 + ((K-N0)/N0) * exp(-rt))

        Args:
            initial_cells: Initial cell count
            growth_rate: Growth rate constant in 1/day
            carrying_capacity: Maximum cell count
            time: Time array in days

        Returns:
            Cell count array
        """
        if initial_cells <= 0 or initial_cells >= carrying_capacity:
            return np.full_like(time, initial_cells)

        a = (carrying_capacity - initial_cells) / initial_cells
        cell_count = carrying_capacity / (1 + a * np.exp(-growth_rate * time))

        return cell_count

    def bioreactor_oxygen_transfer(self, flow_rate: float, volume: float,
                                  kla: float, oxygen_saturation: float,
                                  consumption_rate: float) -> float:
        """
        Calculate steady-state oxygen concentration in bioreactor

        Args:
            flow_rate: Gas flow rate in L/min
            volume: Bioreactor volume in L
            kla: Oxygen transfer coefficient in 1/h
            oxygen_saturation: Saturation concentration in mg/L
            consumption_rate: Cell oxygen consumption in mg/(L·h)

        Returns:
            Steady-state dissolved oxygen in mg/L
        """
        # Convert flow rate to 1/h
        flow_rate_h = flow_rate * 60

        # Steady-state balance
        # kLa*(C_sat - C) = OUR
        dissolved_oxygen = oxygen_saturation - consumption_rate / kla

        return max(dissolved_oxygen, 0)


def run_demo():
    """Demonstrate biomedical engineering calculations"""
    print("=" * 70)
    print("BIOMEDICAL ENGINEERING LAB - Production Demo")
    print("=" * 70)

    lab = BiomedicalEngineeringLab()

    # Biomechanics
    print("\n1. BIOMECHANICS ANALYSIS")
    print("-" * 40)
    bone = lab.stress_strain_bone(50e6, 0.001)
    print(f"Bone under 50 MPa stress:")
    print(f"  Strain: {bone['total_strain']:.4f}")
    print(f"  Safety factor: {bone['safety_factor']:.2f}")
    print(f"  Fracture risk: {bone['fracture_risk']}")

    joint = lab.joint_force_analysis(700, 'running')
    print(f"Joint forces during running (700N body weight):")
    print(f"  Hip: {joint['hip_force']:.0f} N")
    print(f"  Knee: {joint['knee_force']:.0f} N")

    # Drug Delivery
    print("\n2. DRUG DELIVERY MODELING")
    print("-" * 40)
    time = np.linspace(0, 24, 100)  # 24 hours
    pk = lab.compartment_model_pk(100, 50, 5, time, ka=1.0)
    print(f"Pharmacokinetics (100mg oral dose):")
    print(f"  Peak concentration: {np.max(pk):.2f} mg/L")
    print(f"  Time to peak: {time[np.argmax(pk)]:.1f} hours")

    diffusion = lab.drug_diffusion_tissue(100, 0.001, 3600)
    print(f"Drug diffusion (1mm, 1 hour): {diffusion:.2f} mol/m³")

    # Cardiovascular
    print("\n3. CARDIOVASCULAR ANALYSIS")
    print("-" * 40)
    resistance = lab.blood_flow_resistance(0.002, 0.1, flow_rate=8.3e-5)
    print(f"Artery (2mm radius, 10cm length):")
    print(f"  Resistance: {resistance['resistance']:.2e} Pa·s/m³")
    print(f"  Pressure drop: {resistance['pressure_drop_mmHg']:.1f} mmHg")

    shear = lab.wall_shear_stress(8.3e-5, 0.002)
    print(f"Wall shear stress: {shear:.2f} Pa")

    # Biosignal Processing
    print("\n4. BIOSIGNAL PROCESSING")
    print("-" * 40)
    # Simulate simple ECG
    t = np.linspace(0, 2, 2000)
    ecg = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
    ecg[::int(1000/1.2)] = 2  # R peaks

    features = lab.ecg_signal_features(ecg, 1000)
    print(f"ECG Analysis:")
    print(f"  Heart rate: {features['heart_rate']:.0f} bpm")
    print(f"  Duration: {features['signal_duration']:.1f} s")

    fft_result = lab.fourier_transform_biosignal(ecg[:1000], 1000)
    print(f"  Dominant frequency: {fft_result['dominant_frequency']:.2f} Hz")

    # Medical Device Design
    print("\n5. MEDICAL DEVICE & IMPLANT DESIGN")
    print("-" * 40)
    titanium_modulus = 110e9  # Pa
    shielding = lab.implant_stress_shielding(titanium_modulus)
    print(f"Titanium implant stress shielding: {shielding:.2%}")

    biocompat = lab.biocompatibility_index(0.95, 0.5, 0.8, 0.1)
    print(f"Biocompatibility index: {biocompat:.1f}/100")

    # Tissue Engineering
    print("\n6. TISSUE ENGINEERING")
    print("-" * 40)
    scaffold = lab.tissue_scaffold_porosity(200e-6, 0.005, 1e-9)
    print(f"Scaffold design (200μm pores):")
    print(f"  Optimal porosity: {scaffold['porosity']:.1%}")
    print(f"  Cell capacity: {scaffold['cell_capacity']:.2e} cells/m³")

    days = np.linspace(0, 14, 100)
    growth = lab.cell_growth_logistic(1e4, 0.5, 1e7, days)
    print(f"Cell culture (14 days):")
    print(f"  Initial: {1e4:.0f} cells")
    print(f"  Final: {growth[-1]:.2e} cells")
    print(f"  Fold increase: {growth[-1]/1e4:.1f}x")

    print("\n" + "=" * 70)
    print("Demo completed successfully!")


if __name__ == '__main__':
    run_demo()