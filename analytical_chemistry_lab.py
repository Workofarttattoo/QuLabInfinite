"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ANALYTICAL CHEMISTRY LAB
Production-ready analytical chemistry algorithms and simulations.
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
import math
from scipy import stats
from scipy.optimize import curve_fit, minimize
from scipy.signal import find_peaks, savgol_filter
from scipy.integrate import trapz, simps
from enum import Enum

class AnalyticalTechnique(Enum):
    """Common analytical techniques"""
    UV_VIS = "UV-Vis Spectroscopy"
    FTIR = "Fourier Transform Infrared"
    NMR = "Nuclear Magnetic Resonance"
    MS = "Mass Spectrometry"
    GC = "Gas Chromatography"
    HPLC = "High Performance Liquid Chromatography"
    AAS = "Atomic Absorption Spectroscopy"
    ICP = "Inductively Coupled Plasma"
    XRF = "X-ray Fluorescence"
    XRD = "X-ray Diffraction"
    FLUORESCENCE = "Fluorescence Spectroscopy"
    ELECTROCHEMICAL = "Electrochemical Analysis"

class SeparationMode(Enum):
    """Chromatography separation modes"""
    NORMAL_PHASE = "normal_phase"
    REVERSE_PHASE = "reverse_phase"
    ION_EXCHANGE = "ion_exchange"
    SIZE_EXCLUSION = "size_exclusion"
    AFFINITY = "affinity"
    CHIRAL = "chiral"

@dataclass
class AnalyticalSignal:
    """Analytical measurement signal"""
    wavelength: np.ndarray  # nm or m/z or retention time
    intensity: np.ndarray   # Absorbance, counts, etc.
    noise_level: float = 0.01
    baseline: np.ndarray = None

    def __post_init__(self):
        if self.baseline is None:
            self.baseline = np.zeros_like(self.intensity)

@dataclass
class ChromatographicPeak:
    """Chromatographic peak parameters"""
    retention_time: float  # minutes
    area: float
    height: float
    width: float  # at half height
    symmetry: float  # asymmetry factor
    plates: int  # theoretical plates
    resolution: float = 0.0
    tailing_factor: float = 1.0

@dataclass
class CalibrationCurve:
    """Calibration curve data and parameters"""
    concentrations: np.ndarray
    responses: np.ndarray
    equation_type: str  # linear, quadratic, power
    coefficients: np.ndarray = None
    r_squared: float = 0.0
    residuals: np.ndarray = None
    lod: float = 0.0  # Limit of detection
    loq: float = 0.0  # Limit of quantification

class AnalyticalChemistryLab:
    """Production-ready analytical chemistry laboratory"""

    def __init__(self):
        self.instruments = {}
        self.methods = {}
        self.standards = {}

    # === CALIBRATION METHODS ===

    def create_calibration_curve(self, concentrations: np.ndarray,
                                responses: np.ndarray,
                                curve_type: str = "linear",
                                weighted: bool = False) -> CalibrationCurve:
        """Create and fit calibration curve"""
        curve = CalibrationCurve(
            concentrations=concentrations,
            responses=responses,
            equation_type=curve_type
        )

        # Remove any NaN or infinite values
        valid = np.isfinite(concentrations) & np.isfinite(responses)
        x = concentrations[valid]
        y = responses[valid]

        if len(x) < 2:
            raise ValueError("Need at least 2 valid calibration points")

        # Weights for weighted regression
        weights = 1 / y if weighted and np.all(y > 0) else None

        # Fit calibration curve
        if curve_type == "linear":
            # y = ax + b
            if weights is not None:
                coeffs = np.polyfit(x, y, 1, w=weights)
            else:
                coeffs = np.polyfit(x, y, 1)
            curve.coefficients = coeffs
            y_fit = np.polyval(coeffs, x)

        elif curve_type == "quadratic":
            # y = ax² + bx + c
            if weights is not None:
                coeffs = np.polyfit(x, y, 2, w=weights)
            else:
                coeffs = np.polyfit(x, y, 2)
            curve.coefficients = coeffs
            y_fit = np.polyval(coeffs, x)

        elif curve_type == "power":
            # y = ax^b (linearize as log(y) = log(a) + b*log(x))
            log_x = np.log(x[x > 0])
            log_y = np.log(y[x > 0])
            coeffs = np.polyfit(log_x, log_y, 1)
            curve.coefficients = [np.exp(coeffs[1]), coeffs[0]]  # [a, b]
            y_fit = curve.coefficients[0] * x ** curve.coefficients[1]

        else:
            raise ValueError(f"Unknown curve type: {curve_type}")

        # Calculate statistics
        curve.residuals = y - y_fit
        ss_res = np.sum(curve.residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        curve.r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return curve

    def calculate_lod_loq(self, curve: CalibrationCurve,
                         blank_responses: np.ndarray = None) -> Tuple[float, float]:
        """Calculate limit of detection (LOD) and limit of quantification (LOQ)"""

        if blank_responses is not None and len(blank_responses) > 1:
            # Method 1: Based on blank measurements
            blank_mean = np.mean(blank_responses)
            blank_std = np.std(blank_responses)

            # LOD = mean_blank + 3*std_blank
            # LOQ = mean_blank + 10*std_blank
            lod_response = blank_mean + 3 * blank_std
            loq_response = blank_mean + 10 * blank_std

        else:
            # Method 2: Based on calibration curve residuals
            if curve.residuals is not None and len(curve.residuals) > 2:
                sy = np.std(curve.residuals)
            else:
                # Estimate from lowest standards
                low_concs = curve.concentrations[:3]
                low_resp = curve.responses[:3]
                if len(low_concs) >= 2:
                    sy = np.std(low_resp) / np.mean(low_resp) * np.mean(low_resp)
                else:
                    sy = 0.05 * np.mean(curve.responses)  # 5% estimate

            # For linear calibration: LOD = 3.3 * σ/S, LOQ = 10 * σ/S
            if curve.equation_type == "linear" and curve.coefficients is not None:
                slope = curve.coefficients[0]
                lod_response = 3.3 * sy
                loq_response = 10 * sy
            else:
                lod_response = np.min(curve.responses)
                loq_response = np.min(curve.responses) * 3

        # Convert response to concentration
        if curve.equation_type == "linear":
            curve.lod = (lod_response - curve.coefficients[1]) / curve.coefficients[0]
            curve.loq = (loq_response - curve.coefficients[1]) / curve.coefficients[0]
        else:
            # Use interpolation for non-linear curves
            curve.lod = np.interp(lod_response, curve.responses, curve.concentrations)
            curve.loq = np.interp(loq_response, curve.responses, curve.concentrations)

        # Ensure positive values
        curve.lod = max(0, curve.lod)
        curve.loq = max(0, curve.loq)

        return curve.lod, curve.loq

    def internal_standard_correction(self, analyte_response: float,
                                    istd_response: float,
                                    istd_concentration: float) -> float:
        """Apply internal standard correction"""
        # Response factor = (analyte_response / istd_response) * istd_concentration
        if istd_response > 0:
            response_factor = (analyte_response / istd_response) * istd_concentration
        else:
            response_factor = 0
        return response_factor

    # === CHROMATOGRAPHY METHODS ===

    def simulate_chromatogram(self, compounds: List[Dict],
                             time_points: np.ndarray,
                             mode: SeparationMode = SeparationMode.REVERSE_PHASE) -> AnalyticalSignal:
        """Simulate a chromatogram with multiple peaks"""

        signal = np.zeros_like(time_points)

        for compound in compounds:
            rt = compound.get('retention_time', 5.0)
            height = compound.get('height', 1000)
            width = compound.get('width', 0.5)
            tailing = compound.get('tailing', 1.0)

            # Exponentially modified Gaussian for realistic peak shape
            if tailing > 1.0:
                # EMG peak shape for tailing
                gaussian = height * np.exp(-0.5 * ((time_points - rt) / width) ** 2)
                exponential = np.exp((time_points - rt) / (width * tailing))
                exponential[time_points < rt] = 1.0
                peak = gaussian * exponential
            else:
                # Simple Gaussian
                peak = height * np.exp(-0.5 * ((time_points - rt) / width) ** 2)

            signal += peak

        # Add baseline drift and noise
        baseline = 10 + 0.5 * time_points + 2 * np.sin(0.1 * time_points)
        noise = np.random.normal(0, height * 0.01, len(time_points))

        return AnalyticalSignal(
            wavelength=time_points,
            intensity=signal + baseline + noise,
            baseline=baseline
        )

    def integrate_peak(self, signal: AnalyticalSignal,
                      start_time: float, end_time: float,
                      baseline_correction: bool = True) -> float:
        """Integrate chromatographic peak area"""

        # Find indices for integration window
        mask = (signal.wavelength >= start_time) & (signal.wavelength <= end_time)
        x = signal.wavelength[mask]
        y = signal.intensity[mask]

        if len(x) < 2:
            return 0.0

        # Baseline correction
        if baseline_correction:
            # Linear baseline between start and end points
            baseline_start = y[0]
            baseline_end = y[-1]
            baseline = np.linspace(baseline_start, baseline_end, len(y))
            y_corrected = y - baseline
        else:
            y_corrected = y

        # Trapezoidal integration
        area = trapz(y_corrected, x)

        return abs(area)

    def calculate_resolution(self, peak1: ChromatographicPeak,
                           peak2: ChromatographicPeak) -> float:
        """Calculate chromatographic resolution between peaks"""
        # Rs = 2 * (RT2 - RT1) / (W1 + W2)
        if peak1.width + peak2.width > 0:
            resolution = 2 * abs(peak2.retention_time - peak1.retention_time) / \
                        (peak1.width + peak2.width)
        else:
            resolution = 0.0
        return resolution

    def calculate_theoretical_plates(self, retention_time: float,
                                    peak_width: float,
                                    width_type: str = "half_height") -> int:
        """Calculate number of theoretical plates"""
        if peak_width <= 0:
            return 0

        if width_type == "half_height":
            # N = 5.54 * (tr/w0.5)²
            N = 5.54 * (retention_time / peak_width) ** 2
        elif width_type == "base":
            # N = 16 * (tr/wb)²
            N = 16 * (retention_time / peak_width) ** 2
        else:
            N = 0

        return int(N)

    def van_deemter_equation(self, flow_rate: float,
                            A: float = 0.5, B: float = 2.0,
                            C: float = 0.05) -> float:
        """Van Deemter equation for plate height"""
        # H = A + B/u + C*u
        # A: Eddy diffusion, B: Longitudinal diffusion, C: Mass transfer
        if flow_rate > 0:
            H = A + B / flow_rate + C * flow_rate
        else:
            H = float('inf')
        return H

    # === MASS SPECTROMETRY METHODS ===

    def simulate_mass_spectrum(self, molecular_formula: str,
                             fragmentation_pattern: Dict[int, float]) -> AnalyticalSignal:
        """Simulate a mass spectrum with isotope patterns"""

        # Calculate molecular mass (simplified)
        element_masses = {
            'C': 12.011, 'H': 1.008, 'N': 14.007, 'O': 15.999,
            'S': 32.065, 'P': 30.974, 'F': 18.998, 'Cl': 35.453
        }

        # Parse formula (simplified parser)
        import re
        pattern = r'([A-Z][a-z]?)(\d*)'
        elements = re.findall(pattern, molecular_formula)

        molecular_mass = 0
        for element, count in elements:
            count = int(count) if count else 1
            molecular_mass += element_masses.get(element, 0) * count

        # Create m/z array
        mz = np.arange(10, molecular_mass + 50, 0.1)
        intensity = np.zeros_like(mz)

        # Add molecular ion peak
        mol_idx = np.argmin(abs(mz - molecular_mass))
        intensity[mol_idx] = 100  # Base peak

        # Add isotope peaks (simplified)
        # C13 isotope (1.1% per carbon)
        n_carbons = sum(int(count) if count else 1 for elem, count in elements if elem == 'C')
        if n_carbons > 0:
            isotope_idx = np.argmin(abs(mz - (molecular_mass + 1)))
            intensity[isotope_idx] = 1.1 * n_carbons

        # Add fragmentation peaks
        for fragment_mass, relative_intensity in fragmentation_pattern.items():
            frag_idx = np.argmin(abs(mz - fragment_mass))
            intensity[frag_idx] = relative_intensity

        # Add noise
        noise = np.random.poisson(0.1, len(mz))
        intensity = intensity + noise

        return AnalyticalSignal(
            wavelength=mz,  # m/z values
            intensity=intensity,
            noise_level=0.1
        )

    def calculate_exact_mass(self, molecular_formula: str) -> float:
        """Calculate exact molecular mass"""
        # Exact masses for common isotopes
        exact_masses = {
            'C': 12.0000, 'H': 1.00783, 'N': 14.0031, 'O': 15.9949,
            'S': 31.9721, 'P': 30.9738, 'F': 18.9984, 'Cl': 34.9689,
            'Br': 78.9183, 'I': 126.9045
        }

        import re
        pattern = r'([A-Z][a-z]?)(\d*)'
        elements = re.findall(pattern, molecular_formula)

        exact_mass = 0
        for element, count in elements:
            count = int(count) if count else 1
            exact_mass += exact_masses.get(element, 0) * count

        return exact_mass

    def mass_accuracy_ppm(self, measured_mass: float, exact_mass: float) -> float:
        """Calculate mass accuracy in ppm"""
        if exact_mass > 0:
            ppm = ((measured_mass - exact_mass) / exact_mass) * 1e6
        else:
            ppm = 0
        return abs(ppm)

    # === SPECTROSCOPY METHODS ===

    def beer_lambert_law(self, epsilon: float, path_length: float,
                        concentration: float) -> float:
        """Calculate absorbance using Beer-Lambert law"""
        # A = ε * l * c
        absorbance = epsilon * path_length * concentration
        return absorbance

    def calculate_transmittance(self, absorbance: float) -> float:
        """Convert absorbance to transmittance"""
        # T = 10^(-A)
        transmittance = 10 ** (-absorbance)
        return transmittance

    def spectral_deconvolution(self, signal: AnalyticalSignal,
                              n_components: int = 2) -> List[Dict]:
        """Deconvolute overlapping spectral peaks"""
        components = []

        # Use peak finding to get initial guesses
        peaks, properties = find_peaks(signal.intensity,
                                      height=np.max(signal.intensity) * 0.1)

        if len(peaks) == 0:
            return components

        # Fit Gaussian components
        def multi_gaussian(x, *params):
            y = np.zeros_like(x)
            for i in range(0, len(params), 3):
                amp = params[i]
                cen = params[i+1]
                wid = params[i+2]
                y += amp * np.exp(-0.5 * ((x - cen) / wid) ** 2)
            return y

        # Initial parameters
        p0 = []
        for i in range(min(n_components, len(peaks))):
            p0.extend([signal.intensity[peaks[i]],
                      signal.wavelength[peaks[i]],
                      10.0])  # amplitude, center, width

        try:
            popt, _ = curve_fit(multi_gaussian, signal.wavelength,
                              signal.intensity, p0=p0)

            # Extract component parameters
            for i in range(0, len(popt), 3):
                components.append({
                    'amplitude': popt[i],
                    'center': popt[i+1],
                    'width': abs(popt[i+2]),
                    'area': popt[i] * abs(popt[i+2]) * np.sqrt(2 * np.pi)
                })
        except:
            # Fallback if fitting fails
            for peak_idx in peaks[:n_components]:
                components.append({
                    'amplitude': signal.intensity[peak_idx],
                    'center': signal.wavelength[peak_idx],
                    'width': 10.0,
                    'area': signal.intensity[peak_idx] * 10.0 * np.sqrt(2 * np.pi)
                })

        return components

    def fluorescence_quantum_yield(self, sample_area: float, sample_absorbance: float,
                                  reference_area: float, reference_absorbance: float,
                                  reference_qy: float = 0.95,
                                  n_sample: float = 1.33,
                                  n_reference: float = 1.33) -> float:
        """Calculate fluorescence quantum yield"""
        # Φ_x = Φ_r * (A_x/A_r) * (Abs_r/Abs_x) * (n_x²/n_r²)
        if reference_area > 0 and sample_absorbance > 0:
            qy = reference_qy * (sample_area / reference_area) * \
                 (reference_absorbance / sample_absorbance) * \
                 (n_sample ** 2 / n_reference ** 2)
        else:
            qy = 0
        return min(qy, 1.0)  # Quantum yield cannot exceed 1

    # === METHOD VALIDATION ===

    def calculate_method_precision(self, replicate_measurements: np.ndarray) -> Dict:
        """Calculate method precision statistics"""
        precision = {
            'mean': np.mean(replicate_measurements),
            'std_dev': np.std(replicate_measurements, ddof=1),
            'rsd': 0.0,  # Relative standard deviation (CV)
            'confidence_95': 0.0,
            'n': len(replicate_measurements)
        }

        if precision['mean'] > 0:
            precision['rsd'] = (precision['std_dev'] / precision['mean']) * 100

        # 95% confidence interval
        if precision['n'] > 1:
            t_value = stats.t.ppf(0.975, precision['n'] - 1)
            precision['confidence_95'] = t_value * precision['std_dev'] / np.sqrt(precision['n'])

        return precision

    def calculate_recovery(self, measured: float, expected: float) -> float:
        """Calculate analytical recovery percentage"""
        if expected > 0:
            recovery = (measured / expected) * 100
        else:
            recovery = 0
        return recovery

    def matrix_spike_recovery(self, sample: float, spiked_sample: float,
                             spike_amount: float) -> float:
        """Calculate matrix spike recovery"""
        # Recovery = (Spiked - Unspiked) / Spike_Added * 100
        if spike_amount > 0:
            recovery = ((spiked_sample - sample) / spike_amount) * 100
        else:
            recovery = 0
        return recovery

    def calculate_linearity(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """Assess linearity of analytical method"""
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Calculate residuals
        y_pred = slope * x + intercept
        residuals = y - y_pred

        # Test for lack of fit
        # Simplified: check if residuals show pattern
        residual_runs = np.sum(np.diff(np.sign(residuals)) != 0)
        expected_runs = len(residuals) / 2

        linearity = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_error': std_err,
            'residuals': residuals,
            'residual_runs_test': abs(residual_runs - expected_runs) < len(residuals) * 0.2
        }

        return linearity

    def method_detection_limit(self, blank_measurements: np.ndarray,
                              low_conc_measurements: np.ndarray,
                              concentration: float) -> float:
        """Calculate method detection limit (MDL)"""
        # EPA Method: MDL = t * s
        # where t is Student's t-value and s is standard deviation

        if len(low_conc_measurements) > 1:
            s = np.std(low_conc_measurements, ddof=1)
            n = len(low_conc_measurements)
            t_value = stats.t.ppf(0.99, n - 1)  # 99% confidence
            mdl = t_value * s
        else:
            # Alternative: 3 * std of blanks
            mdl = 3 * np.std(blank_measurements) if len(blank_measurements) > 0 else 0

        return mdl

    # === DATA PROCESSING ===

    def signal_smoothing(self, signal: AnalyticalSignal,
                        window_length: int = 11,
                        polyorder: int = 3) -> AnalyticalSignal:
        """Apply Savitzky-Golay filter for signal smoothing"""
        smoothed_intensity = savgol_filter(signal.intensity, window_length, polyorder)

        return AnalyticalSignal(
            wavelength=signal.wavelength,
            intensity=smoothed_intensity,
            noise_level=signal.noise_level * 0.5,  # Reduced noise
            baseline=signal.baseline
        )

    def baseline_correction(self, signal: AnalyticalSignal,
                          method: str = "polynomial") -> AnalyticalSignal:
        """Perform baseline correction"""

        if method == "polynomial":
            # Fit polynomial to minima
            window = len(signal.intensity) // 10
            minima = []
            minima_idx = []

            for i in range(0, len(signal.intensity), window):
                segment = signal.intensity[i:i+window]
                if len(segment) > 0:
                    min_idx = i + np.argmin(segment)
                    minima_idx.append(min_idx)
                    minima.append(signal.intensity[min_idx])

            if len(minima_idx) > 2:
                # Fit polynomial through minima
                baseline_coeffs = np.polyfit(signal.wavelength[minima_idx], minima, 2)
                baseline = np.polyval(baseline_coeffs, signal.wavelength)
            else:
                baseline = np.zeros_like(signal.intensity)

        elif method == "linear":
            # Simple linear baseline from endpoints
            baseline = np.linspace(signal.intensity[0], signal.intensity[-1],
                                  len(signal.intensity))
        else:
            baseline = np.zeros_like(signal.intensity)

        corrected_intensity = signal.intensity - baseline

        return AnalyticalSignal(
            wavelength=signal.wavelength,
            intensity=corrected_intensity,
            noise_level=signal.noise_level,
            baseline=baseline
        )

    def peak_detection(self, signal: AnalyticalSignal,
                      height_threshold: float = None,
                      prominence_threshold: float = None) -> List[Dict]:
        """Detect and characterize peaks in signal"""

        if height_threshold is None:
            height_threshold = np.mean(signal.intensity) + 2 * np.std(signal.intensity)

        peaks, properties = find_peaks(signal.intensity,
                                      height=height_threshold,
                                      prominence=prominence_threshold)

        peak_list = []
        for i, peak_idx in enumerate(peaks):
            # Calculate peak parameters
            peak_height = signal.intensity[peak_idx]

            # Find peak width at half height
            half_height = peak_height / 2
            left_idx = peak_idx
            right_idx = peak_idx

            while left_idx > 0 and signal.intensity[left_idx] > half_height:
                left_idx -= 1
            while right_idx < len(signal.intensity) - 1 and signal.intensity[right_idx] > half_height:
                right_idx += 1

            if right_idx > left_idx:
                width = signal.wavelength[right_idx] - signal.wavelength[left_idx]
            else:
                width = 0

            # Calculate peak area
            area = self.integrate_peak(signal,
                                      signal.wavelength[max(0, peak_idx-10)],
                                      signal.wavelength[min(len(signal.wavelength)-1, peak_idx+10)],
                                      baseline_correction=True)

            peak_list.append({
                'position': signal.wavelength[peak_idx],
                'height': peak_height,
                'width': width,
                'area': area,
                'prominence': properties.get('prominences', [0])[i] if 'prominences' in properties else 0
            })

        return peak_list

    def calculate_signal_to_noise(self, signal: AnalyticalSignal,
                                 signal_region: Tuple[float, float],
                                 noise_region: Tuple[float, float]) -> float:
        """Calculate signal-to-noise ratio"""

        # Extract signal region
        signal_mask = (signal.wavelength >= signal_region[0]) & \
                     (signal.wavelength <= signal_region[1])
        signal_values = signal.intensity[signal_mask]

        # Extract noise region
        noise_mask = (signal.wavelength >= noise_region[0]) & \
                     (signal.wavelength <= noise_region[1])
        noise_values = signal.intensity[noise_mask]

        if len(signal_values) > 0 and len(noise_values) > 0:
            signal_amplitude = np.max(signal_values) - np.min(signal_values)
            noise_std = np.std(noise_values)

            if noise_std > 0:
                snr = signal_amplitude / (2 * noise_std)
            else:
                snr = float('inf')
        else:
            snr = 0

        return snr

    def uncertainty_propagation(self, values: List[float],
                              uncertainties: List[float],
                              operation: str) -> Tuple[float, float]:
        """Calculate propagated uncertainty for basic operations"""

        if len(values) != len(uncertainties):
            raise ValueError("Values and uncertainties must have same length")

        if operation == "sum":
            result = sum(values)
            # Uncertainty in sum: √(Σu²)
            unc = np.sqrt(sum(u**2 for u in uncertainties))

        elif operation == "product":
            result = np.prod(values)
            # Relative uncertainty in product: √(Σ(u/v)²)
            if result != 0 and all(v != 0 for v in values):
                rel_unc = np.sqrt(sum((u/v)**2 for u, v in zip(uncertainties, values)))
                unc = abs(result * rel_unc)
            else:
                unc = 0

        elif operation == "quotient" and len(values) == 2:
            if values[1] != 0:
                result = values[0] / values[1]
                # Relative uncertainty in quotient
                rel_unc = np.sqrt((uncertainties[0]/values[0])**2 +
                                 (uncertainties[1]/values[1])**2)
                unc = abs(result * rel_unc)
            else:
                result = 0
                unc = 0

        else:
            result = np.mean(values)
            unc = np.std(values)

        return result, unc

def run_demo():
    """Demonstrate analytical chemistry lab capabilities"""
    print("=" * 80)
    print("ANALYTICAL CHEMISTRY LAB - Production Demo")
    print("Copyright (c) 2025 Joshua Hendricks Cole")
    print("=" * 80)

    lab = AnalyticalChemistryLab()

    # 1. Calibration curve
    print("\n1. Creating calibration curve...")
    concentrations = np.array([0, 0.5, 1.0, 2.0, 5.0, 10.0])  # mg/L
    responses = np.array([0.002, 0.051, 0.102, 0.198, 0.495, 0.985])  # Absorbance

    curve = lab.create_calibration_curve(concentrations, responses, "linear")
    print(f"   Equation: y = {curve.coefficients[0]:.4f}x + {curve.coefficients[1]:.4f}")
    print(f"   R² = {curve.r_squared:.4f}")

    # 2. LOD/LOQ calculation
    print("\n2. Calculating LOD and LOQ...")
    blank_responses = np.array([0.001, 0.002, 0.001, 0.003, 0.002])
    lod, loq = lab.calculate_lod_loq(curve, blank_responses)
    print(f"   LOD: {lod:.3f} mg/L")
    print(f"   LOQ: {loq:.3f} mg/L")

    # 3. Chromatography simulation
    print("\n3. Simulating chromatogram...")
    compounds = [
        {'retention_time': 3.5, 'height': 1000, 'width': 0.3, 'tailing': 1.1},
        {'retention_time': 5.2, 'height': 1500, 'width': 0.35, 'tailing': 1.2},
        {'retention_time': 7.8, 'height': 800, 'width': 0.4, 'tailing': 1.0}
    ]
    time_points = np.linspace(0, 10, 1000)
    chromatogram = lab.simulate_chromatogram(compounds, time_points)
    print(f"   Generated {len(compounds)} peaks")

    # 4. Peak integration
    print("\n4. Integrating chromatographic peaks...")
    area1 = lab.integrate_peak(chromatogram, 3.0, 4.0)
    area2 = lab.integrate_peak(chromatogram, 4.8, 5.8)
    print(f"   Peak 1 area: {area1:.1f}")
    print(f"   Peak 2 area: {area2:.1f}")

    # 5. Resolution calculation
    print("\n5. Calculating chromatographic resolution...")
    peak1 = ChromatographicPeak(retention_time=3.5, area=area1, height=1000,
                                width=0.3, symmetry=1.0, plates=5000)
    peak2 = ChromatographicPeak(retention_time=5.2, area=area2, height=1500,
                                width=0.35, symmetry=1.0, plates=5000)
    resolution = lab.calculate_resolution(peak1, peak2)
    print(f"   Resolution: {resolution:.2f}")

    # 6. Theoretical plates
    print("\n6. Calculating theoretical plates...")
    N = lab.calculate_theoretical_plates(5.2, 0.35, "half_height")
    print(f"   Theoretical plates: {N}")
    H = lab.van_deemter_equation(flow_rate=1.0)
    print(f"   Plate height (Van Deemter): {H:.3f} mm")

    # 7. Mass spectrometry
    print("\n7. Simulating mass spectrum...")
    fragmentation = {77: 40, 105: 60, 120: 30}  # m/z: intensity
    ms_spectrum = lab.simulate_mass_spectrum("C8H10N4O2", fragmentation)  # Caffeine
    exact_mass = lab.calculate_exact_mass("C8H10N4O2")
    print(f"   Exact mass: {exact_mass:.4f} Da")
    ppm_error = lab.mass_accuracy_ppm(194.0825, exact_mass)
    print(f"   Mass accuracy: {ppm_error:.1f} ppm")

    # 8. Beer-Lambert law
    print("\n8. UV-Vis spectroscopy...")
    epsilon = 15000  # M⁻¹cm⁻¹
    path_length = 1.0  # cm
    concentration = 1e-5  # M
    absorbance = lab.beer_lambert_law(epsilon, path_length, concentration)
    transmittance = lab.calculate_transmittance(absorbance)
    print(f"   Absorbance: {absorbance:.3f}")
    print(f"   Transmittance: {transmittance:.1%}")

    # 9. Method validation
    print("\n9. Method validation statistics...")
    replicates = np.array([9.8, 10.1, 9.9, 10.2, 10.0, 9.9, 10.1])
    precision = lab.calculate_method_precision(replicates)
    print(f"   Mean: {precision['mean']:.2f} ± {precision['confidence_95']:.2f}")
    print(f"   RSD: {precision['rsd']:.1f}%")

    recovery = lab.calculate_recovery(measured=9.5, expected=10.0)
    print(f"   Recovery: {recovery:.1f}%")

    # 10. Signal processing
    print("\n10. Signal processing...")
    # Create noisy signal
    x = np.linspace(200, 800, 500)
    y = 100 * np.exp(-0.5 * ((x - 400) / 50) ** 2) + \
        50 * np.exp(-0.5 * ((x - 550) / 30) ** 2) + \
        np.random.normal(0, 2, len(x))

    noisy_signal = AnalyticalSignal(wavelength=x, intensity=y)

    # Smooth signal
    smoothed = lab.signal_smoothing(noisy_signal)

    # Detect peaks
    peaks = lab.peak_detection(smoothed, height_threshold=10)
    print(f"   Detected {len(peaks)} peaks")
    for i, peak in enumerate(peaks):
        print(f"   Peak {i+1}: λ = {peak['position']:.1f} nm, height = {peak['height']:.1f}")

    # 11. Signal-to-noise
    print("\n11. Signal-to-noise ratio...")
    snr = lab.calculate_signal_to_noise(smoothed,
                                       signal_region=(380, 420),
                                       noise_region=(700, 800))
    print(f"   S/N ratio: {snr:.1f}")

    # 12. Uncertainty propagation
    print("\n12. Uncertainty propagation...")
    values = [10.0, 2.0]
    uncertainties = [0.1, 0.05]
    result, unc = lab.uncertainty_propagation(values, uncertainties, "quotient")
    print(f"   Result: {result:.2f} ± {unc:.2f}")

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("Visit: https://aios.is | https://thegavl.com")
    print("=" * 80)

if __name__ == '__main__':
    run_demo()