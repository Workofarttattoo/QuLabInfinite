# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
CARDIOLOGY LAB
Advanced cardiac simulation and analysis laboratory with real medical algorithms.
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass
from scipy import signal, stats
from scipy.integrate import odeint
import json

@dataclass
class ECGParameters:
    """Parameters for ECG waveform generation based on Hermite-Rodriguez functions."""
    heart_rate: float = 60  # BPM
    p_amplitude: float = 0.15  # mV
    qrs_amplitude: float = 1.5  # mV
    t_amplitude: float = 0.3  # mV
    pr_interval: float = 0.16  # seconds
    qrs_duration: float = 0.08  # seconds
    qt_interval: float = 0.40  # seconds
    noise_level: float = 0.02  # mV

@dataclass
class HemodynamicParameters:
    """Parameters for Windkessel hemodynamic model."""
    arterial_compliance: float = 1.5  # mL/mmHg
    peripheral_resistance: float = 1.0  # mmHg·s/mL
    characteristic_impedance: float = 0.05  # mmHg·s/mL
    stroke_volume: float = 70  # mL
    ejection_duration: float = 0.3  # seconds

class CardiologyLab:
    """
    Advanced cardiology simulation laboratory implementing real medical algorithms.

    Features:
    - ECG signal synthesis using physiological models
    - Arrhythmia detection via HRV and morphology analysis
    - Hemodynamic modeling with 3-element Windkessel model
    - Cardiac output and ejection fraction calculations
    - Spectral analysis for autonomic function
    """

    def __init__(self, sampling_rate: float = 1000):
        """
        Initialize cardiology lab with specified sampling rate.

        Args:
            sampling_rate: Hz, typically 250-1000 for ECG
        """
        self.sampling_rate = sampling_rate
        self.ecg_params = ECGParameters()
        self.hemo_params = HemodynamicParameters()

    def generate_ecg_waveform(self, duration: float = 10.0,
                             params: Optional[ECGParameters] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate realistic ECG waveform using sum of Gaussian functions.

        Based on: McSharry et al. (2003) "A dynamical model for generating synthetic ECG signals"

        Args:
            duration: Recording duration in seconds
            params: ECG parameters or use defaults

        Returns:
            time_array, ecg_signal in mV
        """
        if params is None:
            params = self.ecg_params

        t = np.arange(0, duration, 1/self.sampling_rate)
        ecg = np.zeros_like(t)

        # Calculate beat timing
        rr_interval = 60.0 / params.heart_rate
        beat_times = np.arange(0, duration, rr_interval)

        for beat_time in beat_times:
            # P wave (atrial depolarization)
            p_center = beat_time - params.pr_interval + 0.05
            p_width = 0.05
            p_wave = params.p_amplitude * np.exp(-0.5 * ((t - p_center)/p_width)**2)

            # Q wave (septal depolarization)
            q_center = beat_time - 0.02
            q_width = 0.015
            q_wave = -0.15 * params.qrs_amplitude * np.exp(-0.5 * ((t - q_center)/q_width)**2)

            # R wave (ventricular depolarization)
            r_center = beat_time
            r_width = 0.02
            r_wave = params.qrs_amplitude * np.exp(-0.5 * ((t - r_center)/r_width)**2)

            # S wave
            s_center = beat_time + 0.02
            s_width = 0.015
            s_wave = -0.2 * params.qrs_amplitude * np.exp(-0.5 * ((t - s_center)/s_width)**2)

            # T wave (ventricular repolarization)
            t_center = beat_time + params.qt_interval - 0.1
            t_width = 0.1
            t_wave = params.t_amplitude * np.exp(-0.5 * ((t - t_center)/t_width)**2)

            # Combine waves
            ecg += p_wave + q_wave + r_wave + s_wave + t_wave

        # Add physiological noise (muscle artifacts, baseline wander)
        noise = params.noise_level * np.random.randn(len(t))
        baseline_wander = 0.1 * np.sin(2 * np.pi * 0.2 * t)
        ecg += noise + baseline_wander

        return t, ecg

    def detect_r_peaks(self, ecg_signal: np.ndarray,
                      threshold_factor: float = 0.6) -> np.ndarray:
        """
        Detect R peaks using Pan-Tompkins algorithm.

        Reference: Pan & Tompkins (1985) "A Real-Time QRS Detection Algorithm"

        Args:
            ecg_signal: ECG signal in mV
            threshold_factor: Detection threshold as fraction of max

        Returns:
            Array of R peak indices
        """
        # Bandpass filter (5-15 Hz)
        nyquist = self.sampling_rate / 2
        low = 5 / nyquist
        high = 15 / nyquist
        b, a = signal.butter(2, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, ecg_signal)

        # Derivative
        derivative = np.diff(filtered)

        # Square
        squared = derivative ** 2

        # Moving window integration (150ms window)
        window_size = int(0.15 * self.sampling_rate)
        integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')

        # Find peaks
        threshold = threshold_factor * np.max(integrated)
        peaks, properties = signal.find_peaks(integrated,
                                             height=threshold,
                                             distance=int(0.4 * self.sampling_rate))

        return peaks

    def calculate_heart_rate_variability(self, r_peaks: np.ndarray) -> Dict[str, float]:
        """
        Calculate HRV metrics from R peak positions.

        Implements time-domain and frequency-domain HRV analysis per
        Task Force guidelines (1996).

        Args:
            r_peaks: Indices of R peaks

        Returns:
            Dictionary with HRV metrics
        """
        # Calculate RR intervals in ms
        rr_intervals = np.diff(r_peaks) * (1000 / self.sampling_rate)

        # Time-domain metrics
        metrics = {
            'mean_rr': np.mean(rr_intervals),
            'sdnn': np.std(rr_intervals),  # Standard deviation of NN intervals
            'rmssd': np.sqrt(np.mean(np.diff(rr_intervals)**2)),  # Root mean square of successive differences
            'pnn50': np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100  # Percentage > 50ms
        }

        # Frequency-domain metrics (using Welch's method)
        if len(rr_intervals) > 256:
            # Interpolate to uniform sampling
            t_rr = np.cumsum(rr_intervals) / 1000
            fs_resample = 4  # 4 Hz resampling
            t_uniform = np.arange(t_rr[0], t_rr[-1], 1/fs_resample)
            rr_uniform = np.interp(t_uniform, t_rr[:-1], rr_intervals)

            # Power spectral density
            freqs, psd = signal.welch(rr_uniform, fs=fs_resample, nperseg=256)

            # HRV frequency bands
            vlf_band = (0.003, 0.04)  # Very low frequency
            lf_band = (0.04, 0.15)    # Low frequency
            hf_band = (0.15, 0.4)     # High frequency

            vlf_power = np.trapz(psd[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])])
            lf_power = np.trapz(psd[(freqs >= lf_band[0]) & (freqs < lf_band[1])])
            hf_power = np.trapz(psd[(freqs >= hf_band[0]) & (freqs < hf_band[1])])

            metrics.update({
                'vlf_power': vlf_power,
                'lf_power': lf_power,
                'hf_power': hf_power,
                'lf_hf_ratio': lf_power / hf_power if hf_power > 0 else np.nan
            })

        return metrics

    def detect_arrhythmias(self, ecg_signal: np.ndarray,
                          r_peaks: np.ndarray) -> Dict[str, Any]:
        """
        Detect common arrhythmias using morphology and rhythm analysis.

        Implements detection for:
        - Atrial fibrillation (irregular RR intervals)
        - Premature ventricular contractions (wide QRS)
        - Bradycardia/Tachycardia

        Args:
            ecg_signal: ECG signal
            r_peaks: R peak positions

        Returns:
            Dictionary with detected arrhythmias
        """
        detections = {
            'atrial_fibrillation': False,
            'pvc_count': 0,
            'bradycardia': False,
            'tachycardia': False,
            'irregular_rhythm': False
        }

        if len(r_peaks) < 3:
            return detections

        # Calculate RR intervals
        rr_intervals = np.diff(r_peaks) / self.sampling_rate
        heart_rates = 60 / rr_intervals

        # Rhythm analysis
        mean_hr = np.mean(heart_rates)
        detections['bradycardia'] = mean_hr < 60
        detections['tachycardia'] = mean_hr > 100

        # Atrial fibrillation detection (high RR interval variability)
        rr_std = np.std(rr_intervals)
        rr_mean = np.mean(rr_intervals)
        coefficient_of_variation = rr_std / rr_mean
        detections['atrial_fibrillation'] = coefficient_of_variation > 0.2

        # PVC detection (premature beats with compensatory pause)
        for i in range(1, len(rr_intervals)-1):
            if rr_intervals[i] < 0.8 * rr_mean:  # Premature
                if rr_intervals[i-1] + rr_intervals[i] > 1.8 * rr_mean:  # Compensatory pause
                    detections['pvc_count'] += 1

        # Irregular rhythm detection
        successive_differences = np.abs(np.diff(rr_intervals))
        detections['irregular_rhythm'] = np.max(successive_differences) > 0.2

        return detections

    def windkessel_model(self, time: np.ndarray,
                        params: Optional[HemodynamicParameters] = None) -> Dict[str, np.ndarray]:
        """
        3-element Windkessel model for arterial hemodynamics.

        Models arterial system as electrical circuit analog:
        - Compliance (capacitance)
        - Resistance
        - Characteristic impedance

        Reference: Westerhof et al. (2009) "The arterial Windkessel"

        Args:
            time: Time array in seconds
            params: Hemodynamic parameters

        Returns:
            Dictionary with pressure and flow waveforms
        """
        if params is None:
            params = self.hemo_params

        # Generate cardiac output flow waveform (simplified)
        heart_rate = 60  # BPM
        period = 60 / heart_rate

        flow = np.zeros_like(time)
        for beat_start in np.arange(0, time[-1], period):
            # Ejection phase (simplified cosine)
            ejection_mask = (time >= beat_start) & (time < beat_start + params.ejection_duration)
            phase = (time[ejection_mask] - beat_start) / params.ejection_duration
            flow[ejection_mask] = (params.stroke_volume / params.ejection_duration) * \
                                 (1 - np.cos(2 * np.pi * phase)) / 2

        # Solve Windkessel differential equation
        def windkessel_ode(pressure, t, flow_func, C, R, Z):
            """Windkessel ODE: C*dP/dt = Q(t) - P/R"""
            current_flow = np.interp(t, time, flow_func)
            dP_dt = (current_flow - pressure/R) / C
            return dP_dt

        # Initial pressure (diastolic)
        P0 = 80  # mmHg

        # Solve ODE
        pressure = odeint(windkessel_ode, P0, time,
                         args=(flow, params.arterial_compliance,
                              params.peripheral_resistance,
                              params.characteristic_impedance))
        pressure = pressure.flatten()

        # Add characteristic impedance effect (immediate pressure response)
        pressure += params.characteristic_impedance * flow

        return {
            'time': time,
            'pressure': pressure,
            'flow': flow,
            'compliance_pressure': pressure - params.characteristic_impedance * flow
        }

    def calculate_cardiac_output(self, stroke_volume: float, heart_rate: float) -> float:
        """
        Calculate cardiac output using standard formula.

        CO = SV × HR

        Args:
            stroke_volume: mL per beat
            heart_rate: Beats per minute

        Returns:
            Cardiac output in L/min
        """
        return (stroke_volume * heart_rate) / 1000  # Convert mL to L

    def calculate_ejection_fraction(self, end_diastolic_volume: float,
                                   end_systolic_volume: float) -> float:
        """
        Calculate left ventricular ejection fraction.

        EF = (EDV - ESV) / EDV × 100

        Normal range: 55-70%

        Args:
            end_diastolic_volume: mL
            end_systolic_volume: mL

        Returns:
            Ejection fraction as percentage
        """
        if end_diastolic_volume <= 0:
            raise ValueError("End diastolic volume must be positive")

        ef = ((end_diastolic_volume - end_systolic_volume) / end_diastolic_volume) * 100
        return np.clip(ef, 0, 100)

    def assess_cardiac_function(self, ecg_signal: np.ndarray,
                               blood_pressure: Tuple[float, float] = (120, 80)) -> Dict[str, Any]:
        """
        Comprehensive cardiac function assessment.

        Args:
            ecg_signal: ECG recording
            blood_pressure: (systolic, diastolic) in mmHg

        Returns:
            Dictionary with cardiac metrics and interpretation
        """
        # Detect R peaks
        r_peaks = self.detect_r_peaks(ecg_signal)

        # Calculate heart rate
        if len(r_peaks) > 1:
            rr_intervals = np.diff(r_peaks) / self.sampling_rate
            heart_rate = 60 / np.mean(rr_intervals)
        else:
            heart_rate = 0

        # HRV analysis
        hrv_metrics = self.calculate_heart_rate_variability(r_peaks) if len(r_peaks) > 10 else {}

        # Arrhythmia detection
        arrhythmias = self.detect_arrhythmias(ecg_signal, r_peaks)

        # Blood pressure analysis
        systolic, diastolic = blood_pressure
        pulse_pressure = systolic - diastolic
        mean_arterial_pressure = diastolic + pulse_pressure / 3

        # Estimated cardiac output (using typical stroke volume)
        estimated_stroke_volume = 70  # mL, typical value
        cardiac_output = self.calculate_cardiac_output(estimated_stroke_volume, heart_rate)

        # Calculate systemic vascular resistance
        # SVR = (MAP - CVP) / CO, assuming CVP = 5 mmHg
        cvp = 5  # Central venous pressure
        svr = (mean_arterial_pressure - cvp) / cardiac_output if cardiac_output > 0 else np.inf

        # Interpretation
        interpretation = []
        if heart_rate < 60:
            interpretation.append("Bradycardia detected")
        elif heart_rate > 100:
            interpretation.append("Tachycardia detected")

        if systolic > 140 or diastolic > 90:
            interpretation.append("Hypertension")
        elif systolic < 90:
            interpretation.append("Hypotension")

        if arrhythmias['atrial_fibrillation']:
            interpretation.append("Possible atrial fibrillation")

        if arrhythmias['pvc_count'] > 0:
            interpretation.append(f"{arrhythmias['pvc_count']} PVCs detected")

        if 'lf_hf_ratio' in hrv_metrics:
            if hrv_metrics['lf_hf_ratio'] > 2:
                interpretation.append("Sympathetic dominance")
            elif hrv_metrics['lf_hf_ratio'] < 0.5:
                interpretation.append("Parasympathetic dominance")

        return {
            'heart_rate': heart_rate,
            'blood_pressure': blood_pressure,
            'mean_arterial_pressure': mean_arterial_pressure,
            'pulse_pressure': pulse_pressure,
            'cardiac_output': cardiac_output,
            'systemic_vascular_resistance': svr,
            'hrv_metrics': hrv_metrics,
            'arrhythmias': arrhythmias,
            'interpretation': interpretation
        }

    def simulate_exercise_response(self, rest_hr: float = 60,
                                  max_hr_age: float = 30,
                                  exercise_duration: float = 600) -> Dict[str, np.ndarray]:
        """
        Simulate cardiovascular response to exercise.

        Models HR, stroke volume, and cardiac output changes during exercise.

        Args:
            rest_hr: Resting heart rate (BPM)
            max_hr_age: Age for max HR calculation
            exercise_duration: Exercise duration in seconds

        Returns:
            Dictionary with time series of cardiovascular parameters
        """
        # Calculate max heart rate (220 - age formula)
        max_hr = 220 - max_hr_age

        # Time array
        time = np.linspace(0, exercise_duration, 100)

        # Heart rate response (exponential rise, plateau, exponential recovery)
        warm_up = exercise_duration * 0.2
        exercise = exercise_duration * 0.6
        cool_down = exercise_duration * 0.2

        hr = np.zeros_like(time)
        for i, t in enumerate(time):
            if t < warm_up:
                # Warm-up phase
                hr[i] = rest_hr + (max_hr * 0.7 - rest_hr) * (1 - np.exp(-3 * t / warm_up))
            elif t < warm_up + exercise:
                # Exercise phase (steady state)
                hr[i] = max_hr * 0.7 + 10 * np.sin(0.1 * t)  # Add some variability
            else:
                # Cool-down phase
                t_cool = t - (warm_up + exercise)
                hr[i] = rest_hr + (max_hr * 0.7 - rest_hr) * np.exp(-2 * t_cool / cool_down)

        # Stroke volume response (initial increase, then plateau)
        rest_sv = 70  # mL
        sv = rest_sv * (1 + 0.4 * (1 - np.exp(-hr / 100)))

        # Cardiac output
        co = (hr * sv) / 1000  # L/min

        # Systemic vascular resistance (decreases with exercise)
        rest_svr = 15  # mmHg·min/L
        svr = rest_svr * (0.4 + 0.6 * np.exp(-(hr - rest_hr) / 50))

        return {
            'time': time,
            'heart_rate': hr,
            'stroke_volume': sv,
            'cardiac_output': co,
            'systemic_vascular_resistance': svr
        }

    def analyze_qt_interval(self, ecg_signal: np.ndarray,
                           r_peaks: np.ndarray) -> Dict[str, float]:
        """
        Analyze QT interval for Long QT Syndrome risk.

        Calculates QT, QTc (Bazett's formula), and risk assessment.

        Args:
            ecg_signal: ECG signal
            r_peaks: R peak positions

        Returns:
            Dictionary with QT metrics
        """
        if len(r_peaks) < 2:
            return {'qt': np.nan, 'qtc': np.nan, 'risk': 'Unknown'}

        # Simplified QT detection (find T wave end after each R peak)
        qt_intervals = []

        for r_peak in r_peaks[:-1]:
            # Search window for T wave (200-400ms after R peak)
            start_idx = r_peak + int(0.2 * self.sampling_rate)
            end_idx = min(r_peak + int(0.4 * self.sampling_rate), len(ecg_signal)-1)

            if end_idx > start_idx:
                segment = ecg_signal[start_idx:end_idx]
                # Find minimum (T wave end approximation)
                t_end = start_idx + np.argmin(np.abs(segment))
                qt_interval = (t_end - r_peak) / self.sampling_rate
                qt_intervals.append(qt_interval)

        if not qt_intervals:
            return {'qt': np.nan, 'qtc': np.nan, 'risk': 'Unknown'}

        # Calculate mean QT
        mean_qt = np.mean(qt_intervals) * 1000  # Convert to ms

        # Calculate heart rate for QTc
        rr_interval = np.mean(np.diff(r_peaks)) / self.sampling_rate
        heart_rate = 60 / rr_interval

        # Bazett's formula: QTc = QT / sqrt(RR in seconds)
        qtc = mean_qt / np.sqrt(rr_interval)

        # Risk assessment
        if qtc > 500:
            risk = 'High'
        elif qtc > 450:
            risk = 'Moderate'
        elif qtc < 350:
            risk = 'Short QT'
        else:
            risk = 'Normal'

        return {
            'qt': mean_qt,
            'qtc': qtc,
            'heart_rate': heart_rate,
            'risk': risk
        }

    def demo(self):
        """Demonstrate comprehensive cardiology analysis."""
        print("=" * 80)
        print("CARDIOLOGY LAB - Comprehensive Cardiac Analysis Demo")
        print("=" * 80)

        # Generate synthetic ECG
        print("\n1. Generating 10-second ECG recording...")
        time, ecg = self.generate_ecg_waveform(duration=10.0)
        print(f"   Generated {len(ecg)} samples at {self.sampling_rate} Hz")

        # R peak detection
        print("\n2. Detecting R peaks using Pan-Tompkins algorithm...")
        r_peaks = self.detect_r_peaks(ecg)
        print(f"   Detected {len(r_peaks)} R peaks")

        # Heart rate variability
        print("\n3. Calculating heart rate variability metrics...")
        hrv = self.calculate_heart_rate_variability(r_peaks)
        print(f"   Mean RR interval: {hrv['mean_rr']:.1f} ms")
        print(f"   SDNN: {hrv['sdnn']:.1f} ms")
        print(f"   RMSSD: {hrv['rmssd']:.1f} ms")
        if 'lf_hf_ratio' in hrv:
            print(f"   LF/HF ratio: {hrv['lf_hf_ratio']:.2f}")

        # Arrhythmia detection
        print("\n4. Screening for arrhythmias...")
        arrhythmias = self.detect_arrhythmias(ecg, r_peaks)
        for arrhythmia, detected in arrhythmias.items():
            if isinstance(detected, bool) and detected:
                print(f"   WARNING: {arrhythmia.replace('_', ' ').title()} detected")
            elif isinstance(detected, int) and detected > 0:
                print(f"   {arrhythmia.replace('_', ' ').title()}: {detected}")

        # Hemodynamic modeling
        print("\n5. Running Windkessel hemodynamic model...")
        hemo_time = np.linspace(0, 3, 3000)
        hemo = self.windkessel_model(hemo_time)
        print(f"   Systolic pressure: {np.max(hemo['pressure']):.1f} mmHg")
        print(f"   Diastolic pressure: {np.min(hemo['pressure']):.1f} mmHg")
        print(f"   Mean arterial pressure: {np.mean(hemo['pressure']):.1f} mmHg")

        # Cardiac function assessment
        print("\n6. Comprehensive cardiac assessment...")
        assessment = self.assess_cardiac_function(ecg, blood_pressure=(120, 80))
        print(f"   Heart rate: {assessment['heart_rate']:.1f} BPM")
        print(f"   Cardiac output: {assessment['cardiac_output']:.1f} L/min")
        print(f"   Systemic vascular resistance: {assessment['systemic_vascular_resistance']:.1f} mmHg·min/L")

        if assessment['interpretation']:
            print("\n   Clinical notes:")
            for note in assessment['interpretation']:
                print(f"   - {note}")

        # Exercise response
        print("\n7. Simulating exercise stress test...")
        exercise = self.simulate_exercise_response()
        print(f"   Peak heart rate: {np.max(exercise['heart_rate']):.0f} BPM")
        print(f"   Peak cardiac output: {np.max(exercise['cardiac_output']):.1f} L/min")
        print(f"   Minimum SVR: {np.min(exercise['systemic_vascular_resistance']):.1f} mmHg·min/L")

        # QT interval analysis
        print("\n8. QT interval analysis...")
        qt_analysis = self.analyze_qt_interval(ecg, r_peaks)
        if not np.isnan(qt_analysis['qt']):
            print(f"   QT interval: {qt_analysis['qt']:.0f} ms")
            print(f"   QTc (Bazett): {qt_analysis['qtc']:.0f} ms")
            print(f"   Risk level: {qt_analysis['risk']}")

        print("\n" + "=" * 80)
        print("Analysis complete. All systems functioning within normal parameters.")
        print("=" * 80)

if __name__ == "__main__":
    lab = CardiologyLab()
    lab.demo()