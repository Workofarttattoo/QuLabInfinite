
import pytest
import numpy as np
from cardiology_lab import CardiologyLaboratory, ECGParameters, HemodynamicParameters, CardiacDrug

class TestCardiologyLabMerged:

    def setup_method(self):
        self.lab = CardiologyLaboratory(seed=42)

    def test_original_functionality_cardiac_cycle(self):
        """Test the cardiac cycle simulation from the original package."""
        cycle = self.lab.simulate_cardiac_cycle(heart_rate=70, contractility=1.0, duration_s=1.0)
        assert 'heart_rate' in cycle
        assert 'stroke_volume' in cycle
        assert 'cardiac_output' in cycle
        assert cycle['heart_rate'] == 70

    def test_original_functionality_ecg_signal(self):
        """Test the basic ECG signal generation from the original package."""
        ecg = self.lab.generate_ecg_signal(heart_rate=70, rhythm='normal_sinus', duration_s=1.0)
        assert ecg.heart_rate == 70
        assert len(ecg.signal) == 500  # Default 500Hz * 1s

    def test_original_functionality_blood_flow(self):
        """Test blood flow calculation from the original package."""
        flow = self.lab.calculate_blood_flow(vessel_radius_mm=2.0, vessel_length_cm=5.0, pressure_drop_mmHg=50.0)
        assert flow.flow_rate > 0
        assert flow.velocity > 0

    def test_merged_functionality_ecg_waveform(self):
        """Test the advanced ECG waveform generation from the root file."""
        t, ecg = self.lab.generate_ecg_waveform(duration=2.0)
        assert len(t) == len(ecg)
        assert len(t) == 2.0 * self.lab.sampling_rate

    def test_merged_functionality_r_peak_detection(self):
        """Test R-peak detection from the root file."""
        t, ecg = self.lab.generate_ecg_waveform(duration=5.0)
        peaks = self.lab.detect_r_peaks(ecg)
        # With default 60 BPM, we expect around 5 beats in 5 seconds
        assert 3 <= len(peaks) <= 7

    def test_merged_functionality_hrv(self):
        """Test HRV calculation from the root file."""
        # Generate a longer signal to ensure enough peaks for HRV
        t, ecg = self.lab.generate_ecg_waveform(duration=10.0)
        peaks = self.lab.detect_r_peaks(ecg)
        hrv = self.lab.calculate_heart_rate_variability(peaks)
        assert 'mean_rr' in hrv
        assert 'sdnn' in hrv

    def test_merged_functionality_windkessel(self):
        """Test Windkessel model from the root file."""
        time = np.linspace(0, 1, 1000)
        hemo = self.lab.windkessel_model(time)
        assert 'pressure' in hemo
        assert 'flow' in hemo
        assert len(hemo['pressure']) == len(time)

    def test_merged_functionality_qt_analysis(self):
        """Test QT interval analysis from the root file."""
        t, ecg = self.lab.generate_ecg_waveform(duration=5.0)
        peaks = self.lab.detect_r_peaks(ecg)
        qt = self.lab.analyze_qt_interval(ecg, peaks)
        assert 'qt' in qt
        assert 'qtc' in qt

    def test_merged_functionality_exercise(self):
        """Test exercise response simulation from the root file."""
        exercise = self.lab.simulate_exercise_response(exercise_duration=60)
        assert 'heart_rate' in exercise
        assert len(exercise['heart_rate']) == 100 # default 100 points
