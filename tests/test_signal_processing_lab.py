import pytest
import sys
import importlib

# Instead of complex mock classes, we will use pytest.importorskip
# This ensures that we test the real implementation when dependencies are present,
# and skip the tests gracefully in environments without them.

np = pytest.importorskip("numpy")
scipy = pytest.importorskip("scipy")
import scipy.signal
from scipy.fft import fft, ifft, fftfreq, rfft, rfftfreq

import signal_processing_lab

class TestSignalProcessor:
    @pytest.fixture
    def processor(self):
        return signal_processing_lab.SignalProcessor(sampling_rate=1000.0)

    def test_init(self, processor):
        assert processor.sampling_rate == 1000.0
        assert processor.nyquist_freq == 500.0

    def test_generate_time_vector(self, processor):
        t = processor.generate_time_vector(1.0)
        assert len(t) == 1000
        assert t[0] == 0.0

    def test_generate_sinusoid(self, processor):
        t, sig = processor.generate_sinusoid(10.0, 1.0)
        assert len(t) == 1000
        assert len(sig) == 1000

    def test_design_butterworth_lowpass(self, processor):
        b, a = processor.design_butterworth_lowpass(100.0, 4)
        assert b is not None
        assert a is not None

    def test_apply_filter(self, processor):
        t, sig = processor.generate_sinusoid(10.0, 1.0)
        b, a = processor.design_butterworth_lowpass(100.0, 4)
        filtered = processor.apply_filter(sig, b, a)
        assert len(filtered) == 1000

    def test_compute_fft(self, processor):
        t, sig = processor.generate_sinusoid(10.0, 1.0)
        freqs, mag = processor.compute_fft(sig, window=None)
        assert len(freqs) > 0
        assert len(mag) > 0
        assert len(freqs) == len(mag)

    def test_compute_snr(self, processor):
        clean = processor.generate_time_vector(1.0)
        noisy = clean + 0.1
        snr = processor.compute_snr(clean, noisy)
        assert isinstance(snr, float)

    def test_add_noise(self, processor):
        t, sig = processor.generate_sinusoid(10.0, 1.0)
        noisy = processor.add_noise(sig, 20.0, 'white')
        assert len(noisy) == 1000

    def test_correlate(self, processor):
        t, sig = processor.generate_sinusoid(10.0, 1.0)
        corr = processor.correlate(sig, sig, mode='same')
        assert len(corr) == 1000

    def test_design_butterworth_highpass(self, processor):
        b, a = processor.design_butterworth_highpass(100.0, 4)
        assert b is not None
        assert a is not None

    def test_design_butterworth_bandpass(self, processor):
        b, a = processor.design_butterworth_bandpass(100.0, 200.0, 4)
        assert b is not None
        assert a is not None

    def test_design_chebyshev_lowpass(self, processor):
        b, a = processor.design_chebyshev_lowpass(100.0, 4, 0.5)
        assert b is not None
        assert a is not None

    def test_design_fir_filter(self, processor):
        h = processor.design_fir_filter(101, 100.0, 'lowpass')
        assert h is not None

    def test_filter_frequency_response(self, processor):
        b, a = processor.design_butterworth_lowpass(100.0, 4)
        w, mag, phase = processor.filter_frequency_response(b, a)
        assert len(w) == 512
        assert len(mag) == 512
        assert len(phase) == 512

    def test_generate_multitone(self, processor):
        t, sig = processor.generate_multitone([10.0, 20.0], 1.0)
        assert len(sig) == 1000

    def test_generate_chirp(self, processor):
        t, sig = processor.generate_chirp(10.0, 100.0, 1.0)
        assert len(sig) == 1000

    def test_compute_power_spectrum(self, processor):
        t, sig = processor.generate_sinusoid(10.0, 1.0)
        freqs, psd = processor.compute_power_spectrum(sig)
        assert len(freqs) > 0
        assert len(psd) > 0

    def test_compute_spectrogram(self, processor):
        t, sig = processor.generate_sinusoid(10.0, 1.0)
        f, t_spec, sxx = processor.compute_spectrogram(sig)
        assert len(f) > 0

    def test_convolve(self, processor):
        sig1 = processor.generate_time_vector(0.1)
        sig2 = processor.generate_time_vector(0.05)
        conv = processor.convolve(sig1, sig2, mode='full')
        assert len(conv) > 0

    def test_autocorrelate(self, processor):
        sig1 = processor.generate_time_vector(0.1)
        acorr = processor.autocorrelate(sig1, mode='full')
        assert len(acorr) > 0

    def test_compute_thd(self, processor):
        t, sig = processor.generate_sinusoid(10.0, 1.0)
        thd = processor.compute_thd(sig, 10.0, 3)
        assert isinstance(thd, float)
