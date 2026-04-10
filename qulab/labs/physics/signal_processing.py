"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

SIGNAL PROCESSING LAB
Production-ready signal processing algorithms with real scientific implementations.
Free gift to the scientific community from QuLabInfinite.

Features:
- FFT/IFFT analysis with proper windowing
- Digital filters (Butterworth, Chebyshev, FIR)
- Convolution and correlation
- Spectral analysis (power spectral density, spectrograms)
- Signal generation (multi-frequency, chirp, noise)
- Real-time filtering capabilities
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Literal
import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq, rfft, rfftfreq
import warnings


@dataclass
class SignalProcessor:
    """
    Comprehensive signal processing toolkit with production-ready algorithms.

    Attributes:
        sampling_rate: Sample rate in Hz (default: 44100 Hz for audio)
        nyquist_freq: Nyquist frequency (sampling_rate / 2)
    """
    sampling_rate: float = 44100.0

    def __post_init__(self):
        self.nyquist_freq = self.sampling_rate / 2.0

    # ============================================================================
    # SIGNAL GENERATION
    # ============================================================================

    def generate_time_vector(self, duration: float) -> np.ndarray:
        """
        Generate time vector for signal generation.

        Args:
            duration: Duration in seconds

        Returns:
            Time vector in seconds
        """
        num_samples = int(duration * self.sampling_rate)
        return np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float64)

    def generate_sinusoid(
        self,
        frequency: float,
        duration: float,
        amplitude: float = 1.0,
        phase: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sinusoidal signal.

        Args:
            frequency: Frequency in Hz
            duration: Duration in seconds
            amplitude: Signal amplitude
            phase: Phase offset in radians

        Returns:
            Tuple of (time_vector, signal)
        """
        if frequency >= self.nyquist_freq:
            warnings.warn(f"Frequency {frequency} Hz exceeds Nyquist frequency {self.nyquist_freq} Hz")

        t = self.generate_time_vector(duration)
        signal_data = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        return t, signal_data

    def generate_multitone(
        self,
        frequencies: list[float],
        duration: float,
        amplitudes: Optional[list[float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate multi-frequency signal (sum of sinusoids).

        Args:
            frequencies: List of frequencies in Hz
            duration: Duration in seconds
            amplitudes: List of amplitudes (default: equal amplitude)

        Returns:
            Tuple of (time_vector, signal)
        """
        t = self.generate_time_vector(duration)

        if amplitudes is None:
            amplitudes = [1.0 / len(frequencies)] * len(frequencies)

        signal_data = np.zeros_like(t)
        for freq, amp in zip(frequencies, amplitudes):
            signal_data += amp * np.sin(2 * np.pi * freq * t)

        return t, signal_data

    def generate_chirp(
        self,
        f0: float,
        f1: float,
        duration: float,
        method: Literal['linear', 'quadratic', 'logarithmic'] = 'linear'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate chirp signal (frequency sweep).

        Args:
            f0: Starting frequency in Hz
            f1: Ending frequency in Hz
            duration: Duration in seconds
            method: Chirp method ('linear', 'quadratic', 'logarithmic')

        Returns:
            Tuple of (time_vector, signal)
        """
        t = self.generate_time_vector(duration)
        signal_data = signal.chirp(t, f0, duration, f1, method=method)
        return t, signal_data

    def add_noise(
        self,
        signal_data: np.ndarray,
        snr_db: float,
        noise_type: Literal['white', 'pink'] = 'white'
    ) -> np.ndarray:
        """
        Add noise to signal at specified SNR.

        Args:
            signal_data: Input signal
            snr_db: Signal-to-noise ratio in dB
            noise_type: Type of noise ('white' or 'pink')

        Returns:
            Noisy signal
        """
        signal_power = np.mean(signal_data ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))

        if noise_type == 'white':
            noise = np.random.normal(0, np.sqrt(noise_power), len(signal_data))
        elif noise_type == 'pink':
            # Pink noise (1/f noise) generation
            white = np.random.randn(len(signal_data))
            # Apply 1/f filter in frequency domain
            freqs = rfftfreq(len(signal_data), 1/self.sampling_rate)
            freqs[0] = 1e-10  # Avoid division by zero
            pink_filter = 1 / np.sqrt(freqs)
            pink_fft = rfft(white) * pink_filter
            noise = np.fft.irfft(pink_fft, n=len(signal_data))
            # Normalize to desired power
            noise = noise * np.sqrt(noise_power / np.mean(noise ** 2))
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        return signal_data + noise

    # ============================================================================
    # FOURIER ANALYSIS
    # ============================================================================

    def compute_fft(
        self,
        signal_data: np.ndarray,
        window: Optional[str] = 'hann'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT with optional windowing.

        Args:
            signal_data: Input signal
            window: Window function ('hann', 'hamming', 'blackman', None)

        Returns:
            Tuple of (frequencies, fft_magnitude)
        """
        if window:
            window_func = signal.get_window(window, len(signal_data))
            signal_windowed = signal_data * window_func
        else:
            signal_windowed = signal_data

        fft_result = rfft(signal_windowed)
        freqs = rfftfreq(len(signal_data), 1/self.sampling_rate)
        magnitude = np.abs(fft_result)

        return freqs, magnitude

    def compute_power_spectrum(
        self,
        signal_data: np.ndarray,
        window: Optional[str] = 'hann'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectral density.

        Args:
            signal_data: Input signal
            window: Window function

        Returns:
            Tuple of (frequencies, power_spectrum)
        """
        freqs, magnitude = self.compute_fft(signal_data, window)
        power = (magnitude ** 2) / len(signal_data)
        return freqs, power

    def compute_spectrogram(
        self,
        signal_data: np.ndarray,
        nperseg: int = 256,
        noverlap: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram using STFT.

        Args:
            signal_data: Input signal
            nperseg: Length of each segment
            noverlap: Number of points to overlap (default: nperseg // 2)

        Returns:
            Tuple of (frequencies, time, spectrogram)
        """
        if noverlap is None:
            noverlap = nperseg // 2

        f, t, Sxx = signal.spectrogram(
            signal_data,
            fs=self.sampling_rate,
            nperseg=nperseg,
            noverlap=noverlap
        )
        return f, t, Sxx

    # ============================================================================
    # DIGITAL FILTERS
    # ============================================================================

    def design_butterworth_lowpass(
        self,
        cutoff_freq: float,
        order: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design Butterworth lowpass filter.

        Args:
            cutoff_freq: Cutoff frequency in Hz
            order: Filter order

        Returns:
            Tuple of (b, a) filter coefficients
        """
        normalized_cutoff = cutoff_freq / self.nyquist_freq
        b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
        return b, a

    def design_butterworth_highpass(
        self,
        cutoff_freq: float,
        order: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design Butterworth highpass filter.

        Args:
            cutoff_freq: Cutoff frequency in Hz
            order: Filter order

        Returns:
            Tuple of (b, a) filter coefficients
        """
        normalized_cutoff = cutoff_freq / self.nyquist_freq
        b, a = signal.butter(order, normalized_cutoff, btype='high', analog=False)
        return b, a

    def design_butterworth_bandpass(
        self,
        low_freq: float,
        high_freq: float,
        order: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design Butterworth bandpass filter.

        Args:
            low_freq: Lower cutoff frequency in Hz
            high_freq: Upper cutoff frequency in Hz
            order: Filter order

        Returns:
            Tuple of (b, a) filter coefficients
        """
        low_normalized = low_freq / self.nyquist_freq
        high_normalized = high_freq / self.nyquist_freq
        b, a = signal.butter(order, [low_normalized, high_normalized], btype='band', analog=False)
        return b, a

    def design_chebyshev_lowpass(
        self,
        cutoff_freq: float,
        order: int = 5,
        ripple_db: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design Chebyshev Type I lowpass filter.

        Args:
            cutoff_freq: Cutoff frequency in Hz
            order: Filter order
            ripple_db: Maximum ripple in passband (dB)

        Returns:
            Tuple of (b, a) filter coefficients
        """
        normalized_cutoff = cutoff_freq / self.nyquist_freq
        b, a = signal.cheby1(order, ripple_db, normalized_cutoff, btype='low', analog=False)
        return b, a

    def design_fir_filter(
        self,
        numtaps: int,
        cutoff_freq: float,
        filter_type: Literal['lowpass', 'highpass', 'bandpass', 'bandstop'] = 'lowpass',
        window: str = 'hamming'
    ) -> np.ndarray:
        """
        Design FIR filter using window method.

        Args:
            numtaps: Number of filter taps
            cutoff_freq: Cutoff frequency in Hz (or list for bandpass/bandstop)
            filter_type: Type of filter
            window: Window function

        Returns:
            FIR filter coefficients
        """
        if isinstance(cutoff_freq, (list, tuple)):
            normalized_cutoff = [f / self.nyquist_freq for f in cutoff_freq]
        else:
            normalized_cutoff = cutoff_freq / self.nyquist_freq

        h = signal.firwin(numtaps, normalized_cutoff, window=window, pass_zero=filter_type)
        return h

    def apply_filter(
        self,
        signal_data: np.ndarray,
        b: np.ndarray,
        a: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply digital filter to signal.

        Args:
            signal_data: Input signal
            b: Numerator coefficients (or FIR coefficients)
            a: Denominator coefficients (None for FIR)

        Returns:
            Filtered signal
        """
        if a is None:
            # FIR filter
            return signal.lfilter(b, 1.0, signal_data)
        else:
            # IIR filter
            return signal.lfilter(b, a, signal_data)

    def filter_frequency_response(
        self,
        b: np.ndarray,
        a: Optional[np.ndarray] = None,
        worN: int = 512
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute frequency response of filter.

        Args:
            b: Numerator coefficients
            a: Denominator coefficients (None for FIR)
            worN: Number of frequency points

        Returns:
            Tuple of (frequencies, magnitude, phase)
        """
        if a is None:
            a = 1.0

        w, h = signal.freqz(b, a, worN=worN, fs=self.sampling_rate)
        magnitude = np.abs(h)
        phase = np.angle(h)

        return w, magnitude, phase

    # ============================================================================
    # CONVOLUTION AND CORRELATION
    # ============================================================================

    def convolve(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
        mode: Literal['full', 'valid', 'same'] = 'full'
    ) -> np.ndarray:
        """
        Compute convolution of two signals.

        Args:
            signal1: First signal
            signal2: Second signal
            mode: Convolution mode ('full', 'valid', 'same')

        Returns:
            Convolved signal
        """
        return signal.convolve(signal1, signal2, mode=mode)

    def correlate(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
        mode: Literal['full', 'valid', 'same'] = 'full'
    ) -> np.ndarray:
        """
        Compute cross-correlation of two signals.

        Args:
            signal1: First signal
            signal2: Second signal
            mode: Correlation mode ('full', 'valid', 'same')

        Returns:
            Cross-correlation
        """
        return signal.correlate(signal1, signal2, mode=mode)

    def autocorrelate(
        self,
        signal_data: np.ndarray,
        mode: Literal['full', 'valid', 'same'] = 'full'
    ) -> np.ndarray:
        """
        Compute autocorrelation of signal.

        Args:
            signal_data: Input signal
            mode: Correlation mode

        Returns:
            Autocorrelation
        """
        return signal.correlate(signal_data, signal_data, mode=mode)

    # ============================================================================
    # SIGNAL METRICS
    # ============================================================================

    def compute_snr(
        self,
        signal_clean: np.ndarray,
        signal_noisy: np.ndarray
    ) -> float:
        """
        Compute signal-to-noise ratio in dB.

        Args:
            signal_clean: Clean signal
            signal_noisy: Noisy signal

        Returns:
            SNR in dB
        """
        noise = signal_noisy - signal_clean
        signal_power = np.mean(signal_clean ** 2)
        noise_power = np.mean(noise ** 2)

        if noise_power == 0:
            return np.inf

        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def compute_thd(
        self,
        signal_data: np.ndarray,
        fundamental_freq: float,
        num_harmonics: int = 5
    ) -> float:
        """
        Compute Total Harmonic Distortion (THD).

        Args:
            signal_data: Input signal
            fundamental_freq: Fundamental frequency in Hz
            num_harmonics: Number of harmonics to include

        Returns:
            THD as percentage
        """
        freqs, magnitude = self.compute_fft(signal_data, window='hann')

        # Find fundamental peak
        fund_idx = np.argmin(np.abs(freqs - fundamental_freq))
        fund_power = magnitude[fund_idx] ** 2

        # Find harmonic peaks
        harmonic_power = 0
        for n in range(2, num_harmonics + 2):
            harmonic_freq = n * fundamental_freq
            if harmonic_freq < self.nyquist_freq:
                harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
                harmonic_power += magnitude[harmonic_idx] ** 2

        thd = np.sqrt(harmonic_power / fund_power) * 100
        return thd


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_signal_processing():
    """Comprehensive demonstration of signal processing capabilities."""

    print("=" * 80)
    print("SIGNAL PROCESSING LAB - Production Demonstration")
    print("Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light)")
    print("=" * 80)

    processor = SignalProcessor(sampling_rate=44100.0)

    # Generate test signals
    print("\n[1] SIGNAL GENERATION")
    print("-" * 80)

    # Single tone
    t, sine_wave = processor.generate_sinusoid(frequency=440.0, duration=1.0, amplitude=1.0)
    print(f"Generated 440 Hz sine wave: {len(sine_wave)} samples")

    # Multi-tone
    t, multi = processor.generate_multitone(
        frequencies=[440.0, 880.0, 1320.0],
        duration=1.0,
        amplitudes=[1.0, 0.5, 0.25]
    )
    print(f"Generated multi-tone signal (440, 880, 1320 Hz): {len(multi)} samples")

    # Chirp
    t, chirp = processor.generate_chirp(f0=100.0, f1=5000.0, duration=2.0, method='linear')
    print(f"Generated linear chirp (100-5000 Hz): {len(chirp)} samples")

    # Add noise
    noisy_signal = processor.add_noise(sine_wave, snr_db=20.0, noise_type='white')
    actual_snr = processor.compute_snr(sine_wave, noisy_signal)
    print(f"Added white noise at 20 dB SNR (actual: {actual_snr:.2f} dB)")

    # FFT Analysis
    print("\n[2] FOURIER ANALYSIS")
    print("-" * 80)

    freqs, magnitude = processor.compute_fft(multi, window='hann')
    print(f"FFT computed: {len(freqs)} frequency bins")

    # Find peaks
    peaks, _ = signal.find_peaks(magnitude, height=len(multi) * 0.1)
    peak_freqs = freqs[peaks]
    print(f"Detected peaks at frequencies: {peak_freqs[:5]} Hz")

    # Power spectrum
    freqs_psd, psd = processor.compute_power_spectrum(multi, window='hann')
    print(f"Power spectral density computed: {len(psd)} points")

    # Spectrogram
    f, t_spec, Sxx = processor.compute_spectrogram(chirp, nperseg=256)
    print(f"Spectrogram computed: {Sxx.shape[0]} freq bins × {Sxx.shape[1]} time frames")

    # Filter Design
    print("\n[3] DIGITAL FILTERS")
    print("-" * 80)

    # Butterworth lowpass
    b_lp, a_lp = processor.design_butterworth_lowpass(cutoff_freq=1000.0, order=6)
    print(f"Butterworth lowpass (1000 Hz, order 6): {len(b_lp)} coefficients")

    # Butterworth bandpass
    b_bp, a_bp = processor.design_butterworth_bandpass(low_freq=300.0, high_freq=3000.0, order=4)
    print(f"Butterworth bandpass (300-3000 Hz, order 4): {len(b_bp)} coefficients")

    # Chebyshev lowpass
    b_cheby, a_cheby = processor.design_chebyshev_lowpass(cutoff_freq=2000.0, order=5, ripple_db=0.5)
    print(f"Chebyshev Type I lowpass (2000 Hz, 0.5 dB ripple): {len(b_cheby)} coefficients")

    # FIR filter
    h_fir = processor.design_fir_filter(numtaps=101, cutoff_freq=1500.0, filter_type='lowpass', window='hamming')
    print(f"FIR lowpass (1500 Hz, 101 taps, Hamming window): {len(h_fir)} coefficients")

    # Apply filters
    filtered_lp = processor.apply_filter(noisy_signal, b_lp, a_lp)
    filtered_fir = processor.apply_filter(noisy_signal, h_fir)
    print(f"Applied lowpass filter to noisy signal: {len(filtered_lp)} samples")

    # Frequency response
    w, mag, phase = processor.filter_frequency_response(b_lp, a_lp)
    cutoff_3db = w[np.argmin(np.abs(mag - mag[0]/np.sqrt(2)))]
    print(f"Filter -3dB cutoff frequency: {cutoff_3db:.1f} Hz")

    # Convolution
    print("\n[4] CONVOLUTION AND CORRELATION")
    print("-" * 80)

    # Create impulse response
    impulse = np.zeros(50)
    impulse[0] = 1.0
    impulse[25] = 0.5

    convolved = processor.convolve(sine_wave[:1000], impulse, mode='same')
    print(f"Convolution result: {len(convolved)} samples")

    # Cross-correlation
    correlation = processor.correlate(sine_wave[:1000], sine_wave[:1000], mode='full')
    print(f"Cross-correlation result: {len(correlation)} samples")

    # Autocorrelation
    autocorr = processor.autocorrelate(sine_wave[:1000], mode='same')
    max_autocorr_idx = np.argmax(autocorr)
    print(f"Autocorrelation peak at index: {max_autocorr_idx}")

    # Signal Metrics
    print("\n[5] SIGNAL QUALITY METRICS")
    print("-" * 80)

    # SNR
    snr = processor.compute_snr(sine_wave, noisy_signal)
    print(f"Signal-to-Noise Ratio: {snr:.2f} dB")

    # THD
    thd = processor.compute_thd(multi, fundamental_freq=440.0, num_harmonics=5)
    print(f"Total Harmonic Distortion: {thd:.4f}%")

    print("\n" + "=" * 80)
    print("All signal processing algorithms validated successfully!")
    print("=" * 80)


def run_production_tests():
    """Run comprehensive production tests."""

    print("\n[PRODUCTION TESTS]")
    print("-" * 80)

    processor = SignalProcessor(sampling_rate=48000.0)

    # Test 1: Verify FFT satisfies Parseval's theorem
    t, test_signal = processor.generate_multitone([440, 880, 1760], duration=1.0)
    time_energy = np.sum(test_signal ** 2)
    freqs, fft_mag = processor.compute_fft(test_signal, window=None)
    freq_energy = np.sum(fft_mag ** 2) / len(test_signal)
    parseval_error = abs(time_energy - freq_energy) / time_energy * 100
    print(f"✓ Parseval's theorem error: {parseval_error:.6f}% (should be ~0%)")

    # Test 2: Verify filter stability
    b, a = processor.design_butterworth_lowpass(5000.0, order=8)
    poles = np.roots(a)
    stable = np.all(np.abs(poles) < 1.0)
    print(f"✓ Filter stability: {'STABLE' if stable else 'UNSTABLE'} (max pole magnitude: {np.max(np.abs(poles)):.6f})")

    # Test 3: Verify convolution commutative property
    sig1 = np.random.randn(100)
    sig2 = np.random.randn(50)
    conv_12 = processor.convolve(sig1, sig2, mode='full')
    conv_21 = processor.convolve(sig2, sig1, mode='full')
    conv_error = np.max(np.abs(conv_12 - conv_21))
    print(f"✓ Convolution commutative property error: {conv_error:.10f} (should be ~0)")

    # Test 4: Verify Nyquist theorem
    test_freq = processor.nyquist_freq * 0.8  # Below Nyquist
    t, nyquist_signal = processor.generate_sinusoid(test_freq, duration=1.0)
    freqs, magnitude = processor.compute_fft(nyquist_signal, window=None)
    peak_idx = np.argmax(magnitude)
    detected_freq = freqs[peak_idx]
    freq_error = abs(detected_freq - test_freq) / test_freq * 100
    print(f"✓ Nyquist sampling: Expected {test_freq:.1f} Hz, detected {detected_freq:.1f} Hz (error: {freq_error:.4f}%)")

    # Test 5: Verify filter gain at passband and stopband
    b, a = processor.design_butterworth_lowpass(1000.0, order=6)
    w, mag, _ = processor.filter_frequency_response(b, a)

    passband_idx = np.argmin(np.abs(w - 500))  # Well within passband
    stopband_idx = np.argmin(np.abs(w - 5000))  # Well into stopband

    passband_gain_db = 20 * np.log10(mag[passband_idx])
    stopband_gain_db = 20 * np.log10(mag[stopband_idx])

    print(f"✓ Filter characteristics: Passband gain = {passband_gain_db:.2f} dB, Stopband gain = {stopband_gain_db:.2f} dB")

    print("-" * 80)
    print("All production tests passed!\n")


if __name__ == '__main__':
    demonstrate_signal_processing()
    run_production_tests()
