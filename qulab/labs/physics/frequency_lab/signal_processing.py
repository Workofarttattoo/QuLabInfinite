
import numpy as np
from scipy import signal

class SignalProcessor:
    def calculate_psd(self, samples, sample_rate, nfft=1024):
        """
        Calculate the Power Spectral Density (PSD) of a signal.
        """
        if samples is None:
            return None, None
        
        freqs, psd = signal.welch(
            samples,
            fs=sample_rate,
            nperseg=nfft,
            nfft=nfft,
            return_onesided=False,
            scaling='density'
        )
        
        # Shift the frequencies to be centered at 0
        freqs = np.fft.fftshift(freqs)
        psd = np.fft.fftshift(psd)
        
        return freqs, 10 * np.log10(psd) # convert to dB

    def filter_signal(self, samples, filter_type, cutoff_freq, sample_rate, order=5):
        """
        Apply a digital filter to the signal.
        """
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        
        b, a = signal.butter(order, normal_cutoff, btype=filter_type, analog=False)
        filtered_samples = signal.lfilter(b, a, samples)
        
        return filtered_samples

    def demodulate_fm(self, samples):
        """
        Demodulate an FM signal.
        This is a very basic implementation.
        """
        # Simple FM demodulation using differentiation and angle
        differentiated_signal = np.diff(samples)
        # Calculate the instantaneous phase angle
        instantaneous_phase = np.angle(differentiated_signal)
        # The instantaneous frequency is the derivative of the phase
        # The derivative can be approximated by the difference
        demodulated_signal = np.diff(np.unwrap(instantaneous_phase))
        return demodulated_signal
