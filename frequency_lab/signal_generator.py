"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

signal_generator - Part of Frequency Lab
"""


import numpy as np

class SignalGenerator:
    def __init__(self, sample_rate, duration):
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)
        self.t = np.arange(self.num_samples) / sample_rate

    def generate_tone(self, frequency, amplitude=0.5):
        """
        Generate a simple sine wave (tone).
        """
        print(f"Generating tone at {frequency} Hz...")
        signal = amplitude * np.exp(2j * np.pi * frequency * self.t)
        return signal

    def generate_fm_signal(self, carrier_freq, modulator_freq, deviation, amplitude=0.5):
        """
        Generate an FM modulated signal.
        """
        print(f"Generating FM signal with carrier {carrier_freq} Hz and modulator {modulator_freq} Hz...")
        # Integrator for phase
        modulator_wave = np.sin(2 * np.pi * modulator_freq * self.t)
        # Phase modulation
        phase = 2 * np.pi * deviation * np.cumsum(modulator_wave) / self.sample_rate
        # FM Signal
        fm_signal = amplitude * np.exp(2j * (2 * np.pi * carrier_freq * self.t + phase))
        return fm_signal

    def generate_noise(self, amplitude=0.1):
        """
        Generate complex white Gaussian noise.
        """
        print("Generating white noise...")
        noise = amplitude * (np.random.randn(self.num_samples) + 1j * np.random.randn(self.num_samples))
        return noise
        
    def generate_chirp(self, start_freq, end_freq, amplitude=0.5):
        """
        Generate a frequency chirp.
        """
        print(f"Generating chirp from {start_freq} Hz to {end_freq} Hz...")
        instantaneous_freq = np.linspace(start_freq, end_freq, self.num_samples)
        phase = 2 * np.pi * np.cumsum(instantaneous_freq) / self.sample_rate
        chirp_signal = amplitude * np.exp(2j * phase)
        return chirp_signal
