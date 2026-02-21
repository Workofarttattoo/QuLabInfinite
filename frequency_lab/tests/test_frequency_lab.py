"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

test_frequency_lab - Part of Frequency Lab
"""


import unittest
import numpy as np
from ..frequency_lab import FrequencyLab
from ..sdr_interface import SDRInterface
from ..signal_processing import SignalProcessor
from ..signal_generator import SignalGenerator

class TestSignalGeneration(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 1e6
        self.duration = 0.5
        self.generator = SignalGenerator(self.sample_rate, self.duration)

    def test_tone_generation(self):
        freq = 10e3
        signal = self.generator.generate_tone(freq)
        self.assertEqual(len(signal), int(self.sample_rate * self.duration))
        # Verify the frequency content is where we expect it
        fft_signal = np.fft.fft(signal)
        fft_freqs = np.fft.fftfreq(len(signal), 1/self.sample_rate)
        peak_freq = fft_freqs[np.argmax(np.abs(fft_signal))]
        self.assertAlmostEqual(peak_freq, freq, delta=2)

    def test_noise_generation(self):
        noise = self.generator.generate_noise()
        self.assertEqual(len(noise), int(self.sample_rate * self.duration))
        # Check if it's complex
        self.assertTrue(np.iscomplexobj(noise))


class TestFrequencyLab(unittest.TestCase):

    def setUp(self):
        # Use the dummy SDR for testing
        sdr_config = {'driver': 'dummy'}
        self.lab = FrequencyLab(sdr_config)
        self.processor = SignalProcessor()

    def test_sdr_connection(self):
        # Test if the SDR interface connects without errors
        self.assertIsNotNone(self.lab.sdr.sdr_device)

    def test_capture_and_analyze(self):
        # Test the main capture and analysis function
        center_freq = 100e6
        bandwidth = 1e6
        sample_rate = 2e6
        
        freqs, psd = self.lab.capture_and_analyze(center_freq, bandwidth, sample_rate)
        
        self.assertIsNotNone(freqs)
        self.assertIsNotNone(psd)
        self.assertEqual(len(freqs), len(psd))
        
    def test_wifi_beacon_scan(self):
        # Test the Wi-Fi beacon scanning
        beacons = self.lab.scan_wifi_beacons()
        self.assertIsNotNone(beacons)
        self.assertIsInstance(beacons, list)
        # Check if dummy data is returned as expected
        if beacons:
            self.assertIn('ssid', beacons[0])
            self.assertIn('bssid', beacons[0])

    def test_signal_processing_psd(self):
        # Test the PSD calculation
        sample_rate = 1e6
        num_samples = 2048
        # Simple sine wave for testing
        t = np.arange(num_samples) / sample_rate
        samples = 1.0 * np.exp(2j * np.pi * 100e3 * t) # 100 kHz tone
        
        freqs, psd = self.processor.calculate_psd(samples, sample_rate, nfft=1024)
        
        self.assertIsNotNone(freqs)
        self.assertIsNotNone(psd)
        
        # Check that the peak is around 100 kHz
        peak_freq_index = np.argmax(psd)
        peak_freq = freqs[peak_freq_index]
        self.assertAlmostEqual(peak_freq, 100e3, delta=sample_rate/1024)

    def test_transmit(self):
        # Test the transmission functionality (dummy test)
        sample_rate_tx = 2e6
        t = np.arange(0, 0.1, 1/sample_rate_tx)
        tone = 0.5 * np.exp(2j * np.pi * 1e3 * t)
        
        # This will just print to console in dummy mode, no real assertion to make
        try:
            self.lab.transmit_signal(tone, 433e6, sample_rate_tx)
            # If it runs without error, it's a pass for the dummy interface
            ran_successfully = True
        except Exception as e:
            print(f"Transmission test failed: {e}")
            ran_successfully = False
            
        self.assertTrue(ran_successfully)


if __name__ == '__main__':
    unittest.main()
