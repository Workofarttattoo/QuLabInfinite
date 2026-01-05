"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

frequency_lab - Part of Frequency Lab
"""


import numpy as np
from .sdr_interface import SDRInterface
from .signal_processing import SignalProcessor
from .wifi_analyzer import WifiAnalyzer
from .signal_generator import SignalGenerator

class FrequencyLab:
    def __init__(self, sdr_config):
        self.sdr = SDRInterface(sdr_config)
        self.processor = SignalProcessor()
        # The WifiAnalyzer might need to be initialized on-demand with correct params
        self.wifi_analyzer = None

    def capture_and_analyze(self, center_freq, bandwidth, sample_rate):
        """
        Capture signals from the SDR and perform frequency analysis.
        """
        print(f"Capturing at {center_freq / 1e6} MHz...")
        samples = self.sdr.capture_data(center_freq, bandwidth, sample_rate)
        
        if samples is not None:
            print("Processing signals...")
            freqs, psd = self.processor.calculate_psd(samples, sample_rate)
            return freqs, psd
        else:
            print("Failed to capture data.")
            return None, None

    def scan_wifi_beacons(self):
        """
        Scan for Wi-Fi beacons in the 2.4 GHz band.
        This is a simplified demonstration. A real scan would hop channels.
        """
        # We'll scan a common Wi-Fi channel, e.g., Channel 6 @ 2.437 GHz
        center_freq = 2.437e9
        sample_rate = 20e6 # Need a wide bandwidth for Wi-Fi
        bandwidth = 20e6

        print(f"Scanning for Wi-Fi beacons around {center_freq / 1e9} GHz...")
        samples = self.sdr.capture_data(center_freq, bandwidth, sample_rate, num_samples=1024*100)

        if samples is not None:
            if self.wifi_analyzer is None:
                self.wifi_analyzer = WifiAnalyzer(sample_rate, center_freq)
            
            beacons = self.wifi_analyzer.find_wifi_beacons(samples)
            return beacons
        else:
            print("Failed to capture Wi-Fi data.")
            return []

    def deauth_wifi_target(self, bssid, client):
        """
        Transmit deauthentication packets to a target client on a network.
        WARNING: This is for educational purposes only. Unauthorized deauthentication
        attacks are illegal.
        """
        print("\n*** WARNING: Deauthentication Attack Simulation ***")
        print("This function demonstrates crafting deauth packets.")
        print("TRANSMITTING THESE PACKETS ON A NETWORK YOU DO NOT OWN IS ILLEGAL.")
        
        if self.wifi_analyzer is None:
            # Need some default values if not initialized via scanning
            self.wifi_analyzer = WifiAnalyzer(sample_rate=20e6, center_freq=2.412e9)
            
        deauth_packet = self.wifi_analyzer.craft_deauth_packet(bssid, client)
        
        # This is the dangerous part. To actually transmit this, you would need
        # to convert the scapy packet to IQ samples and send it via the SDR.
        # This is a highly complex process involving modulating the signal.
        # For safety, we will NOT implement the transmission of these packets.
        
        print(f"Crafted deauth packet targeting Client: {client} on BSSID: {bssid}")
        print("Transmission of attack packets is NOT implemented for safety and legal reasons.")
        # deauth_iq_samples = self.modulate_packet(deauth_packet)
        # self.transmit_signal(deauth_iq_samples, center_freq, sample_rate)
        print("*** END WARNING ***\n")


    def generate_and_transmit(self, signal_params):
        """
        Generate a signal based on params and transmit it.
        """
        print("\n--- Signal Generation and Transmission ---")
        sr = signal_params.get('sample_rate')
        duration = signal_params.get('duration')
        center_freq = signal_params.get('center_freq')
        sig_type = signal_params.get('type')

        generator = SignalGenerator(sr, duration)
        signal_to_tx = None

        if sig_type == 'tone':
            freq = signal_params.get('tone_freq', 1e3)
            signal_to_tx = generator.generate_tone(freq)
        elif sig_type == 'noise':
            signal_to_tx = generator.generate_noise()
        elif sig_type == 'chirp':
            start_freq = signal_params.get('start_freq', -sr/4)
            end_freq = signal_params.get('end_freq', sr/4)
            signal_to_tx = generator.generate_chirp(start_freq, end_freq)
        else:
            print(f"Error: Unknown signal type '{sig_type}'")
            return

        if signal_to_tx is not None:
            self.transmit_signal(signal_to_tx, center_freq, sr)
        print("----------------------------------------\n")


    def transmit_signal(self, signal, center_freq, sample_rate):
        """
        Transmit a signal using the SDR.
        """
        print(f"Transmitting at {center_freq / 1e6} MHz...")
        self.sdr.transmit_data(signal, center_freq, sample_rate)
        print("Transmission complete.")

    def run_sweep(self, start_freq, end_freq, step_freq, bandwidth, sample_rate):
        """
        Sweep a frequency range and analyze the spectrum.
        """
        all_freqs = []
        all_psd = []

        for freq in np.arange(start_freq, end_freq, step_freq):
            freqs, psd = self.capture_and_analyze(freq, bandwidth, sample_rate)
            if freqs is not None:
                all_freqs.append(freqs)
                all_psd.append(psd)
        
        return all_freqs, all_psd

if __name__ == '__main__':
    # Example Usage
    sdr_config = {
        'driver': 'uhd',  # Example for USRP Hardware Driver
        'serial': ''      # Serial number of the SDR device
    }
    
    lab = FrequencyLab(sdr_config)
    
    # Capture and analyze a specific frequency
    center_frequency = 101.1e6  # FM Radio Station
    bw = 2e6
    sr = 2e6
    freqs, psd = lab.capture_and_analyze(center_frequency, bw, sr)
    
    if freqs is not None:
        # Here you would typically plot the results
        print(f"Captured {len(freqs)} frequency points.")

    # Example of transmitting a simple tone
    # DANGEROUS: Transmitting without proper knowledge and licensing is illegal.
    # This is for illustrative purposes only.
    #
    # sample_rate_tx = 2e6
    # carrier_freq = 1e3 # 1 kHz tone
    # t = np.arange(0, 1, 1/sample_rate_tx)
    # tone = 0.5 * np.exp(2j * np.pi * carrier_freq * t)
    # lab.transmit_signal(tone, 433.92e6, sample_rate_tx)

    # Example of scanning for Wi-Fi
    wifi_beacons = lab.scan_wifi_beacons()
    if wifi_beacons:
        print("\n--- Wi-Fi Beacons Found ---")
        for beacon in wifi_beacons:
            print(f"SSID: {beacon['ssid']:<20} BSSID: {beacon['bssid']:<20} Channel: {beacon['channel']:<5} Signal: {beacon['signal']} dBm")
        print("---------------------------\n")
