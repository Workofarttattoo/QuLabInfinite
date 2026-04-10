
import argparse
from .frequency_lab import FrequencyLab
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Frequency Lab CLI")
    
    parser.add_argument('--mode', choices=['scan', 'tx', 'wifi', 'deauth'], default='scan',
                        help="Operating mode: 'scan', 'tx', 'wifi', or 'deauth'.")
    parser.add_argument('--freq', type=float, default=101.1e6,
                        help="Center frequency in Hz (e.g., 101.1e6 for 101.1 MHz).")
    parser.add_argument('--bw', type=float, default=2e6,
                        help="Bandwidth in Hz.")
    parser.add_argument('--sr', type=float, default=2e6,
                        help="Sample rate in Hz.")
    parser.add_argument('--driver', type=str, default='dummy',
                        help="SDR driver to use (e.g., 'uhd', 'soapy').")
    parser.add_argument('--serial', type=str, default='',
                        help="SDR device serial number.")
    # Arguments for deauth mode
    parser.add_argument('--bssid', type=str, help="Target BSSID for deauth attack.")
    parser.add_argument('--client', type=str, default='ff:ff:ff:ff:ff:ff', 
                        help="Target client MAC for deauth (default: broadcast).")

    # Arguments for tx mode
    parser.add_argument('--signal_type', type=str, default='tone',
                        help="Signal to transmit: 'tone', 'noise', 'chirp'.")
    parser.add_argument('--duration', type=float, default=1.0,
                        help="Duration of the transmitted signal in seconds.")
    parser.add_argument('--tone_freq', type=float, default=10e3,
                        help="Frequency of the tone to transmit (offset from center).")


    args = parser.parse_args()

    sdr_config = {
        'driver': args.driver,
        'serial': args.serial
    }
    
    lab = FrequencyLab(sdr_config)

    if args.mode == 'scan':
        print(f"Scanning at {args.freq / 1e6} MHz...")
        freqs, psd = lab.capture_and_analyze(args.freq, args.bw, args.sr)
        
        if freqs is not None and psd is not None:
            plt.figure()
            plt.plot(freqs / 1e6, psd)
            plt.title(f"Spectrum at {args.freq / 1e6} MHz")
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("PSD (dB/Hz)")
            plt.grid(True)
            plt.show()
    
    elif args.mode == 'tx':
        print("\n*** WARNING: Transmitting Signals ***")
        print("Transmitting radio signals without the proper licensing and knowledge is illegal.")
        print("Ensure you are operating in a safe, controlled environment (e.g., with a dummy load).")
        print("--------------------------------------\n")
        
        signal_params = {
            'type': args.signal_type,
            'sample_rate': args.sr,
            'duration': args.duration,
            'center_freq': args.freq,
            'tone_freq': args.tone_freq
            # Add other params for chirp etc. as needed
        }
        lab.generate_and_transmit(signal_params)
    
    elif args.mode == 'wifi':
        beacons = lab.scan_wifi_beacons()
        if beacons:
            print("\n--- Wi-Fi Beacons Found ---")
            # Sort by signal strength
            beacons.sort(key=lambda x: x.get('signal', -100), reverse=True)
            for beacon in beacons:
                ssid = beacon.get('ssid', '<hidden>')
                if not ssid:
                    ssid = '<hidden>'
                bssid = beacon.get('bssid', 'N/A')
                channel = beacon.get('channel', '?')
                signal = beacon.get('signal', 'N/A')
                print(f"SSID: {ssid:<20} BSSID: {bssid:<20} Chan: {str(channel):<4} Signal: {signal} dBm")
            print("---------------------------\n")
        else:
            print("No Wi-Fi beacons found.")

    elif args.mode == 'deauth':
        if not args.bssid:
            print("Error: --bssid is required for deauth mode.")
            return
        lab.deauth_wifi_target(args.bssid, args.client)


if __name__ == '__main__':
    main()
