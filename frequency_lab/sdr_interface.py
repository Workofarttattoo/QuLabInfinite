
import numpy as np

class SDRInterface:
    def __init__(self, config):
        self.driver = config.get('driver', 'dummy')
        self.serial = config.get('serial', '')
        self.sdr_device = None
        self._connect()

    def _connect(self):
        """
        Connect to the SDR device.
        This is a placeholder. In a real implementation, you would use a library
        like SoapySDR, pyUHD, etc., to connect to the hardware.
        """
        print(f"Connecting to SDR with driver '{self.driver}' and serial '{self.serial}'...")
        if self.driver == 'dummy':
            print("Using dummy SDR driver.")
            self.sdr_device = self._create_dummy_sdr()
        else:
            try:
                # Example:
                # import SoapySDR
                # self.sdr_device = SoapySDR.Device(f"driver={self.driver},serial={self.serial}")
                print(f"Successfully connected to SDR (simulated).")
            except Exception as e:
                print(f"Error connecting to SDR: {e}")
                self.sdr_device = None

    def _create_dummy_sdr(self):
        """
        Create a dummy SDR object for testing without hardware.
        """
        class DummySDR:
            def capture(self, center_freq, bandwidth, sample_rate, num_samples):
                print(f"Dummy SDR: Simulating capture of {num_samples} at {center_freq/1e6} MHz")
                # Generate some noise with a sine wave
                t = np.arange(num_samples) / sample_rate
                # a carrier signal
                carrier = 0.5 * np.exp(2j * np.pi * (bandwidth / 4) * t)
                noise = 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
                return carrier + noise

            def transmit(self, signal, center_freq, sample_rate):
                print(f"Dummy SDR: Simulating transmission at {center_freq/1e6} MHz")
                # In a dummy, we don't need to do anything.
                pass
        
        return DummySDR()

    def capture_data(self, center_freq, bandwidth, sample_rate, num_samples=1024*16):
        """
        Capture data from the SDR.
        Returns a numpy array of complex samples.
        """
        if self.sdr_device:
            return self.sdr_device.capture(center_freq, bandwidth, sample_rate, num_samples)
        else:
            print("SDR not connected.")
            return None

    def transmit_data(self, signal, center_freq, sample_rate):
        """
        Transmit data using the SDR.
        'signal' should be a numpy array of complex samples.
        """
        if self.sdr_device:
            self.sdr_device.transmit(signal, center_freq, sample_rate)
        else:
            print("SDR not connected.")

    def __del__(self):
        """
        Clean up resources when the object is destroyed.
        """
        if self.sdr_device and self.driver != 'dummy':
            # In a real implementation, you would properly close the device connection.
            # Example: self.sdr_device.close()
            print("SDR connection closed.")
