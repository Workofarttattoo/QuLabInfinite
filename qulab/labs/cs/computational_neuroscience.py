"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
Computational Neuroscience Lab - Neural dynamics and brain modeling
"""

import numpy as np
from typing import Dict, List, Tuple

class ComputationalNeuroscienceLab:
    """Simulate neural dynamics and brain circuits"""
    
    def __init__(self):
        self.neuron_types = ['pyramidal', 'interneuron', 'granule', 'purkinje']
        self.brain_regions = ['cortex', 'hippocampus', 'cerebellum', 'thalamus']
        
    def simulate_spiking_neuron(self, input_current: float, duration_ms: float = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Leaky integrate-and-fire neuron model"""
        dt = 0.1  # ms
        time = np.arange(0, duration_ms, dt)
        
        # Neuron parameters
        tau_m = 10.0  # membrane time constant
        v_rest = -70  # mV
        v_threshold = -50  # mV
        v_reset = -70  # mV
        
        voltage = np.zeros(len(time))
        voltage[0] = v_rest
        spikes = []
        
        for i in range(1, len(time)):
            dv = (-(voltage[i-1] - v_rest) + input_current) / tau_m
            voltage[i] = voltage[i-1] + dv * dt
            
            if voltage[i] >= v_threshold:
                spikes.append(time[i])
                voltage[i] = v_reset
                
        return time, voltage, spikes
        
    def analyze_eeg_rhythm(self, signal: np.ndarray, sampling_rate: float = 250) -> Dict:
        """Analyze EEG frequency bands"""
        # FFT for frequency analysis
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sampling_rate)
        power = np.abs(fft)**2
        
        # Frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        band_power = {}
        for band, (low, high) in bands.items():
            idx = (freqs >= low) & (freqs < high)
            band_power[band] = power[idx].sum()
            
        # Normalize
        total_power = sum(band_power.values())
        for band in band_power:
            band_power[band] /= total_power
            
        return band_power
        
    def simulate_neural_network(self, n_neurons: int = 100, simulation_time: float = 1000) -> np.ndarray:
        """Simulate recurrent neural network dynamics"""
        # Random connectivity
        W = np.random.randn(n_neurons, n_neurons) * 0.1 / np.sqrt(n_neurons)
        
        # Dale's principle: 80% excitatory, 20% inhibitory
        inhibitory = np.random.random(n_neurons) < 0.2
        W[:, inhibitory] = -np.abs(W[:, inhibitory])
        
        # Simulate
        dt = 0.1
        tau = 10.0
        activity = np.random.randn(n_neurons) * 0.1
        trajectory = []
        
        for _ in range(int(simulation_time / dt)):
            input_current = W @ activity + np.random.randn(n_neurons) * 0.01
            activity += (-activity + np.tanh(input_current)) * dt / tau
            trajectory.append(activity.copy())
            
        return np.array(trajectory)