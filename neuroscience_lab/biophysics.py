"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

NEUROSCIENCE LAB - Production-Ready Implementation
Free gift to the scientific community from QuLabInfinite.

This module implements comprehensive computational neuroscience models including:
- Hodgkin-Huxley biophysical neuron model
- Integrate-and-fire models (LIF, AdEx, Izhikevich)
- Synaptic plasticity (STDP, BCM, Hebbian)
- Network connectivity patterns
- EEG/LFP field potential simulation
- Neurotransmitter dynamics and receptor kinetics
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from scipy import signal, stats
from scipy.integrate import odeint, solve_ivp
from scipy.sparse import csr_matrix, lil_matrix
import warnings

# Physical constants
FARADAY = 96485.33  # C/mol
GAS_CONSTANT = 8.314  # J/(mol*K)
TEMPERATURE = 310.15  # K (37°C)

@dataclass
class IonChannel:
    """Ion channel with voltage-dependent gating dynamics."""
    conductance: float  # mS/cm^2
    reversal_potential: float  # mV
    activation_gates: int = 1
    inactivation_gates: int = 0

    def alpha_m(self, V: float) -> float:
        """Activation gate opening rate."""
        if abs(V + 40) < 0.01:
            return 1.0
        return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

    def beta_m(self, V: float) -> float:
        """Activation gate closing rate."""
        return 4.0 * np.exp(-(V + 65) / 18)

    def alpha_h(self, V: float) -> float:
        """Inactivation gate opening rate."""
        return 0.07 * np.exp(-(V + 65) / 20)

    def beta_h(self, V: float) -> float:
        """Inactivation gate closing rate."""
        return 1.0 / (1 + np.exp(-(V + 35) / 10))

    def alpha_n(self, V: float) -> float:
        """K+ activation gate opening rate."""
        if abs(V + 55) < 0.01:
            return 0.1
        return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))

    def beta_n(self, V: float) -> float:
        """K+ activation gate closing rate."""
        return 0.125 * np.exp(-(V + 65) / 80)

class HodgkinHuxleyNeuron:
    """Complete Hodgkin-Huxley biophysical neuron model."""

    def __init__(self):
        self.C_m = 1.0  # membrane capacitance (uF/cm^2)
        self.g_Na = 120.0  # sodium conductance (mS/cm^2)
        self.g_K = 36.0  # potassium conductance
        self.g_L = 0.3  # leak conductance
        self.E_Na = 50.0  # sodium reversal potential (mV)
        self.E_K = -77.0  # potassium reversal
        self.E_L = -54.4  # leak reversal

        # State variables
        self.V = -65.0  # membrane potential
        self.m = 0.05  # Na activation
        self.h = 0.6  # Na inactivation
        self.n = 0.32  # K activation

    def derivatives(self, state: np.ndarray, t: float, I_ext: float) -> np.ndarray:
        """Calculate derivatives for ODE integration."""
        V, m, h, n = state

        # Voltage-dependent rate constants
        alpha_m = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10)) if abs(V + 40) > 0.01 else 1.0
        beta_m = 4.0 * np.exp(-(V + 65) / 18)
        alpha_h = 0.07 * np.exp(-(V + 65) / 20)
        beta_h = 1.0 / (1 + np.exp(-(V + 35) / 10))
        alpha_n = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10)) if abs(V + 55) > 0.01 else 0.1
        beta_n = 0.125 * np.exp(-(V + 65) / 80)

        # Ionic currents
        I_Na = self.g_Na * m**3 * h * (V - self.E_Na)
        I_K = self.g_K * n**4 * (V - self.E_K)
        I_L = self.g_L * (V - self.E_L)

        # Differential equations
        dVdt = (I_ext - I_Na - I_K - I_L) / self.C_m
        dmdt = alpha_m * (1 - m) - beta_m * m
        dhdt = alpha_h * (1 - h) - beta_h * h
        dndt = alpha_n * (1 - n) - beta_n * n

        return np.array([dVdt, dmdt, dhdt, dndt])

    def simulate(self, t_span: Tuple[float, float], I_ext: Callable, dt: float = 0.01) -> Dict:
        """Simulate neuron dynamics."""
        t = np.arange(t_span[0], t_span[1], dt)
        state0 = [self.V, self.m, self.h, self.n]

        # Integrate with stimulus
        solution = np.zeros((len(t), 4))
        solution[0] = state0

        for i in range(1, len(t)):
            I = I_ext(t[i]) if callable(I_ext) else I_ext
            k1 = self.derivatives(solution[i-1], t[i-1], I)
            k2 = self.derivatives(solution[i-1] + 0.5*dt*k1, t[i-1] + 0.5*dt, I)
            k3 = self.derivatives(solution[i-1] + 0.5*dt*k2, t[i-1] + 0.5*dt, I)
            k4 = self.derivatives(solution[i-1] + dt*k3, t[i-1] + dt, I)
            solution[i] = solution[i-1] + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        return {
            'time': t,
            'V': solution[:, 0],
            'm': solution[:, 1],
            'h': solution[:, 2],
            'n': solution[:, 3],
            'spikes': self._detect_spikes(solution[:, 0])
        }

    def _detect_spikes(self, V: np.ndarray, threshold: float = 0) -> np.ndarray:
        """Detect action potentials."""
        crossings = np.where(np.diff(V > threshold))[0]
        return crossings[::2] if len(crossings) > 0 else np.array([])

class LeakyIntegrateFireNeuron:
    """Leaky Integrate-and-Fire (LIF) neuron model."""

    def __init__(self, tau_m: float = 20.0, V_rest: float = -65.0,
                 V_thresh: float = -50.0, V_reset: float = -65.0,
                 R_m: float = 10.0, tau_ref: float = 2.0):
        self.tau_m = tau_m  # membrane time constant (ms)
        self.V_rest = V_rest  # resting potential (mV)
        self.V_thresh = V_thresh  # spike threshold
        self.V_reset = V_reset  # reset potential
        self.R_m = R_m  # membrane resistance (MOhm)
        self.tau_ref = tau_ref  # refractory period (ms)
        self.V = V_rest
        self.t_last_spike = -np.inf

    def update(self, I_ext: float, dt: float) -> bool:
        """Update membrane potential and check for spike."""
        t_since_spike = dt - self.t_last_spike

        if t_since_spike < self.tau_ref:
            return False  # In refractory period

        # Exponential integration
        self.V += dt * (-(self.V - self.V_rest) + self.R_m * I_ext) / self.tau_m

        if self.V >= self.V_thresh:
            self.V = self.V_reset
            self.t_last_spike = dt
            return True
        return False

    def simulate(self, t_span: Tuple[float, float], I_ext: Callable, dt: float = 0.1) -> Dict:
        """Simulate LIF dynamics."""
        t = np.arange(t_span[0], t_span[1], dt)
        V_trace = np.zeros(len(t))
        spikes = []

        self.V = self.V_rest
        for i, time in enumerate(t):
            I = I_ext(time) if callable(I_ext) else I_ext
            spiked = self.update(I, time)
            V_trace[i] = self.V
            if spiked:
                spikes.append(i)
                V_trace[i] = 40  # Visual spike marker

        return {'time': t, 'V': V_trace, 'spikes': np.array(spikes)}

class AdaptiveExponentialNeuron:
    """Adaptive Exponential Integrate-and-Fire (AdEx) model."""

    def __init__(self, C: float = 200.0, g_L: float = 10.0, E_L: float = -70.0,
                 V_T: float = -50.0, Delta_T: float = 2.0, a: float = 2.0,
                 tau_w: float = 30.0, b: float = 60.0, V_reset: float = -58.0):
        self.C = C  # capacitance (pF)
        self.g_L = g_L  # leak conductance (nS)
        self.E_L = E_L  # leak reversal (mV)
        self.V_T = V_T  # threshold slope factor
        self.Delta_T = Delta_T  # slope factor
        self.a = a  # subthreshold adaptation
        self.tau_w = tau_w  # adaptation time constant
        self.b = b  # spike-triggered adaptation
        self.V_reset = V_reset
        self.V_peak = 20.0

        self.V = E_L
        self.w = 0  # adaptation current

    def derivatives(self, state: np.ndarray, I_ext: float) -> np.ndarray:
        """Calculate derivatives."""
        V, w = state

        # Exponential spike mechanism
        exp_term = self.Delta_T * np.exp((V - self.V_T) / self.Delta_T)

        dVdt = (-self.g_L * (V - self.E_L) + self.g_L * exp_term - w + I_ext) / self.C
        dwdt = (self.a * (V - self.E_L) - w) / self.tau_w

        return np.array([dVdt, dwdt])

    def simulate(self, t_span: Tuple[float, float], I_ext: Callable, dt: float = 0.05) -> Dict:
        """Simulate AdEx dynamics with spike detection."""
        t = np.arange(t_span[0], t_span[1], dt)
        solution = np.zeros((len(t), 2))
        solution[0] = [self.V, self.w]
        spikes = []

        for i in range(1, len(t)):
            I = I_ext(t[i]) if callable(I_ext) else I_ext

            # RK4 integration
            k1 = self.derivatives(solution[i-1], I)
            k2 = self.derivatives(solution[i-1] + 0.5*dt*k1, I)
            k3 = self.derivatives(solution[i-1] + 0.5*dt*k2, I)
            k4 = self.derivatives(solution[i-1] + dt*k3, I)
            solution[i] = solution[i-1] + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

            # Spike detection and reset
            if solution[i, 0] > self.V_peak:
                spikes.append(i)
                solution[i, 0] = self.V_reset
                solution[i, 1] += self.b

        return {'time': t, 'V': solution[:, 0], 'w': solution[:, 1], 'spikes': np.array(spikes)}

class IzhikevichNeuron:
    """Izhikevich neuron model - simple yet biologically realistic."""

    def __init__(self, neuron_type: str = 'RS'):
        """Initialize with specific neuron type parameters."""
        params = {
            'RS': (0.02, 0.2, -65, 8),  # Regular spiking
            'IB': (0.02, 0.2, -55, 4),  # Intrinsically bursting
            'CH': (0.02, 0.2, -50, 2),  # Chattering
            'FS': (0.1, 0.2, -65, 2),   # Fast spiking
            'LTS': (0.02, 0.25, -65, 2)  # Low-threshold spiking
        }

        self.a, self.b, self.c, self.d = params.get(neuron_type, params['RS'])
        self.v = -70  # membrane potential
        self.u = self.b * self.v  # recovery variable

    def update(self, I: float, dt: float = 0.5) -> bool:
        """Update state and return spike status."""
        v_prev = self.v
        self.v += dt * (0.04 * v_prev**2 + 5 * v_prev + 140 - self.u + I)
        self.u += dt * self.a * (self.b * v_prev - self.u)

        if self.v >= 30:
            self.v = self.c
            self.u += self.d
            return True
        return False

class SynapticPlasticity:
    """Synaptic plasticity rules including STDP and BCM."""

    @staticmethod
    def stdp(dt: float, A_plus: float = 0.01, A_minus: float = 0.01,
             tau_plus: float = 20.0, tau_minus: float = 20.0) -> float:
        """Spike-Timing Dependent Plasticity."""
        if dt > 0:
            return A_plus * np.exp(-dt / tau_plus)
        else:
            return -A_minus * np.exp(dt / tau_minus)

    @staticmethod
    def bcm(post_rate: float, pre_rate: float, theta: float = 15.0,
            tau_theta: float = 1000.0) -> Tuple[float, float]:
        """Bienenstock-Cooper-Munro plasticity rule."""
        dw = pre_rate * post_rate * (post_rate - theta)
        dtheta = (post_rate**2 - theta) / tau_theta
        return dw, dtheta

    @staticmethod
    def hebbian(pre: float, post: float, learning_rate: float = 0.001) -> float:
        """Basic Hebbian learning: neurons that fire together wire together."""
        return learning_rate * pre * post

    @staticmethod
    def oja(pre: float, post: float, w: float, alpha: float = 0.001) -> float:
        """Oja's rule - Hebbian with normalization."""
        return alpha * (pre * post - post**2 * w)

class NetworkConnectivity:
    """Generate various network connectivity patterns."""

    @staticmethod
    def random_network(N: int, p: float, weight_mean: float = 1.0,
                       weight_std: float = 0.2) -> csr_matrix:
        """Erdős-Rényi random network."""
        W = np.random.rand(N, N) < p
        W = W.astype(float) * np.random.normal(weight_mean, weight_std, (N, N))
        np.fill_diagonal(W, 0)  # No self-connections
        return csr_matrix(W)

    @staticmethod
    def small_world(N: int, k: int, p: float, weight: float = 1.0) -> csr_matrix:
        """Watts-Strogatz small-world network."""
        W = lil_matrix((N, N))

        # Regular ring lattice
        for i in range(N):
            for j in range(1, k//2 + 1):
                W[i, (i + j) % N] = weight
                W[i, (i - j) % N] = weight

        # Rewire edges with probability p
        for i in range(N):
            for j in range(i + 1, min(i + k//2 + 1, N)):
                if np.random.rand() < p:
                    new_target = np.random.choice([x for x in range(N) if x != i])
                    W[i, j % N] = 0
                    W[i, new_target] = weight

        return W.tocsr()

    @staticmethod
    def scale_free(N: int, m: int, weight: float = 1.0) -> csr_matrix:
        """Barabási-Albert scale-free network."""
        W = lil_matrix((N, N))
        degrees = np.zeros(N)

        # Start with m fully connected nodes
        for i in range(m):
            for j in range(i + 1, m):
                W[i, j] = W[j, i] = weight
                degrees[i] += 1
                degrees[j] += 1

        # Add remaining nodes with preferential attachment
        for i in range(m, N):
            if degrees[:i].sum() > 0:
                probs = degrees[:i] / degrees[:i].sum()
                targets = np.random.choice(i, min(m, i), replace=False, p=probs)
                for t in targets:
                    W[i, t] = W[t, i] = weight
                    degrees[i] += 1
                    degrees[t] += 1

        return W.tocsr()

class NeurotransmitterDynamics:
    """Model neurotransmitter release and receptor dynamics."""

    def __init__(self, transmitter_type: str = 'glutamate'):
        self.type = transmitter_type
        self.params = {
            'glutamate': {'tau_rise': 0.5, 'tau_decay': 3.0, 'g_max': 1.0},
            'gaba': {'tau_rise': 1.0, 'tau_decay': 7.0, 'g_max': 1.0},
            'dopamine': {'tau_rise': 2.0, 'tau_decay': 20.0, 'g_max': 0.5},
            'serotonin': {'tau_rise': 5.0, 'tau_decay': 100.0, 'g_max': 0.3}
        }
        self.p = self.params.get(transmitter_type, self.params['glutamate'])

    def synaptic_current(self, t: np.ndarray, t_spike: float) -> np.ndarray:
        """Calculate postsynaptic current following presynaptic spike."""
        t_post = t - t_spike
        mask = t_post > 0

        # Difference of exponentials
        tau_r, tau_d = self.p['tau_rise'], self.p['tau_decay']
        g_max = self.p['g_max']

        current = np.zeros_like(t)
        norm = (tau_d / tau_r) ** (tau_r / (tau_d - tau_r))
        current[mask] = g_max * norm * (np.exp(-t_post[mask] / tau_d) -
                                        np.exp(-t_post[mask] / tau_r))
        return current

    def receptor_dynamics(self, concentration: float, state: Dict) -> Dict:
        """Model receptor state transitions."""
        # Simplified kinetic model
        k_on = 1.0  # binding rate
        k_off = 0.1  # unbinding rate

        bound_fraction = concentration * k_on / (concentration * k_on + k_off)

        return {
            'bound': bound_fraction,
            'conductance': bound_fraction * self.p['g_max'],
            'desensitization': 1 - np.exp(-concentration / 10)  # Simple desensitization
        }

class EEGSimulator:
    """Simulate EEG/LFP signals from neural populations."""

    def __init__(self, n_sources: int = 10, sampling_rate: float = 1000.0):
        self.n_sources = n_sources
        self.fs = sampling_rate
        self.dipole_moments = np.random.randn(n_sources, 3) * 1e-9  # nAm
        self.source_positions = np.random.randn(n_sources, 3) * 0.05  # meters

    def forward_model(self, dipoles: np.ndarray, electrode_pos: np.ndarray) -> np.ndarray:
        """Calculate scalp potentials from dipole sources (simplified)."""
        n_electrodes = electrode_pos.shape[0]
        n_times = dipoles.shape[0]
        potentials = np.zeros((n_times, n_electrodes))

        sigma = 0.33  # conductivity (S/m)

        for e in range(n_electrodes):
            for s in range(self.n_sources):
                r = np.linalg.norm(electrode_pos[e] - self.source_positions[s])
                if r > 0:
                    # Simplified dipole in infinite homogeneous conductor
                    lead_field = 1 / (4 * np.pi * sigma * r**2)
                    potentials[:, e] += lead_field * dipoles[:, s]

        return potentials

    def generate_oscillation(self, duration: float, freq: float,
                           phase_noise: float = 0.1) -> np.ndarray:
        """Generate oscillatory activity."""
        t = np.arange(0, duration, 1/self.fs)
        phase = 2 * np.pi * freq * t + phase_noise * np.cumsum(np.random.randn(len(t)))
        return np.sin(phase)

    def simulate_eeg(self, duration: float, bands: Dict[str, Tuple[float, float]]) -> Dict:
        """Simulate multi-band EEG signals."""
        t = np.arange(0, duration, 1/self.fs)
        n_samples = len(t)

        # Generate source activity for different frequency bands
        sources = np.zeros((n_samples, self.n_sources))

        band_weights = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }

        for i in range(self.n_sources):
            # Each source has different band contributions
            for band, (f_min, f_max) in band_weights.items():
                if band in bands:
                    weight = bands[band]
                    freq = np.random.uniform(f_min, f_max)
                    sources[:, i] += weight * self.generate_oscillation(duration, freq)

        # Add 1/f noise
        sources += self._pink_noise(n_samples, self.n_sources)

        # Calculate scalp potentials (10-20 system positions)
        electrode_positions = self._get_1020_positions()
        eeg = self.forward_model(sources, electrode_positions)

        return {
            'time': t,
            'eeg': eeg,
            'sources': sources,
            'sampling_rate': self.fs,
            'electrode_names': ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
        }

    def _pink_noise(self, n_samples: int, n_channels: int) -> np.ndarray:
        """Generate 1/f pink noise."""
        white = np.random.randn(n_samples, n_channels)

        # Apply 1/f filter in frequency domain
        fft = np.fft.rfft(white, axis=0)
        freqs = np.fft.rfftfreq(n_samples)
        freqs[0] = 1  # Avoid division by zero

        # 1/f amplitude scaling
        fft = fft / np.sqrt(freqs[:, np.newaxis])
        pink = np.fft.irfft(fft, n=n_samples, axis=0)

        return pink * 0.1  # Scale amplitude

    def _get_1020_positions(self) -> np.ndarray:
        """Get approximate 10-20 electrode positions."""
        # Simplified spherical positions (in meters)
        positions = np.array([
            [-0.03, 0.08, 0.05],   # Fp1
            [0.03, 0.08, 0.05],    # Fp2
            [-0.05, 0.04, 0.08],   # F3
            [0.05, 0.04, 0.08],    # F4
            [-0.07, 0, 0.09],      # C3
            [0.07, 0, 0.09],       # C4
            [-0.05, -0.04, 0.08],  # P3
            [0.05, -0.04, 0.08],   # P4
            [-0.03, -0.08, 0.05],  # O1
            [0.03, -0.08, 0.05]    # O2
        ])
        return positions

class NeuralNetworkSimulator:
    """Simulate large-scale neural networks with biological detail."""

    def __init__(self, n_exc: int = 800, n_inh: int = 200):
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_total = n_exc + n_inh

        # Initialize neurons
        self.neurons = []
        for i in range(n_exc):
            self.neurons.append(IzhikevichNeuron('RS'))  # Regular spiking excitatory
        for i in range(n_inh):
            self.neurons.append(IzhikevichNeuron('FS'))  # Fast spiking inhibitory

        # Create connectivity
        self.W = self._create_connectivity()

        # Synaptic parameters
        self.tau_ampa = 5.0  # ms
        self.tau_gaba = 10.0  # ms
        self.synaptic_delay = 1.0  # ms

    def _create_connectivity(self) -> np.ndarray:
        """Create biologically realistic connectivity."""
        W = np.zeros((self.n_total, self.n_total))

        # Connection probabilities
        p_ee = 0.1  # exc to exc
        p_ei = 0.2  # exc to inh
        p_ie = 0.4  # inh to exc
        p_ii = 0.3  # inh to inh

        # Excitatory connections (positive weights)
        W[:self.n_exc, :self.n_exc] = (np.random.rand(self.n_exc, self.n_exc) < p_ee) * \
                                       np.random.gamma(2, 0.5, (self.n_exc, self.n_exc))
        W[:self.n_exc, self.n_exc:] = (np.random.rand(self.n_exc, self.n_inh) < p_ei) * \
                                       np.random.gamma(2, 0.5, (self.n_exc, self.n_inh))

        # Inhibitory connections (negative weights)
        W[self.n_exc:, :self.n_exc] = -1.0 * (np.random.rand(self.n_inh, self.n_exc) < p_ie) * \
                                       np.random.gamma(2, 2, (self.n_inh, self.n_exc))
        W[self.n_exc:, self.n_exc:] = -1.0 * (np.random.rand(self.n_inh, self.n_inh) < p_ii) * \
                                       np.random.gamma(2, 2, (self.n_inh, self.n_inh))

        np.fill_diagonal(W, 0)  # No self-connections
        return W

    def simulate(self, duration: float, external_input: Optional[np.ndarray] = None,
                 dt: float = 0.5) -> Dict:
        """Run network simulation."""
        n_steps = int(duration / dt)

        # Initialize recording arrays
        spike_times = []
        spike_neurons = []
        voltage_trace = np.zeros((n_steps, self.n_total))

        # Synaptic currents
        I_syn = np.zeros(self.n_total)

        # External input
        if external_input is None:
            external_input = np.random.randn(n_steps, self.n_total) * 5 + 10

        # Main simulation loop
        for t in range(n_steps):
            # Update synaptic currents (exponential decay)
            I_syn *= np.exp(-dt / self.tau_ampa)

            # Update each neuron
            spikes = np.zeros(self.n_total, dtype=bool)
            for i, neuron in enumerate(self.neurons):
                I_total = I_syn[i] + external_input[t, i]
                spikes[i] = neuron.update(I_total, dt)
                voltage_trace[t, i] = neuron.v

            # Record spikes
            spike_indices = np.where(spikes)[0]
            for idx in spike_indices:
                spike_times.append(t * dt)
                spike_neurons.append(idx)

            # Update synaptic currents from spikes
            if len(spike_indices) > 0:
                I_syn += self.W[:, spike_indices].sum(axis=1)

        return {
            'spike_times': np.array(spike_times),
            'spike_neurons': np.array(spike_neurons),
            'voltage_trace': voltage_trace,
            'time': np.arange(n_steps) * dt,
            'firing_rates': self._calculate_firing_rates(spike_neurons, duration)
        }

    def _calculate_firing_rates(self, spike_neurons: np.ndarray, duration: float) -> np.ndarray:
        """Calculate firing rates for each neuron."""
        rates = np.zeros(self.n_total)
        for i in range(self.n_total):
            rates[i] = np.sum(spike_neurons == i) / duration * 1000  # Hz
        return rates

def demonstrate_neuroscience_lab():
    """Comprehensive demonstration of neuroscience capabilities."""
    print("=" * 80)
    print("NEUROSCIENCE LAB DEMONSTRATION")
    print("Copyright (c) 2025 Corporation of Light. All Rights Reserved.")
    print("=" * 80)

    # 1. Hodgkin-Huxley model
    print("\n1. HODGKIN-HUXLEY BIOPHYSICAL MODEL")
    print("-" * 40)
    hh = HodgkinHuxleyNeuron()

    # Step current injection
    I_step = lambda t: 10 if 20 < t < 80 else 0
    result = hh.simulate((0, 100), I_step, dt=0.01)

    print(f"Simulation complete: {len(result['time'])} time points")
    print(f"Action potentials detected: {len(result['spikes'])}")
    print(f"Average firing rate: {len(result['spikes']) / 0.1:.1f} Hz")

    # 2. Integrate-and-Fire models comparison
    print("\n2. INTEGRATE-AND-FIRE MODEL COMPARISON")
    print("-" * 40)

    lif = LeakyIntegrateFireNeuron()
    adex = AdaptiveExponentialNeuron()
    izhikevich = IzhikevichNeuron('RS')

    # Common input current
    I_test = lambda t: 15 * np.sin(2 * np.pi * 10 * t / 1000) + 20

    lif_result = lif.simulate((0, 200), I_test, dt=0.1)
    adex_result = adex.simulate((0, 200), I_test, dt=0.1)

    print(f"LIF spikes: {len(lif_result['spikes'])}")
    print(f"AdEx spikes: {len(adex_result['spikes'])}")
    print("Izhikevich types: RS, IB, CH, FS, LTS")

    # 3. Synaptic Plasticity
    print("\n3. SYNAPTIC PLASTICITY MECHANISMS")
    print("-" * 40)

    # STDP curve
    dt_values = np.linspace(-50, 50, 100)
    stdp_weights = [SynapticPlasticity.stdp(dt) for dt in dt_values]

    print("STDP: Spike-Timing Dependent Plasticity computed")
    print(f"  Maximum potentiation: {max(stdp_weights):.3f}")
    print(f"  Maximum depression: {min(stdp_weights):.3f}")

    # BCM rule
    post_rate = 20.0  # Hz
    pre_rate = 15.0
    dw, dtheta = SynapticPlasticity.bcm(post_rate, pre_rate)
    print(f"BCM: Weight change = {dw:.4f}, Threshold change = {dtheta:.4f}")

    # 4. Network Connectivity
    print("\n4. NETWORK CONNECTIVITY PATTERNS")
    print("-" * 40)

    N = 100
    random_net = NetworkConnectivity.random_network(N, p=0.1)
    small_world = NetworkConnectivity.small_world(N, k=10, p=0.3)
    scale_free = NetworkConnectivity.scale_free(N, m=3)

    print(f"Random network: {random_net.nnz} connections (density: {random_net.nnz/N**2:.3f})")
    print(f"Small-world: {small_world.nnz} connections")
    print(f"Scale-free: {scale_free.nnz} connections")

    # 5. EEG Simulation
    print("\n5. EEG/LFP FIELD POTENTIAL SIMULATION")
    print("-" * 40)

    eeg_sim = EEGSimulator(n_sources=10, sampling_rate=1000)

    # Simulate with different band powers
    bands = {'alpha': 1.0, 'beta': 0.5, 'gamma': 0.3}
    eeg_data = eeg_sim.simulate_eeg(duration=2.0, bands=bands)

    print(f"EEG channels: {len(eeg_data['electrode_names'])}")
    print(f"Electrodes: {', '.join(eeg_data['electrode_names'])}")
    print(f"Sampling rate: {eeg_data['sampling_rate']} Hz")
    print(f"Signal shape: {eeg_data['eeg'].shape}")

    # 6. Neurotransmitter Dynamics
    print("\n6. NEUROTRANSMITTER DYNAMICS")
    print("-" * 40)

    for nt in ['glutamate', 'gaba', 'dopamine']:
        dynamics = NeurotransmitterDynamics(nt)
        t = np.linspace(0, 50, 500)
        current = dynamics.synaptic_current(t, t_spike=10)
        peak_time = t[np.argmax(current)]
        print(f"{nt.upper()}: Peak at {peak_time:.1f} ms, tau_decay = {dynamics.p['tau_decay']} ms")

    # 7. Large-scale Network Simulation
    print("\n7. LARGE-SCALE NETWORK SIMULATION")
    print("-" * 40)

    network = NeuralNetworkSimulator(n_exc=800, n_inh=200)
    net_result = network.simulate(duration=100, dt=0.5)

    mean_rate_exc = net_result['firing_rates'][:800].mean()
    mean_rate_inh = net_result['firing_rates'][800:].mean()

    print(f"Network size: {network.n_total} neurons (E:{network.n_exc}, I:{network.n_inh})")
    print(f"Total spikes: {len(net_result['spike_times'])}")
    print(f"Mean firing rate (Exc): {mean_rate_exc:.1f} Hz")
    print(f"Mean firing rate (Inh): {mean_rate_inh:.1f} Hz")

    # Calculate synchrony
    if len(net_result['spike_times']) > 0:
        cv = np.std(net_result['spike_times']) / np.mean(net_result['spike_times'])
        print(f"Network synchrony (CV): {cv:.3f}")

    print("\n" + "=" * 80)
    print("Demonstration complete. Visit aios.is for more information.")
    print("=" * 80)

if __name__ == '__main__':
    demonstrate_neuroscience_lab()
