"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

NEURAL NETWORKS LAB - Production-Ready Biological Neural Networks
Free gift to the scientific community from QuLabInfinite.

This module implements comprehensive biological neural network simulations:
- Spiking neural networks with biological dynamics
- Hebbian learning and synaptic plasticity
- Winner-take-all competitive networks
- Attractor networks and associative memory
- Neural oscillations and synchronization
- Population coding and decoding
- Cortical column simulations
- Reservoir computing with biological neurons
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from scipy import signal, stats, sparse
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, lil_matrix
import warnings
from enum import Enum

# Neurophysiological constants
TAU_AMPA = 5.0  # ms - AMPA receptor time constant
TAU_NMDA = 100.0  # ms - NMDA receptor time constant
TAU_GABA_A = 10.0  # ms - GABA-A receptor time constant
TAU_GABA_B = 200.0  # ms - GABA-B receptor time constant

class NeuronType(Enum):
    """Biological neuron types."""
    PYRAMIDAL = "pyramidal"
    INTERNEURON = "interneuron"
    STELLATE = "stellate"
    PURKINJE = "purkinje"
    GRANULE = "granule"
    BASKET = "basket"
    CHANDELIER = "chandelier"
    MARTINOTTI = "martinotti"

@dataclass
class BiologicalNeuron:
    """Biologically realistic neuron with multiple ion channels."""
    neuron_type: NeuronType
    position: np.ndarray  # 3D spatial position
    layer: int  # Cortical layer (1-6)

    # Membrane properties
    C_m: float = 1.0  # Membrane capacitance (μF/cm²)
    V_rest: float = -70.0  # Resting potential (mV)
    V_thresh: float = -55.0  # Spike threshold (mV)
    V_reset: float = -70.0  # Reset potential (mV)
    tau_ref: float = 2.0  # Refractory period (ms)

    # Ion channel conductances
    g_Na: float = 120.0  # Sodium conductance (mS/cm²)
    g_K: float = 36.0  # Potassium conductance
    g_L: float = 0.3  # Leak conductance
    g_Ca: float = 1.0  # Calcium conductance

    # Reversal potentials
    E_Na: float = 50.0  # Sodium reversal (mV)
    E_K: float = -77.0  # Potassium reversal
    E_L: float = -54.4  # Leak reversal
    E_Ca: float = 120.0  # Calcium reversal

    # State variables
    V: float = field(default=-70.0)  # Membrane potential
    u: float = field(default=0.0)  # Recovery variable
    spike_times: List[float] = field(default_factory=list)
    calcium: float = field(default=0.0)  # Intracellular calcium

    def __post_init__(self):
        """Initialize neuron-type specific parameters."""
        if self.neuron_type == NeuronType.PYRAMIDAL:
            self.tau_m = 20.0  # ms
            self.adaptation = 0.02
        elif self.neuron_type == NeuronType.INTERNEURON:
            self.tau_m = 10.0
            self.adaptation = 0.1
        elif self.neuron_type == NeuronType.PURKINJE:
            self.tau_m = 15.0
            self.adaptation = 0.05
            self.g_Ca = 2.0  # Higher calcium conductance
        else:
            self.tau_m = 15.0
            self.adaptation = 0.03

class SpikingNeuralNetwork:
    """Large-scale spiking neural network with biological detail."""

    def __init__(self, n_neurons: int = 1000, connectivity: str = 'random'):
        self.n_neurons = n_neurons
        self.connectivity_type = connectivity

        # Create neuron populations (80% excitatory, 20% inhibitory)
        self.n_exc = int(0.8 * n_neurons)
        self.n_inh = n_neurons - self.n_exc

        self.neurons = self._create_neurons()
        self.W = self._create_connectivity()

        # Synaptic dynamics
        self.synaptic_delays = np.random.uniform(0.5, 5.0, (n_neurons, n_neurons))
        self.synaptic_traces = np.zeros((n_neurons, 4))  # AMPA, NMDA, GABA-A, GABA-B

        # Plasticity parameters
        self.stdp_window = 50.0  # ms
        self.learning_rate = 0.001

    def _create_neurons(self) -> List[BiologicalNeuron]:
        """Create heterogeneous neuron population."""
        neurons = []

        # Excitatory neurons (pyramidal cells)
        for i in range(self.n_exc):
            position = np.random.randn(3) * 100  # μm
            layer = np.random.choice([2, 3, 4, 5, 6], p=[0.1, 0.2, 0.3, 0.3, 0.1])
            neurons.append(BiologicalNeuron(
                neuron_type=NeuronType.PYRAMIDAL,
                position=position,
                layer=layer
            ))

        # Inhibitory neurons (various interneuron types)
        inh_types = [NeuronType.INTERNEURON, NeuronType.BASKET,
                     NeuronType.CHANDELIER, NeuronType.MARTINOTTI]
        for i in range(self.n_inh):
            position = np.random.randn(3) * 100
            layer = np.random.choice([2, 3, 4, 5], p=[0.25, 0.25, 0.25, 0.25])
            neurons.append(BiologicalNeuron(
                neuron_type=np.random.choice(inh_types),
                position=position,
                layer=layer
            ))

        return neurons

    def _create_connectivity(self) -> np.ndarray:
        """Create biologically realistic connectivity matrix."""
        W = np.zeros((self.n_neurons, self.n_neurons))

        if self.connectivity_type == 'random':
            # Random connectivity with distance-dependent probability
            positions = np.array([n.position for n in self.neurons])
            distances = cdist(positions, positions)

            # Connection probability decreases with distance
            sigma = 50.0  # μm
            conn_prob = np.exp(-distances**2 / (2 * sigma**2))

            # Different connection probabilities for E-E, E-I, I-E, I-I
            p_ee, p_ei, p_ie, p_ii = 0.1, 0.2, 0.4, 0.3

            # Excitatory connections
            W[:self.n_exc, :self.n_exc] = (np.random.rand(self.n_exc, self.n_exc) <
                                           p_ee * conn_prob[:self.n_exc, :self.n_exc])
            W[:self.n_exc, self.n_exc:] = (np.random.rand(self.n_exc, self.n_inh) <
                                           p_ei * conn_prob[:self.n_exc, self.n_exc:])

            # Inhibitory connections (negative weights)
            W[self.n_exc:, :self.n_exc] = -(np.random.rand(self.n_inh, self.n_exc) <
                                            p_ie * conn_prob[self.n_exc:, :self.n_exc])
            W[self.n_exc:, self.n_exc:] = -(np.random.rand(self.n_inh, self.n_inh) <
                                            p_ii * conn_prob[self.n_exc:, self.n_exc:])

            # Scale weights
            W[:self.n_exc] *= np.random.gamma(2, 0.5, W[:self.n_exc].shape)
            W[self.n_exc:] *= np.random.gamma(2, 2, W[self.n_exc:].shape)

        elif self.connectivity_type == 'small_world':
            W = self._small_world_connectivity()

        elif self.connectivity_type == 'cortical_column':
            W = self._cortical_column_connectivity()

        np.fill_diagonal(W, 0)  # No self-connections
        return W

    def _small_world_connectivity(self) -> np.ndarray:
        """Create small-world network (Watts-Strogatz)."""
        W = lil_matrix((self.n_neurons, self.n_neurons))
        k = 10  # Average degree
        p = 0.3  # Rewiring probability

        # Regular ring lattice
        for i in range(self.n_neurons):
            for j in range(1, k//2 + 1):
                # Excitatory or inhibitory based on neuron type
                weight = 1.0 if i < self.n_exc else -1.0
                W[i, (i + j) % self.n_neurons] = weight * np.random.gamma(2, 0.5)
                W[i, (i - j) % self.n_neurons] = weight * np.random.gamma(2, 0.5)

        # Rewire with probability p
        W = W.toarray()
        for i in range(self.n_neurons):
            for j in range(i + 1, min(i + k//2 + 1, self.n_neurons)):
                if np.random.rand() < p:
                    new_target = np.random.choice(
                        [x for x in range(self.n_neurons) if x != i]
                    )
                    W[i, j % self.n_neurons] = 0
                    W[i, new_target] = np.random.gamma(2, 0.5)

        return W

    def _cortical_column_connectivity(self) -> np.ndarray:
        """Create cortical column connectivity with layer structure."""
        W = np.zeros((self.n_neurons, self.n_neurons))

        # Layer-specific connection rules
        layer_conn = {
            (2, 3): 0.3, (3, 3): 0.2, (3, 5): 0.4,
            (4, 3): 0.3, (4, 4): 0.1, (4, 5): 0.2,
            (5, 5): 0.2, (5, 6): 0.3,
            (6, 4): 0.2, (6, 6): 0.1
        }

        for i, pre in enumerate(self.neurons):
            for j, post in enumerate(self.neurons):
                if i == j:
                    continue

                # Check layer connectivity
                layer_pair = (pre.layer, post.layer)
                if layer_pair in layer_conn:
                    prob = layer_conn[layer_pair]

                    # Distance-dependent probability
                    dist = np.linalg.norm(pre.position - post.position)
                    dist_prob = np.exp(-dist / 50)

                    if np.random.rand() < prob * dist_prob:
                        # Excitatory or inhibitory
                        if i < self.n_exc:
                            W[i, j] = np.random.gamma(2, 0.5)
                        else:
                            W[i, j] = -np.random.gamma(2, 2)

        return W

    def simulate(self, duration: float, external_input: Optional[np.ndarray] = None,
                 dt: float = 0.1) -> Dict:
        """Simulate network dynamics."""
        n_steps = int(duration / dt)

        # Initialize recording arrays
        spike_raster = []
        voltage_trace = np.zeros((n_steps, self.n_neurons))
        calcium_trace = np.zeros((n_steps, self.n_neurons))

        # External input
        if external_input is None:
            external_input = np.random.randn(n_steps, self.n_neurons) * 5 + 10

        # Main simulation loop
        for t_idx in range(n_steps):
            t = t_idx * dt

            # Update synaptic traces (exponential decay)
            self.synaptic_traces[:, 0] *= np.exp(-dt / TAU_AMPA)  # AMPA
            self.synaptic_traces[:, 1] *= np.exp(-dt / TAU_NMDA)  # NMDA
            self.synaptic_traces[:, 2] *= np.exp(-dt / TAU_GABA_A)  # GABA-A
            self.synaptic_traces[:, 3] *= np.exp(-dt / TAU_GABA_B)  # GABA-B

            # Compute synaptic inputs
            I_syn = self._compute_synaptic_current(t)

            # Update each neuron
            spikes = np.zeros(self.n_neurons, dtype=bool)
            for i, neuron in enumerate(self.neurons):
                # Total input current
                I_total = I_syn[i] + external_input[t_idx, i]

                # Update membrane potential
                dV = self._neuron_dynamics(neuron, I_total, dt)
                neuron.V += dV

                # Check for spike
                if neuron.V >= neuron.V_thresh:
                    spikes[i] = True
                    neuron.V = neuron.V_reset
                    neuron.spike_times.append(t)
                    spike_raster.append((t, i))

                    # Update synaptic traces
                    if i < self.n_exc:
                        self.synaptic_traces[i, 0] += 1.0  # AMPA
                        self.synaptic_traces[i, 1] += 0.5  # NMDA
                    else:
                        self.synaptic_traces[i, 2] += 1.0  # GABA-A
                        self.synaptic_traces[i, 3] += 0.3  # GABA-B

                # Update calcium dynamics
                neuron.calcium *= np.exp(-dt / 50)  # Decay
                if spikes[i]:
                    neuron.calcium += 0.1  # Spike-triggered calcium

                # Record states
                voltage_trace[t_idx, i] = neuron.V
                calcium_trace[t_idx, i] = neuron.calcium

            # Apply STDP if enabled
            if self.learning_rate > 0:
                self._apply_stdp(spikes, t)

        return {
            'spike_raster': spike_raster,
            'voltage_trace': voltage_trace,
            'calcium_trace': calcium_trace,
            'firing_rates': self._calculate_firing_rates(duration),
            'synchrony': self._calculate_synchrony(spike_raster),
            'time': np.arange(n_steps) * dt
        }

    def _neuron_dynamics(self, neuron: BiologicalNeuron, I_ext: float, dt: float) -> float:
        """Compute neuron membrane potential dynamics."""
        # Simplified adaptive exponential integrate-and-fire
        V = neuron.V
        tau = neuron.tau_m

        # Exponential spike mechanism
        if V > neuron.V_thresh - 5:
            exp_term = np.exp((V - neuron.V_thresh) / 5)
        else:
            exp_term = 0

        # Membrane equation
        dV = (-(V - neuron.V_rest) + exp_term + I_ext) / tau

        # Adaptation current
        neuron.u += dt * (neuron.adaptation * (V - neuron.V_rest) - neuron.u) / 100
        dV -= neuron.u / tau

        return dV * dt

    def _compute_synaptic_current(self, t: float) -> np.ndarray:
        """Compute synaptic currents for all neurons."""
        I_syn = np.zeros(self.n_neurons)

        # AMPA (fast excitatory)
        I_syn += self.W.T @ self.synaptic_traces[:, 0]

        # NMDA (slow excitatory, voltage-dependent)
        for i in range(self.n_neurons):
            V = self.neurons[i].V
            mg_block = 1 / (1 + np.exp(-0.062 * V) * 0.33)  # Mg2+ block
            I_syn[i] += mg_block * (self.W[:, i] @ self.synaptic_traces[:, 1])

        # GABA-A (fast inhibitory)
        I_syn += self.W.T @ self.synaptic_traces[:, 2]

        # GABA-B (slow inhibitory)
        I_syn += 0.5 * self.W.T @ self.synaptic_traces[:, 3]

        return I_syn

    def _apply_stdp(self, spikes: np.ndarray, t: float):
        """Apply spike-timing dependent plasticity."""
        spike_indices = np.where(spikes)[0]

        for post_idx in spike_indices:
            # Look for recent pre-synaptic spikes
            for pre_idx in range(self.n_neurons):
                if pre_idx == post_idx or self.W[pre_idx, post_idx] == 0:
                    continue

                # Get recent spike times
                if len(self.neurons[pre_idx].spike_times) > 0:
                    pre_spike_time = self.neurons[pre_idx].spike_times[-1]
                    dt_spike = t - pre_spike_time

                    if abs(dt_spike) < self.stdp_window:
                        # STDP update
                        if dt_spike > 0:  # Pre before post (LTP)
                            dw = self.learning_rate * np.exp(-dt_spike / 20)
                        else:  # Post before pre (LTD)
                            dw = -self.learning_rate * np.exp(dt_spike / 20)

                        # Update weight with bounds
                        self.W[pre_idx, post_idx] += dw
                        self.W[pre_idx, post_idx] = np.clip(
                            self.W[pre_idx, post_idx], -10, 10
                        )

    def _calculate_firing_rates(self, duration: float) -> np.ndarray:
        """Calculate average firing rates."""
        rates = np.zeros(self.n_neurons)
        for i, neuron in enumerate(self.neurons):
            rates[i] = len(neuron.spike_times) / duration * 1000  # Hz
        return rates

    def _calculate_synchrony(self, spike_raster: List[Tuple[float, int]]) -> float:
        """Calculate network synchronization (Kuramoto order parameter)."""
        if len(spike_raster) < 2:
            return 0.0

        # Convert spikes to phases
        spike_times = np.array([s[0] for s in spike_raster])
        T = spike_times[-1] if len(spike_times) > 0 else 1.0

        # Calculate instantaneous phases
        phases = 2 * np.pi * spike_times / T

        # Kuramoto order parameter
        r = np.abs(np.mean(np.exp(1j * phases)))
        return r

class HebbianLearning:
    """Hebbian learning rules for neural networks."""

    @staticmethod
    def basic_hebb(pre: np.ndarray, post: np.ndarray, lr: float = 0.01) -> np.ndarray:
        """Basic Hebbian rule: neurons that fire together wire together."""
        return lr * np.outer(post, pre)

    @staticmethod
    def oja_rule(W: np.ndarray, pre: np.ndarray, post: np.ndarray,
                 lr: float = 0.01) -> np.ndarray:
        """Oja's rule: Hebbian with weight normalization."""
        dW = lr * (np.outer(post, pre) - post[:, np.newaxis]**2 * W)
        return dW

    @staticmethod
    def bcm_rule(W: np.ndarray, pre: np.ndarray, post: np.ndarray,
                 theta: float, lr: float = 0.01, tau_theta: float = 1000) -> Tuple[np.ndarray, float]:
        """Bienenstock-Cooper-Munro rule with sliding threshold."""
        # Weight update
        dW = lr * pre * post[:, np.newaxis] * (post[:, np.newaxis] - theta)

        # Threshold update
        dtheta = (np.mean(post**2) - theta) / tau_theta

        return dW, dtheta

    @staticmethod
    def covariance_rule(W: np.ndarray, pre: np.ndarray, post: np.ndarray,
                        pre_avg: np.ndarray, post_avg: np.ndarray,
                        lr: float = 0.01) -> np.ndarray:
        """Covariance learning rule."""
        pre_centered = pre - pre_avg
        post_centered = post - post_avg
        dW = lr * np.outer(post_centered, pre_centered)
        return dW

class WinnerTakeAllNetwork:
    """Competitive winner-take-all network."""

    def __init__(self, n_units: int, n_inputs: int):
        self.n_units = n_units
        self.n_inputs = n_inputs

        # Weight matrix
        self.W = np.random.randn(n_units, n_inputs) * 0.1

        # Lateral inhibition
        self.lateral_inhibition = -np.ones((n_units, n_units)) + np.eye(n_units) * 2

    def compete(self, input_pattern: np.ndarray, iterations: int = 10) -> np.ndarray:
        """Run competition dynamics."""
        # Initial activation
        activation = self.W @ input_pattern

        # Iterative competition
        for _ in range(iterations):
            # Apply lateral inhibition
            activation = self.W @ input_pattern + self.lateral_inhibition @ activation

            # Soft winner-take-all (softmax-like)
            activation = np.maximum(activation, 0)  # ReLU
            if activation.sum() > 0:
                activation = activation**2 / (activation.sum() + 1e-8)

        # Hard winner-take-all
        winner = np.argmax(activation)
        output = np.zeros(self.n_units)
        output[winner] = 1.0

        return output

    def train(self, patterns: np.ndarray, epochs: int = 100, lr: float = 0.01):
        """Train network with competitive learning."""
        for epoch in range(epochs):
            for pattern in patterns:
                # Competition
                output = self.compete(pattern)
                winner = np.argmax(output)

                # Update winner's weights (move towards input)
                self.W[winner] += lr * (pattern - self.W[winner])

                # Normalize weights
                self.W[winner] /= np.linalg.norm(self.W[winner]) + 1e-8

            # Decay learning rate
            lr *= 0.99

class AttractorNetwork:
    """Hopfield-like attractor network for associative memory."""

    def __init__(self, n_neurons: int):
        self.n_neurons = n_neurons
        self.W = np.zeros((n_neurons, n_neurons))
        self.patterns = []

    def store_pattern(self, pattern: np.ndarray):
        """Store a pattern in the network."""
        pattern = np.sign(pattern)  # Binarize to ±1
        self.patterns.append(pattern)

        # Hebbian learning
        self.W += np.outer(pattern, pattern) / self.n_neurons

        # Remove self-connections
        np.fill_diagonal(self.W, 0)

    def recall(self, partial_pattern: np.ndarray, max_iterations: int = 100) -> np.ndarray:
        """Recall complete pattern from partial/noisy input."""
        state = partial_pattern.copy()
        energy_history = []

        for iteration in range(max_iterations):
            # Asynchronous update
            for i in np.random.permutation(self.n_neurons):
                activation = self.W[i] @ state
                state[i] = np.sign(activation) if activation != 0 else state[i]

            # Calculate energy
            energy = -0.5 * state @ self.W @ state
            energy_history.append(energy)

            # Check for convergence
            if len(energy_history) > 10:
                if np.std(energy_history[-10:]) < 1e-6:
                    break

        return state

    def capacity(self) -> int:
        """Estimate network capacity (number of patterns that can be stored)."""
        # Theoretical capacity is approximately 0.138 * N for Hopfield networks
        return int(0.138 * self.n_neurons)

    def spurious_states(self) -> List[np.ndarray]:
        """Find spurious states (unintended attractors)."""
        spurious = []

        # Check mixture states
        for i in range(len(self.patterns)):
            for j in range(i + 1, len(self.patterns)):
                # Linear combination
                mixture = np.sign(self.patterns[i] + self.patterns[j])
                recalled = self.recall(mixture)

                # Check if it's a spurious state
                is_spurious = True
                for pattern in self.patterns:
                    if np.allclose(recalled, pattern):
                        is_spurious = False
                        break

                if is_spurious:
                    spurious.append(recalled)

        return spurious

class NeuralOscillator:
    """Neural oscillator networks for rhythm generation."""

    def __init__(self, n_oscillators: int):
        self.n_oscillators = n_oscillators

        # Natural frequencies (gamma band: 30-80 Hz)
        self.omega = np.random.uniform(30, 80, n_oscillators) * 2 * np.pi / 1000  # rad/ms

        # Coupling matrix
        self.K = np.random.randn(n_oscillators, n_oscillators) * 0.1
        np.fill_diagonal(self.K, 0)

        # Phase and amplitude
        self.phase = np.random.uniform(0, 2*np.pi, n_oscillators)
        self.amplitude = np.ones(n_oscillators)

    def kuramoto_dynamics(self, dt: float = 0.1) -> np.ndarray:
        """Kuramoto model dynamics for phase coupling."""
        # Phase dynamics
        coupling = np.zeros(self.n_oscillators)
        for i in range(self.n_oscillators):
            for j in range(self.n_oscillators):
                if i != j:
                    coupling[i] += self.K[i, j] * np.sin(self.phase[j] - self.phase[i])

        dphase = self.omega + coupling / self.n_oscillators
        self.phase += dphase * dt
        self.phase = np.mod(self.phase, 2 * np.pi)

        return self.phase

    def wilson_cowan_dynamics(self, E: np.ndarray, I: np.ndarray,
                             dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Wilson-Cowan model for E-I oscillations."""
        # Parameters
        tau_E, tau_I = 10.0, 20.0  # ms
        c_EE, c_EI = 12.0, 4.0
        c_IE, c_II = 13.0, 11.0
        theta_E, theta_I = 2.8, 4.0
        a_E, a_I = 1.2, 1.0
        P = 1.0  # External input

        # Sigmoid activation
        def S(x, theta, a):
            return 1 / (1 + np.exp(-a * (x - theta)))

        # Dynamics
        dE = (-E + S(c_EE * E - c_EI * I + P, theta_E, a_E)) / tau_E
        dI = (-I + S(c_IE * E - c_II * I, theta_I, a_I)) / tau_I

        E_new = E + dE * dt
        I_new = I + dI * dt

        return E_new, I_new

    def simulate_oscillations(self, duration: float, dt: float = 0.1) -> Dict:
        """Simulate neural oscillations."""
        n_steps = int(duration / dt)
        time = np.arange(n_steps) * dt

        # Record variables
        phases = np.zeros((n_steps, self.n_oscillators))
        synchrony = np.zeros(n_steps)

        # E-I populations
        E = np.random.rand(self.n_oscillators) * 0.5
        I = np.random.rand(self.n_oscillators) * 0.5
        E_trace = np.zeros((n_steps, self.n_oscillators))
        I_trace = np.zeros((n_steps, self.n_oscillators))

        for t_idx in range(n_steps):
            # Update phases (Kuramoto)
            phases[t_idx] = self.kuramoto_dynamics(dt)

            # Calculate synchrony (order parameter)
            synchrony[t_idx] = np.abs(np.mean(np.exp(1j * phases[t_idx])))

            # Update E-I dynamics
            E, I = self.wilson_cowan_dynamics(E, I, dt)
            E_trace[t_idx] = E
            I_trace[t_idx] = I

        # Compute frequency spectrum
        freqs = np.fft.fftfreq(n_steps, dt)
        spectrum = np.abs(np.fft.fft(E_trace.mean(axis=1)))**2

        return {
            'time': time,
            'phases': phases,
            'synchrony': synchrony,
            'E_population': E_trace,
            'I_population': I_trace,
            'frequencies': freqs[:n_steps//2],
            'power_spectrum': spectrum[:n_steps//2],
            'dominant_frequency': freqs[np.argmax(spectrum[:n_steps//2])]
        }

class PopulationCoding:
    """Population coding and decoding methods."""

    def __init__(self, n_neurons: int, n_dimensions: int):
        self.n_neurons = n_neurons
        self.n_dimensions = n_dimensions

        # Preferred directions (random unit vectors)
        self.preferred_directions = np.random.randn(n_neurons, n_dimensions)
        norms = np.linalg.norm(self.preferred_directions, axis=1, keepdims=True)
        self.preferred_directions /= norms

        # Tuning widths
        self.tuning_widths = np.random.uniform(0.5, 2.0, n_neurons)

    def encode(self, stimulus: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """Encode stimulus into population activity."""
        # Cosine tuning
        responses = np.zeros(self.n_neurons)
        for i in range(self.n_neurons):
            projection = np.dot(stimulus, self.preferred_directions[i])
            responses[i] = np.exp(projection / self.tuning_widths[i])

        # Add Poisson noise
        responses = np.random.poisson(responses * 10 + noise_level) / 10

        return responses

    def decode_vector(self, population_activity: np.ndarray) -> np.ndarray:
        """Decode stimulus from population vector."""
        # Population vector decoding
        weighted_sum = population_activity @ self.preferred_directions
        total_activity = population_activity.sum() + 1e-8

        decoded = weighted_sum / total_activity
        return decoded

    def decode_bayesian(self, population_activity: np.ndarray,
                       prior_mean: np.ndarray = None,
                       prior_cov: np.ndarray = None) -> np.ndarray:
        """Bayesian decoding with prior."""
        if prior_mean is None:
            prior_mean = np.zeros(self.n_dimensions)
        if prior_cov is None:
            prior_cov = np.eye(self.n_dimensions)

        # Likelihood (assuming Poisson neurons)
        def log_likelihood(stimulus):
            expected_rates = self.encode(stimulus, noise_level=0)
            return np.sum(population_activity * np.log(expected_rates + 1e-8) -
                         expected_rates)

        # Maximum a posteriori estimate
        def neg_log_posterior(stimulus):
            prior_term = 0.5 * (stimulus - prior_mean).T @ np.linalg.inv(prior_cov) @ \
                        (stimulus - prior_mean)
            return -log_likelihood(stimulus) + prior_term

        # Optimize
        from scipy.optimize import minimize
        result = minimize(neg_log_posterior, prior_mean, method='L-BFGS-B')

        return result.x

    def fisher_information(self, stimulus: np.ndarray) -> np.ndarray:
        """Calculate Fisher information matrix."""
        # Tuning curve derivatives
        derivatives = np.zeros((self.n_neurons, self.n_dimensions))
        rates = self.encode(stimulus, noise_level=0)

        for i in range(self.n_neurons):
            derivatives[i] = (rates[i] / self.tuning_widths[i]) * \
                           self.preferred_directions[i]

        # Fisher information
        F = np.zeros((self.n_dimensions, self.n_dimensions))
        for i in range(self.n_neurons):
            F += np.outer(derivatives[i], derivatives[i]) / (rates[i] + 1e-8)

        return F

class CorticalColumn:
    """Canonical cortical microcircuit model."""

    def __init__(self, n_neurons_per_layer: int = 100):
        self.layers = {
            'L1': n_neurons_per_layer // 10,  # Molecular layer
            'L2/3': n_neurons_per_layer,  # Supragranular
            'L4': n_neurons_per_layer,  # Granular (input)
            'L5': n_neurons_per_layer,  # Infragranular (output)
            'L6': n_neurons_per_layer // 2  # Deep layers
        }

        self.n_total = sum(self.layers.values())

        # Create connectivity matrix
        self.W = self._create_laminar_connectivity()

        # Cell types per layer (E/I ratio)
        self.cell_types = self._assign_cell_types()

    def _create_laminar_connectivity(self) -> np.ndarray:
        """Create layer-specific connectivity."""
        W = np.zeros((self.n_total, self.n_total))

        # Canonical connections based on cortical anatomy
        connections = {
            ('L4', 'L2/3'): 0.5,  # Feedforward
            ('L2/3', 'L2/3'): 0.3,  # Recurrent
            ('L2/3', 'L5'): 0.4,  # Feedforward
            ('L5', 'L5'): 0.3,  # Recurrent
            ('L5', 'L6'): 0.3,  # Deep projection
            ('L6', 'L4'): 0.2,  # Feedback
            ('L6', 'L1'): 0.1,  # Modulatory
            ('L1', 'L2/3'): 0.1,  # Top-down
        }

        # Build connectivity
        layer_indices = self._get_layer_indices()

        for (pre_layer, post_layer), strength in connections.items():
            pre_idx = layer_indices[pre_layer]
            post_idx = layer_indices[post_layer]

            n_pre = len(pre_idx)
            n_post = len(post_idx)

            # Random connectivity with specified density
            conn_matrix = np.random.rand(n_post, n_pre) < strength
            weights = conn_matrix * np.random.gamma(2, 0.5, (n_post, n_pre))

            # Insert into main matrix
            for i, post in enumerate(post_idx):
                for j, pre in enumerate(pre_idx):
                    W[post, pre] = weights[i, j]

        return W

    def _assign_cell_types(self) -> Dict[str, np.ndarray]:
        """Assign excitatory/inhibitory cell types."""
        cell_types = {}
        idx = 0

        for layer, n_neurons in self.layers.items():
            # 80% excitatory, 20% inhibitory (except L1)
            if layer == 'L1':
                # L1 is mostly inhibitory
                n_exc = int(0.1 * n_neurons)
            else:
                n_exc = int(0.8 * n_neurons)

            types = np.zeros(n_neurons, dtype=bool)
            types[:n_exc] = True  # True for excitatory
            cell_types[layer] = types
            idx += n_neurons

        return cell_types

    def _get_layer_indices(self) -> Dict[str, np.ndarray]:
        """Get neuron indices for each layer."""
        indices = {}
        start = 0

        for layer, n_neurons in self.layers.items():
            indices[layer] = np.arange(start, start + n_neurons)
            start += n_neurons

        return indices

    def process_input(self, thalamic_input: np.ndarray,
                     simulation_time: float = 100.0) -> Dict:
        """Process thalamic input through column."""
        dt = 0.1
        n_steps = int(simulation_time / dt)

        # Initialize neurons
        V = np.random.randn(self.n_total) * 10 - 70
        spike_times = []

        # Target L4 with thalamic input
        layer_indices = self._get_layer_indices()
        L4_indices = layer_indices['L4']

        # Run simulation
        for t_idx in range(n_steps):
            t = t_idx * dt

            # External input to L4
            I_ext = np.zeros(self.n_total)
            if t < 50:  # Stimulus period
                I_ext[L4_indices] = thalamic_input

            # Simple integrate-and-fire dynamics
            I_syn = self.W.T @ (V > -50).astype(float) * 10
            dV = (-70 - V) / 20 + I_syn + I_ext

            V += dV * dt

            # Spiking
            spiking = V > -50
            spike_times.extend([(t, i) for i in np.where(spiking)[0]])
            V[spiking] = -70

        # Analyze layer-wise activity
        layer_rates = {}
        for layer, indices in layer_indices.items():
            layer_spikes = [s for s in spike_times if s[1] in indices]
            layer_rates[layer] = len(layer_spikes) / (simulation_time * len(indices) / 1000)

        return {
            'spike_times': spike_times,
            'layer_rates': layer_rates,
            'total_spikes': len(spike_times)
        }

class ReservoirComputing:
    """Echo state network with biological neurons."""

    def __init__(self, n_reservoir: int = 1000, spectral_radius: float = 0.9,
                 sparsity: float = 0.1):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity

        # Create recurrent weights
        self.W_res = self._create_reservoir()

        # Input and output weights
        self.W_in = None
        self.W_out = None

        # Reservoir state
        self.state = np.zeros(n_reservoir)

    def _create_reservoir(self) -> np.ndarray:
        """Create sparse recurrent weight matrix."""
        # Random sparse matrix
        W = sparse.random(self.n_reservoir, self.n_reservoir,
                         density=self.sparsity, format='csr')
        W = W.toarray()

        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(W)
        current_radius = np.max(np.abs(eigenvalues))

        if current_radius > 0:
            W *= self.spectral_radius / current_radius

        return W

    def train(self, input_data: np.ndarray, target_data: np.ndarray,
              regularization: float = 1e-6):
        """Train reservoir readout weights."""
        n_samples, n_inputs = input_data.shape
        _, n_outputs = target_data.shape

        # Initialize input weights if needed
        if self.W_in is None:
            self.W_in = np.random.uniform(-1, 1, (self.n_reservoir, n_inputs))

        # Collect reservoir states
        states = []
        self.state = np.zeros(self.n_reservoir)

        for t in range(n_samples):
            self.state = np.tanh(self.W_res @ self.state + self.W_in @ input_data[t])
            states.append(self.state.copy())

        states = np.array(states)

        # Train readout with ridge regression
        X = states
        Y = target_data

        # Ridge regression: W_out = (X'X + λI)^(-1) X'Y
        XTX = X.T @ X
        XTY = X.T @ Y
        self.W_out = np.linalg.solve(XTX + regularization * np.eye(self.n_reservoir), XTY).T

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        n_samples = input_data.shape[0]
        predictions = []

        for t in range(n_samples):
            self.state = np.tanh(self.W_res @ self.state + self.W_in @ input_data[t])
            output = self.W_out @ self.state
            predictions.append(output)

        return np.array(predictions)

    def compute_memory_capacity(self, max_delay: int = 100) -> np.ndarray:
        """Compute memory capacity of reservoir."""
        # Generate random input
        input_signal = np.random.uniform(-1, 1, (max_delay * 10, 1))

        # Initialize
        if self.W_in is None:
            self.W_in = np.random.uniform(-1, 1, (self.n_reservoir, 1))

        capacities = []

        for delay in range(1, max_delay + 1):
            # Create delayed target
            target = np.roll(input_signal, delay, axis=0)
            target[:delay] = 0

            # Train and test
            self.train(input_signal, target)
            predictions = self.predict(input_signal)

            # Calculate capacity (squared correlation)
            correlation = np.corrcoef(predictions[delay:].flatten(),
                                     target[delay:].flatten())[0, 1]
            capacities.append(correlation**2)

        return np.array(capacities)

def demonstrate_neural_networks_lab():
    """Comprehensive demonstration of neural network capabilities."""
    print("=" * 80)
    print("BIOLOGICAL NEURAL NETWORKS LAB DEMONSTRATION")
    print("Copyright (c) 2025 Corporation of Light. All Rights Reserved.")
    print("=" * 80)

    # 1. Spiking Neural Network
    print("\n1. SPIKING NEURAL NETWORK")
    print("-" * 40)

    snn = SpikingNeuralNetwork(n_neurons=100, connectivity='random')
    snn_result = snn.simulate(duration=200, dt=0.1)

    print(f"Network size: {snn.n_neurons} neurons")
    print(f"Total spikes: {len(snn_result['spike_raster'])}")
    print(f"Mean firing rate: {snn_result['firing_rates'].mean():.1f} Hz")
    print(f"Network synchrony: {snn_result['synchrony']:.3f}")

    # 2. Hebbian Learning
    print("\n2. HEBBIAN LEARNING RULES")
    print("-" * 40)

    # Create sample pre/post activity
    pre = np.random.rand(100)
    post = np.random.rand(50)
    W = np.random.randn(50, 100) * 0.1

    # Apply different learning rules
    dW_hebb = HebbianLearning.basic_hebb(pre, post)
    dW_oja = HebbianLearning.oja_rule(W, pre, post)
    dW_bcm, dtheta = HebbianLearning.bcm_rule(W, pre, post, theta=0.5)

    print(f"Hebbian weight change: mean={dW_hebb.mean():.4f}, std={dW_hebb.std():.4f}")
    print(f"Oja weight change: mean={dW_oja.mean():.4f}, std={dW_oja.std():.4f}")
    print(f"BCM weight change: mean={dW_bcm.mean():.4f}, threshold change={dtheta:.4f}")

    # 3. Winner-Take-All
    print("\n3. WINNER-TAKE-ALL NETWORK")
    print("-" * 40)

    wta = WinnerTakeAllNetwork(n_units=10, n_inputs=20)

    # Create input patterns
    patterns = np.random.randn(5, 20)

    # Train network
    wta.train(patterns, epochs=50)

    # Test competition
    test_input = patterns[0] + np.random.randn(20) * 0.1
    winner = wta.compete(test_input)

    print(f"Network units: {wta.n_units}")
    print(f"Winner unit: {np.argmax(winner)}")
    print(f"Competition iterations: 10")

    # 4. Attractor Network
    print("\n4. HOPFIELD ATTRACTOR NETWORK")
    print("-" * 40)

    attractor = AttractorNetwork(n_neurons=100)

    # Store patterns
    patterns_to_store = [np.sign(np.random.randn(100)) for _ in range(5)]
    for pattern in patterns_to_store:
        attractor.store_pattern(pattern)

    # Test recall with noisy pattern
    noisy = patterns_to_store[0].copy()
    noise_indices = np.random.choice(100, 20, replace=False)
    noisy[noise_indices] *= -1  # Flip 20% of bits

    recalled = attractor.recall(noisy)
    overlap = np.dot(recalled, patterns_to_store[0]) / 100

    print(f"Stored patterns: {len(patterns_to_store)}")
    print(f"Network capacity: {attractor.capacity()} patterns")
    print(f"Recall overlap: {overlap:.2f}")
    print(f"Noise level: 20% bit flips")

    # 5. Neural Oscillations
    print("\n5. NEURAL OSCILLATIONS")
    print("-" * 40)

    oscillator = NeuralOscillator(n_oscillators=20)
    osc_result = oscillator.simulate_oscillations(duration=500, dt=0.1)

    print(f"Number of oscillators: {oscillator.n_oscillators}")
    print(f"Mean synchrony: {osc_result['synchrony'].mean():.3f}")
    print(f"Dominant frequency: {abs(osc_result['dominant_frequency']):.1f} Hz")

    # Find frequency bands
    power = osc_result['power_spectrum']
    freqs = osc_result['frequencies']
    gamma_power = power[(freqs > 30) & (freqs < 80)].sum()
    beta_power = power[(freqs > 13) & (freqs < 30)].sum()

    print(f"Gamma band power: {gamma_power:.2f}")
    print(f"Beta band power: {beta_power:.2f}")

    # 6. Population Coding
    print("\n6. POPULATION CODING")
    print("-" * 40)

    pop_code = PopulationCoding(n_neurons=100, n_dimensions=2)

    # Encode stimulus
    stimulus = np.array([1.0, 0.5])
    population_response = pop_code.encode(stimulus)

    # Decode
    decoded_vector = pop_code.decode_vector(population_response)
    decoded_bayesian = pop_code.decode_bayesian(population_response)

    # Fisher information
    fisher = pop_code.fisher_information(stimulus)

    print(f"True stimulus: {stimulus}")
    print(f"Vector decoding: {decoded_vector}")
    print(f"Bayesian decoding: {decoded_bayesian}")
    print(f"Decoding error (vector): {np.linalg.norm(stimulus - decoded_vector):.3f}")
    print(f"Decoding error (Bayesian): {np.linalg.norm(stimulus - decoded_bayesian):.3f}")
    print(f"Fisher information (trace): {np.trace(fisher):.2f}")

    # 7. Cortical Column
    print("\n7. CORTICAL COLUMN SIMULATION")
    print("-" * 40)

    column = CorticalColumn(n_neurons_per_layer=50)

    # Thalamic input
    thalamic_input = np.random.randn(column.layers['L4']) * 10

    # Process input
    column_result = column.process_input(thalamic_input, simulation_time=100)

    print(f"Total neurons: {column.n_total}")
    print("Layer-wise firing rates (Hz):")
    for layer, rate in column_result['layer_rates'].items():
        print(f"  {layer}: {rate:.1f}")
    print(f"Total spikes: {column_result['total_spikes']}")

    # 8. Reservoir Computing
    print("\n8. RESERVOIR COMPUTING")
    print("-" * 40)

    reservoir = ReservoirComputing(n_reservoir=200, spectral_radius=0.95)

    # Generate time series task (e.g., Mackey-Glass)
    def mackey_glass(n_samples, tau=17):
        x = np.zeros(n_samples)
        x[0] = 1.2
        for t in range(1, n_samples):
            if t > tau:
                x[t] = x[t-1] + 0.2 * x[t-tau] / (1 + x[t-tau]**10) - 0.1 * x[t-1]
            else:
                x[t] = x[t-1]
        return x

    # Create data
    time_series = mackey_glass(1000)
    input_data = time_series[:-1].reshape(-1, 1)
    target_data = time_series[1:].reshape(-1, 1)

    # Train reservoir
    train_size = 700
    reservoir.train(input_data[:train_size], target_data[:train_size])

    # Test prediction
    predictions = reservoir.predict(input_data[train_size:])
    mse = np.mean((predictions - target_data[train_size:])**2)

    print(f"Reservoir size: {reservoir.n_reservoir} neurons")
    print(f"Spectral radius: {reservoir.spectral_radius}")
    print(f"Sparsity: {reservoir.sparsity}")
    print(f"Prediction MSE: {mse:.6f}")

    # Memory capacity
    memory_cap = reservoir.compute_memory_capacity(max_delay=30)
    total_capacity = memory_cap.sum()
    print(f"Memory capacity: {total_capacity:.1f} (theoretical max: {reservoir.n_reservoir})")

    print("\n" + "=" * 80)
    print("Demonstration complete. Visit aios.is for more information.")
    print("=" * 80)

if __name__ == '__main__':
    demonstrate_neural_networks_lab()