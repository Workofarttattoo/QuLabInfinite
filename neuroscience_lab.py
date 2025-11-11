"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

NEUROSCIENCE LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.constants import *
from typing import *

@dataclass
class Neuron:
    """Data class representing a neuron with properties."""
    name: str = "generic_neuron"
    resting_potential: float = -70.0  # mV
    threshold_potential: float = -55.0  # mV
    refractory_period: float = 2.0  # ms
    resistance_membrane: float = 1.0e6  # ohm cm^2
    capacitance_membrane: float = 1.0e-6  # F/cm^2
    time_constant: float = resistance_membrane * capacitance_membrane  # ms

@dataclass
class Synapse:
    """Data class representing a synapse with properties."""
    name: str = "generic_synapse"
    strength: float = 0.5  # arbitrary unit, positive for excitatory, negative for inhibitory
    delay: float = 1.0  # ms

@dataclass
class Network:
    """Data class representing a neural network with neurons and synapses."""
    name: str = "generic_network"
    neurons: List[Neuron] = field(default_factory=list)
    synapses: Dict[Tuple[str, str], Synapse] = field(default_factory=dict)

def simulate_voltage_response(neuron: Neuron, stimulus_strength: float) -> np.ndarray:
    """Simulate the voltage response of a neuron to an external stimulus."""
    dt = 0.1  # ms
    time_steps = int(200 / dt)
    current_time = 0.0

    v_rest = neuron.resting_potential
    resistance_membrane = neuron.resistance_membrane
    capacitance_membrane = neuron.capacitance_membrane
    threshold_voltage = neuron.threshold_potential

    voltage_history = np.zeros(time_steps, dtype=np.float64)
    for t in range(1, time_steps):
        current_time += dt
        input_current = stimulus_strength * (v_rest - v_infinity(v_rest))
        dvdt = (-v_rest + v_infinity(v_rest)) / neuron.time_constant

        voltage_history[t] = v_rest + dvdt * dt

        if v_rest >= threshold_voltage:
            v_rest -= 10
            current_time += neuron.refractory_period

    return voltage_history

def simulate_network_response(network: Network, stimulus_strengths: List[float]) -> Dict[str, np.ndarray]:
    """Simulate the network response to multiple stimuli."""
    results = {}
    for i, neuron in enumerate(network.neurons):
        results[f"neuron_{i}"] = simulate_voltage_response(neuron, stimulus_strengths[i])
    return results

def v_infinity(v: float) -> float:
    """Helper function for calculating steady-state potential."""
    alpha_v = 0.1 * (v + 55) / (np.exp((v + 55) / 10) - 1)
    beta_v = 0.125 * np.exp(-0.0125 * (v + 65))
    return alpha_v / (alpha_v + beta_v)

def run_demo():
    neuron1 = Neuron(name="pyramidal_cell", resting_potential=-75, threshold_potential=-50)
    synapse1 = Synapse(name="excitatory_synapse", strength=0.8)
    network = Network()
    network.neurons.append(neuron1)

    stimulus_strengths = [20.0]
    results = simulate_network_response(network, stimulus_strengths)
    for neuron_id, voltages in results.items():
        print(f"Voltage response of {neuron_id}:")
        print(voltages)

if __name__ == '__main__':
    run_demo()