"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Neuroscience Laboratory Demo
"""

from neuroscience_lab import NeuroscienceLaboratory, NeuronType, Neurotransmitter
import numpy as np


def main():
    """Run neuroscience lab demonstration"""
    print("QuLabInfinite Neuroscience Laboratory - Demo")
    print("=" * 70)

    lab = NeuroscienceLaboratory(seed=42)

    # Demo 1: Single neuron spiking
    print("\n1. Single Neuron Spiking Simulation")
    print("-" * 70)

    for neuron_type in [NeuronType.EXCITATORY, NeuronType.INHIBITORY]:
        params = lab.create_izhikevich_neuron(neuron_type)
        time, voltage = lab.simulate_neuron_spike(params, input_current=15, duration_ms=100)

        spikes = np.sum(voltage >= 30)
        print(f"{neuron_type.value.capitalize()}: {spikes} spikes in 100ms")
        print(f"  Resting: {voltage[0]:.1f} mV, Peak: {np.max(voltage):.1f} mV")

    # Demo 2: Neural network dynamics
    print("\n2. Neural Network Simulation")
    print("-" * 70)

    network_sizes = [50, 100, 200]
    for n in network_sizes:
        result = lab.simulate_neural_network(n, connectivity=0.1, duration_ms=500)
        print(f"Network with {n} neurons:")
        print(f"  Total spikes: {result['total_spikes']}")
        print(f"  Avg firing rate: {result['avg_firing_rate']:.2f} Hz")
        print(f"  Synchronization: {result['synchronization']:.3f}")

    # Demo 3: Neurotransmitter dynamics
    print("\n3. Neurotransmitter Dynamics")
    print("-" * 70)

    nt_types = [Neurotransmitter.GLUTAMATE, Neurotransmitter.GABA,
                Neurotransmitter.DOPAMINE, Neurotransmitter.SEROTONIN]

    stim_times = [10, 20, 30, 40, 50, 60]  # ms

    for nt in nt_types:
        result = lab.simulate_neurotransmitter_dynamics(nt, stim_times, duration_ms=150)
        print(f"{nt.value.capitalize()}:")
        print(f"  Peak concentration: {result['peak_concentration']:.4f}")
        print(f"  Total exposure (AUC): {result['area_under_curve']:.2f}")
        print(f"  Clearance time: {result['clearance_time_ms']:.1f} ms")

    # Demo 4: EEG brain states
    print("\n4. EEG Brain State Simulation")
    print("-" * 70)

    brain_states = ['awake', 'drowsy', 'asleep', 'deep_sleep']

    for state in brain_states:
        eeg = lab.simulate_eeg_signal(state, duration_s=5, sampling_rate=256)
        print(f"\n{state.upper()} state:")
        print(f"  Dominant frequency: {eeg.dominant_frequency:.2f} Hz")
        print(f"  Synchronization: {eeg.synchronization:.3f}")
        print(f"  Consciousness metric: {eeg.consciousness_metric:.3f}")
        print(f"  Power spectrum:")
        for band, power in sorted(eeg.power_spectrum.items(),
                                 key=lambda x: x[1], reverse=True)[:3]:
            print(f"    {band}: {power:.1f}")

    # Demo 5: Consciousness measurement
    print("\n5. Consciousness Metric (Integrated Information)")
    print("-" * 70)

    # Simulate different levels of neural integration
    scenarios = {
        'highly_integrated': lambda: np.tile(np.sin(np.linspace(0, 10, 100)), (50, 1)),
        'moderately_integrated': lambda: np.array([np.sin(np.linspace(0, 10, 100) + i*0.5)
                                                   for i in range(50)]),
        'minimally_integrated': lambda: np.random.randn(50, 100)
    }

    for scenario_name, generator in scenarios.items():
        neural_data = generator()
        phi = lab.calculate_consciousness_metric(neural_data)
        print(f"{scenario_name.replace('_', ' ').title()}:")
        print(f"  Phi (Φ) = {phi:.4f}")
        interpretation = 'High consciousness' if phi > 0.6 else \
                        'Moderate consciousness' if phi > 0.3 else \
                        'Low consciousness'
        print(f"  Interpretation: {interpretation}")

    # Demo 6: Network with different connectivity
    print("\n6. Connectivity Effects on Synchronization")
    print("-" * 70)

    connectivities = [0.05, 0.1, 0.2, 0.3]
    for conn in connectivities:
        result = lab.simulate_neural_network(100, connectivity=conn, duration_ms=500)
        print(f"Connectivity {conn*100:.0f}%: "
              f"Sync = {result['synchronization']:.3f}, "
              f"Firing rate = {result['avg_firing_rate']:.2f} Hz")

    print("\n" + "=" * 70)
    print("Demo complete!")


if __name__ == "__main__":
    main()
