"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

QuLabInfinite Neuroscience Laboratory
"""

from .neuroscience_lab import (
    NeuroscienceLaboratory,
    NeuronState,
    SynapticConnection,
    BrainActivity,
    NeuronType,
    Neurotransmitter,
    BrainRegion
)

from .biophysics import (
    IonChannel,
    HodgkinHuxleyNeuron,
    LeakyIntegrateFireNeuron,
    AdaptiveExponentialNeuron,
    IzhikevichNeuron,
    SynapticPlasticity,
    NetworkConnectivity,
    NeurotransmitterDynamics,
    EEGSimulator,
    NeuralNetworkSimulator
)

# Alias for compatibility
NeuroscienceLab = NeuroscienceLaboratory

__all__ = [
    'NeuroscienceLaboratory',
    'NeuroscienceLab',
    'NeuronState',
    'SynapticConnection',
    'BrainActivity',
    'NeuronType',
    'Neurotransmitter',
    'BrainRegion',
    'IonChannel',
    'HodgkinHuxleyNeuron',
    'LeakyIntegrateFireNeuron',
    'AdaptiveExponentialNeuron',
    'IzhikevichNeuron',
    'SynapticPlasticity',
    'NetworkConnectivity',
    'NeurotransmitterDynamics',
    'EEGSimulator',
    'NeuralNetworkSimulator'
]
