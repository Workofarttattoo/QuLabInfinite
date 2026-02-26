import sys
import os
import pytest

# Add current directory to path
sys.path.append(os.getcwd())

def test_imports():
    """Verify that neuroscience_lab classes are importable."""
    import neuroscience_lab

    # Test core lab classes
    assert hasattr(neuroscience_lab, 'NeuroscienceLaboratory')
    assert hasattr(neuroscience_lab, 'NeuroscienceLab') # Alias
    assert hasattr(neuroscience_lab, 'NeuronState')

    # Test biophysics classes
    assert hasattr(neuroscience_lab, 'HodgkinHuxleyNeuron')
    assert hasattr(neuroscience_lab, 'LeakyIntegrateFireNeuron')
    assert hasattr(neuroscience_lab, 'AdaptiveExponentialNeuron')
    assert hasattr(neuroscience_lab, 'IzhikevichNeuron')
    assert hasattr(neuroscience_lab, 'SynapticPlasticity')
    assert hasattr(neuroscience_lab, 'NetworkConnectivity')
    assert hasattr(neuroscience_lab, 'NeurotransmitterDynamics')
    assert hasattr(neuroscience_lab, 'EEGSimulator')
    assert hasattr(neuroscience_lab, 'NeuralNetworkSimulator')

    # Test instantiation
    hh = neuroscience_lab.HodgkinHuxleyNeuron()
    assert hh is not None

    lab = neuroscience_lab.NeuroscienceLaboratory()
    assert lab is not None

if __name__ == "__main__":
    test_imports()
    print("Imports verified successfully.")
