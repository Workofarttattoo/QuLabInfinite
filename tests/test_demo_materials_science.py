import pytest
from unittest.mock import patch, MagicMock
from quantum_lab.demo import demo_materials_science

@patch('builtins.print')
@patch('builtins.input')
@patch('quantum_lab.demo.QuantumLabSimulator')
def test_demo_materials_science_execution(mock_lab_class, mock_input, mock_print):
    """
    Test that demo_materials_science executes successfully without throwing exceptions
    and makes the expected calls to calculate band gaps for specific materials.
    """
    # Create a mock instance of the lab simulator
    mock_lab_instance = MagicMock()
    mock_lab_class.return_value = mock_lab_instance

    # Mock the return values for the materials methods to prevent any actual computation or potential errors
    mock_lab_instance.materials.compute_band_gap.return_value = 1.0
    mock_lab_instance.materials.bcs_critical_temperature.return_value = 1.0
    mock_lab_instance.materials.superconducting_gap.return_value = 1.0
    mock_lab_instance.materials.topological_z2_invariant.return_value = 1
    mock_lab_instance.materials.quantum_phase_transition.return_value = {
        'phase': 'Phase',
        'order_parameter': 1.0,
        'at_critical_point': False
    }

    # Execute the demo
    demo_materials_science()

    # Verify the input function was called to pause at the end
    mock_input.assert_called_once()

    # Verify that the correct band gap calculations were made for the requested materials
    materials_expected = ['silicon', 'germanium', 'gallium_arsenide', 'graphene']

    calls = mock_lab_instance.materials.compute_band_gap.call_args_list
    called_materials = [call[0][0] for call in calls]

    for mat in materials_expected:
        assert mat in called_materials, f"Band gap for {mat} was not calculated in demo_materials_science"

    # Also verify some of the other physics checks
    mock_lab_instance.materials.bcs_critical_temperature.assert_any_call('aluminum')
    mock_lab_instance.materials.topological_z2_invariant.assert_any_call('bismuth_telluride')
    mock_lab_instance.materials.quantum_phase_transition.assert_called()

def test_demo_materials_science_integration():
    """
    Test demo_materials_science against the real QuantumLabSimulator
    to ensure the actual band gap calculations and properties work together.
    """
    with patch('builtins.print'), patch('builtins.input'):
        # Just run the real function - we're mostly checking it doesn't crash
        # and has basic test coverage
        demo_materials_science()
