
import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add the project root to sys.path if it's not already there
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from api.qulab_api import QuLabSimulator, ExperimentRequest, ExperimentType, ExperimentResult

@pytest.fixture
def mock_materials_lab():
    with patch('api.qulab_api.MaterialsLab') as MockMaterialsLab:
        yield MockMaterialsLab.return_value

@pytest.fixture
def mock_environment():
    with patch('api.qulab_api.EnvironmentalSimulator') as MockEnv:
        yield MockEnv.return_value

@pytest.fixture
def mock_validator():
    with patch('api.qulab_api.ResultsValidator') as MockValidator:
        yield MockValidator.return_value

@pytest.fixture
def mock_physics_core():
    with patch('api.qulab_api.PhysicsCore') as MockPhysics:
        yield MockPhysics

@pytest.fixture
def simulator(mock_materials_lab, mock_environment, mock_validator):
    return QuLabSimulator()

class TestQuLabSimulator:

    def test_initialization(self, simulator, mock_materials_lab, mock_environment, mock_validator):
        assert simulator.materials_lab == mock_materials_lab
        assert simulator.environment == mock_environment
        assert simulator.validator == mock_validator
        assert simulator._experiment_counter == 0

    def test_run_unsupported_experiment_type(self, simulator):
        # Create a request with a type that is not in the dispatch table
        # We need to hack the Enum or just pass something that isn't handled if the dispatch table was dynamic,
        # but since it's an Enum, we can try to pass a value that is in the Enum but maybe not in the dispatch table
        # if the dispatch table doesn't cover all cases.
        # Looking at the code: _dispatch_table covers all members of ExperimentType except potentially if one is added later.
        # But if we pass a random string it might fail type checking but let's see.
        # Actually, let's just mock the dispatch table or add a fake enum member if possible,
        # or rely on the fact that the type hint says ExperimentType but python doesn't enforce it at runtime.

        # Let's try passing a dummy object that looks like an enum or just a random value if the code handles it.
        # The code does: handler = self._dispatch_table.get(request.experiment_type)
        # So if we pass something not in the dict, it returns None.

        request = ExperimentRequest(
            experiment_type="INVALID_TYPE",
            description="test",
            parameters={}
        )
        result = simulator.run(request)
        assert result.success is False
        assert "Unsupported experiment type" in result.error_message

    def test_run_material_test_tensile_success(self, simulator, mock_materials_lab, mock_validator):
        # Setup mocks
        mock_materials_lab.get_material.return_value = "SS 304"
        mock_materials_lab.tensile_test.return_value.data = {
            "strain": [0.0, 0.1],
            "stress": [0.0, 100.0],
            "yield_strength": 200.0,
            "ultimate_strength": 300.0,
            "youngs_modulus": 1000.0,
            "elongation_at_break": 10.0,
            "toughness": 50.0
        }

        mock_validator.validate.return_value.status = "pass" # This is an enum usually, need to check
        mock_validator.validate.return_value.error_percent = 0.1
        mock_validator.validate.return_value.z_score = 0.5
        mock_validator.validate.return_value.simulated_value = 200.0
        mock_validator.validate.return_value.reference_value = 200.0
        mock_validator.validate.return_value.uncertainty = 1.0
        mock_validator.validate.return_value.message = "OK"
        mock_validator.validate.return_value.passed_tests = []
        mock_validator.validate.return_value.failed_tests = []

        request = ExperimentRequest(
            experiment_type=ExperimentType.MATERIAL_TEST,
            description="Tensile test",
            parameters={"material": "SS 304", "temperature_c": 25.0, "test_type": "tensile"}
        )

        result = simulator.run(request)

        assert result.success is True
        assert result.data["yield_strength_MPa"] == 200.0
        mock_materials_lab.tensile_test.assert_called_with("SS 304", temperature=298.15)
        mock_validator.validate.assert_called()

    def test_run_material_test_missing_material(self, simulator):
        request = ExperimentRequest(
            experiment_type=ExperimentType.MATERIAL_TEST,
            description="Tensile test",
            parameters={"temperature_c": 25.0} # Missing material
        )
        result = simulator.run(request)
        assert result.success is False
        assert "Parameter 'material' is required" in result.error_message

    def test_run_material_test_not_found(self, simulator, mock_materials_lab):
        mock_materials_lab.get_material.return_value = None

        request = ExperimentRequest(
            experiment_type=ExperimentType.MATERIAL_TEST,
            description="Tensile test",
            parameters={"material": "Unobtainium"}
        )
        result = simulator.run(request)
        assert result.success is False
        assert "not found in database" in result.error_message

    def test_run_material_test_invalid_type(self, simulator, mock_materials_lab):
        mock_materials_lab.get_material.return_value = "SS 304"

        request = ExperimentRequest(
            experiment_type=ExperimentType.MATERIAL_TEST,
            description="Tensile test",
            parameters={"material": "SS 304", "test_type": "compression"}
        )
        result = simulator.run(request)
        assert result.success is False
        assert "Only tensile testing is wired up" in result.error_message

    def test_run_environment_analysis(self, simulator, mock_environment):
        # Mock the controller chain
        mock_controller = mock_environment.controller

        # Mock get_conditions_at_position return value
        mock_controller.get_conditions_at_position.return_value = {
            "temperature_C": 25.0,
            "temperature_K": 298.15,
            "pressure_bar": 1.0,
            "pressure_Pa": 100000.0,
            "wind_velocity_m_s": [0.0, 0.0, 0.0],
            "gravity_m_s2": [0.0, 0.0, -9.81]
        }

        request = ExperimentRequest(
            experiment_type=ExperimentType.ENVIRONMENT_ANALYSIS,
            description="Env analysis",
            parameters={
                "temperature_c": 25.0,
                "pressure_bar": 1.0,
                "wind_mph": 10.0
            }
        )

        result = simulator.run(request)

        assert result.success is True
        assert result.data["temperature_C"] == 25.0

        # Check if environment controller methods were called
        mock_controller.temperature.set_temperature.assert_called_with(25.0, unit="C")
        mock_controller.pressure.set_pressure.assert_called_with(1.0, unit="bar")
        # 10 mph is approx 4.4704 m/s
        args, kwargs = mock_controller.fluid.set_wind.call_args
        assert abs(args[0][0] - 4.4704) < 0.001

    def test_run_physics_probe(self, simulator, mock_physics_core):
        # Setup mock physics core instance and its mechanics
        mock_core_instance = mock_physics_core.return_value
        mock_particle = MagicMock()
        mock_particle.position = [0.0, 0.0, 0.0]
        mock_particle.velocity = [0.0, 0.0, -10.0]

        mock_core_instance.mechanics.particles = [mock_particle]
        mock_core_instance.mechanics.kinetic_energy.return_value = 50.0
        mock_core_instance.mechanics.potential_energy.return_value = 0.0
        mock_core_instance.mechanics.energy_error.return_value = 0.0
        mock_core_instance.step_count = 100

        request = ExperimentRequest(
            experiment_type=ExperimentType.PHYSICS_PROBE,
            description="Drop test",
            parameters={"mass_kg": 1.0, "initial_height_m": 10.0, "duration_s": 2.0}
        )

        result = simulator.run(request)

        assert result.success is True
        assert result.data["mass_kg"] == 1.0
        assert result.data["final_velocity_m_s"] == -10.0
        mock_core_instance.simulate.assert_called_once()

    def test_run_integrated_stack(self, simulator, mock_materials_lab, mock_environment, mock_physics_core):
        # This test touches all three subsystems

        # 1. Environment Mocking
        mock_controller = mock_environment.controller
        mock_controller.get_conditions_at_position.return_value = {
            "temperature_C": 25.0, "temperature_K": 298.15,
            "pressure_bar": 1.0, "pressure_Pa": 100000.0,
            "wind_velocity_m_s": [0.0, 0.0, 0.0],
            "gravity_m_s2": [0.0, 0.0, -9.81]
        }

        # 2. Materials Mocking
        mock_materials_lab.get_material.return_value = "SS 304"
        mock_materials_lab.tensile_test.return_value.data = {
            "strain": [], "stress": [],
            "yield_strength": 200.0, "ultimate_strength": 300.0,
            "youngs_modulus": 1000.0, "elongation_at_break": 10.0, "toughness": 50.0
        }
        mock_materials_lab.get_material_safety.return_value = {"flammability": "low"}

        # 3. Physics Mocking
        mock_core_instance = mock_physics_core.return_value
        mock_particle = MagicMock()
        mock_particle.position = [0.0, 0.0, 0.0]
        mock_particle.velocity = [0.0, 0.0, -10.0]
        mock_core_instance.mechanics.particles = [mock_particle]
        mock_core_instance.mechanics.kinetic_energy.return_value = 50.0
        mock_core_instance.mechanics.potential_energy.return_value = 0.0
        mock_core_instance.mechanics.energy_error.return_value = 0.0
        mock_core_instance.step_count = 100

        request = ExperimentRequest(
            experiment_type=ExperimentType.INTEGRATED,
            description="Integrated test",
            parameters={
                "material": "SS 304",
                "temperature_c": 25.0,
                "pressure_bar": 1.0,
                "specimen_mass_kg": 0.5,
                "drop_height_m": 5.0
            }
        )

        result = simulator.run(request)

        assert result.success is True
        assert "environment" in result.data
        assert "material_test" in result.data
        assert "mechanics_probe" in result.data
        assert "safety" in result.data
        assert result.data["safety"]["flammability"] == "low"

    def test_run_safety_query(self, simulator, mock_materials_lab):
        mock_materials_lab.get_material_safety.return_value = {"hazard": "None"}

        request = ExperimentRequest(
            experiment_type=ExperimentType.SAFETY_QUERY,
            description="Safety check",
            parameters={"material": "Water"}
        )

        result = simulator.run(request)
        assert result.success is True
        assert result.data["available"] is True
        assert result.data["safety"] == {"hazard": "None"}

    def test_run_safety_query_missing_material(self, simulator):
        request = ExperimentRequest(
            experiment_type=ExperimentType.SAFETY_QUERY,
            description="Safety check",
            parameters={}
        )
        result = simulator.run(request)
        assert result.success is False
        assert "Parameter 'material' is required" in result.error_message

    def test_run_ice_analysis(self, simulator, mock_materials_lab):
        mock_materials_lab.simulate_ice_growth.return_value = {"thickness_mm": 1.0}

        request = ExperimentRequest(
            experiment_type=ExperimentType.ICE_ANALYSIS,
            description="Ice check",
            parameters={"material": "Wing", "temperature_c": -10.0}
        )

        result = simulator.run(request)
        assert result.success is True
        assert result.data["ice_analysis"] == {"thickness_mm": 1.0}

    def test_find_materials(self, simulator, mock_materials_lab):
        mock_material = MagicMock()
        mock_material.name = "Test Material"
        mock_materials_lab.search_materials.return_value = [mock_material]

        criteria = {"category": "Metal", "density_min": 5.0}
        results = simulator.find_materials(criteria)

        assert results == ["Test Material"]
        mock_materials_lab.search_materials.assert_called_with(category="Metal", min_density=5.0)

    def test_parse_text_request(self, simulator):
        # Test implicit parsing via run(str)

        # Safety query
        res1 = simulator.run("What is the safety of Water?")
        # It won't actually run successfully without mocks setup for specific calls or if we don't care about the result success but just that it dispatched correctly.
        # But run() calls dispatch.
        # Let's inspect the request object if we can, or check calls.
        # It's easier to unit test _parse_text_request directly if it was public, or trust the integration via run().
        # Since I'm testing the public API, I will check the result or mocking side effects.

        with patch.object(simulator, '_dispatch_table') as mock_dispatch:
            # Safety
            simulator.run("safety data for acetone")
            args, _ = mock_dispatch.get.call_args
            assert args[0] == ExperimentType.SAFETY_QUERY

            # Ice
            simulator.run("ice formation on wing")
            args, _ = mock_dispatch.get.call_args
            assert args[0] == ExperimentType.ICE_ANALYSIS

            # Material
            simulator.run("tensile strength of steel")
            args, _ = mock_dispatch.get.call_args
            assert args[0] == ExperimentType.MATERIAL_TEST

            # Environment
            simulator.run("simulate environment temperature")
            args, _ = mock_dispatch.get.call_args
            assert args[0] == ExperimentType.ENVIRONMENT_ANALYSIS

            # Integrated
            simulator.run("integrated test of steel")
            args, _ = mock_dispatch.get.call_args
            assert args[0] == ExperimentType.INTEGRATED

            # Physics (default)
            simulator.run("drop a ball")
            args, _ = mock_dispatch.get.call_args
            assert args[0] == ExperimentType.PHYSICS_PROBE

    def test_demo(self, simulator):
        # Test that demo runs an integrated experiment
        # We can mock run() to verify it calls it with expected params
        with patch.object(simulator, 'run') as mock_run:
            mock_run.return_value = ExperimentResult(
                experiment_id="test-id", success=True, data={}, validation=None, notes="test"
            )

            result = simulator.demo()

            assert "request" in result
            assert "result" in result
            mock_run.assert_called_once()
            args, _ = mock_run.call_args
            req = args[0]
            assert req.experiment_type == ExperimentType.INTEGRATED
            assert req.parameters["material"] == "SS 304"

    def test_simulate_environment_ice_error(self, simulator, mock_environment, mock_materials_lab):
        # Setup environment mock
        mock_controller = mock_environment.controller
        mock_controller.get_conditions_at_position.return_value = {
            "temperature_C": 0.0, "temperature_K": 273.15,
            "pressure_bar": 1.0, "pressure_Pa": 100000.0,
            "wind_velocity_m_s": [0.0, 0.0, 0.0],
            "gravity_m_s2": [0.0, 0.0, -9.81]
        }

        # Setup materials lab to raise exception during ice simulation
        mock_materials_lab.simulate_ice_growth.side_effect = Exception("Ice Error")

        params = {"material": "Wing", "temperature": 0.0}
        summary = simulator.simulate_environment(params)

        assert "ice_analysis_error" in summary
        assert summary["ice_analysis_error"] == "Ice Error"

    def test_handler_exception(self, simulator):
        # Mock dispatch table to return a handler that raises exception
        mock_handler = MagicMock(side_effect=ValueError("Handler error"))

        # We need to inject this handler into the dispatch table
        # Since _dispatch_table is initialized in __init__, we can modify it
        # However, _dispatch_table keys are Enum members.

        # We can just patch the handler for PHYSICS_PROBE in the existing table
        original_handler = simulator._dispatch_table[ExperimentType.PHYSICS_PROBE]
        simulator._dispatch_table[ExperimentType.PHYSICS_PROBE] = mock_handler

        try:
             request = ExperimentRequest(
                experiment_type=ExperimentType.PHYSICS_PROBE,
                description="test",
                parameters={}
             )
             result = simulator.run(request)

             assert result.success is False
             assert result.error_message == "Handler error"
        finally:
            # Restore
            simulator._dispatch_table[ExperimentType.PHYSICS_PROBE] = original_handler
