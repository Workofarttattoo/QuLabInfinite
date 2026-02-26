import sys
from unittest.mock import MagicMock, patch

# Mock numpy before importing modules that depend on it
mock_np = MagicMock()
sys.modules["numpy"] = mock_np

import pytest

# Now we can import the module under test
from oncology_lab import (
    OncologyLaboratory,
    OncologyLabConfig,
    TumorType,
    CancerStage,
    TumorGrowthModel
)
from oncology_lab.ten_field_controller import create_ech0_three_stage_protocol

class TestOncologyLaboratory:

    @pytest.fixture
    def mock_tumor_simulator(self):
        with patch("oncology_lab.oncology_lab.TumorSimulator") as mock:
            yield mock

    @pytest.fixture
    def mock_field_controller(self):
        with patch("oncology_lab.oncology_lab.TenFieldController") as mock:
            instance = mock.return_value
            # Setup default behaviors for the mock
            instance.fields = {}
            instance.history = []
            instance.calculate_cancer_progression_score.return_value = 50.0
            instance.calculate_metabolic_stress.return_value = 0.5
            yield mock

    @pytest.fixture
    def mock_get_drug(self):
        with patch("oncology_lab.oncology_lab.get_drug_from_database") as mock:
            yield mock

    @pytest.fixture
    def mock_drug_simulator(self):
        with patch("oncology_lab.oncology_lab.DrugSimulator") as mock:
            yield mock

    def test_initialization(self, mock_tumor_simulator, mock_field_controller):
        """Test that the laboratory initializes correctly with config."""
        config = OncologyLabConfig(
            tumor_type=TumorType.LUNG_CANCER,
            stage=CancerStage.STAGE_III,
            initial_tumor_cells=500,
            growth_model=TumorGrowthModel.LOGISTIC
        )

        lab = OncologyLaboratory(config)

        # Verify TumorSimulator initialized with correct args
        mock_tumor_simulator.assert_called_once_with(
            tumor_type=TumorType.LUNG_CANCER.value,
            growth_model=TumorGrowthModel.LOGISTIC,
            initial_cells=500
        )

        # Verify TenFieldController initialized
        mock_field_controller.assert_called_once()

        # Verify field controller configuration
        instance = mock_field_controller.return_value
        instance.apply_cancer_profile.assert_called_once()
        instance.set_cancer_microenvironment.assert_called_once()

    def test_run_experiment(self, mock_tumor_simulator, mock_field_controller):
        """Test running an experiment loop."""
        lab = OncologyLaboratory()

        # Setup mocks for the loop
        lab.tumor.get_statistics.return_value = {
            'alive_cells': 1000,
            'tumor_volume_mm3': 1.5,
            'average_viability': 0.9,
            'total_cells': 1100,
            'dead_cells': 100
        }

        # Run for 24 hours with 1 hour step (default config might be 0.1h)
        lab.config.time_step_hours = 1.0
        lab.run_experiment(duration_days=1.0, report_interval_hours=24.0)

        # Should step 24 times
        assert lab.tumor.step.call_count == 24

        # Should verify interactions in step()
        lab.field_controller.step.assert_not_called() # No protocol active yet
        lab.tumor.apply_field_overrides.assert_called()

    def test_administer_drug(self, mock_tumor_simulator, mock_field_controller, mock_get_drug, mock_drug_simulator):
        """Test drug administration logic."""
        lab = OncologyLaboratory()

        # Setup mock drug
        mock_drug = MagicMock()
        mock_get_drug.return_value = mock_drug

        lab.administer_drug("cisplatin", 100.0)

        mock_get_drug.assert_called_once_with("cisplatin")
        mock_drug_simulator.assert_called_once_with(mock_drug)

        # Verify drug is in active_drugs
        assert "cisplatin" in lab.active_drugs
        lab.active_drugs["cisplatin"].administer_dose.assert_called_once_with(100.0, 0.0)

    def test_administer_invalid_drug(self, mock_get_drug, mock_tumor_simulator, mock_field_controller):
        """Test error handling for invalid drug."""
        lab = OncologyLaboratory()
        mock_get_drug.return_value = None

        with patch("oncology_lab.oncology_lab.list_available_drugs") as mock_list:
            mock_list.return_value = ["drug1", "drug2"]
            with pytest.raises(ValueError, match="Drug 'bad_drug' not found"):
                lab.administer_drug("bad_drug", 100.0)

    def test_apply_intervention_protocol(self, mock_tumor_simulator, mock_field_controller):
        """Test applying an intervention protocol."""
        lab = OncologyLaboratory()
        protocol = MagicMock()
        protocol.name = "Test Protocol"
        protocol.description = "Test Description"
        protocol.interventions = []

        lab.apply_intervention_protocol(protocol)

        assert lab.active_protocol == protocol

        # Run a step to ensure protocol is used
        lab.step(dt=1.0)
        lab.field_controller.step.assert_called_once_with(protocol, 1.0)

    def test_get_results(self, mock_tumor_simulator, mock_field_controller):
        """Test result aggregation."""
        lab = OncologyLaboratory()

        # Setup mock return values
        lab.tumor.get_statistics.return_value = {'stat': 1}

        # Mock history in field controller
        mock_history_item = MagicMock()
        mock_history_item.time_hours = 10.0
        mock_history_item.tumor_cell_count = 100
        mock_history_item.tumor_viability = 0.8
        mock_history_item.field_values = {'ph': 7.0}

        lab.field_controller.history = [mock_history_item]
        lab.field_controller.fields = {'ph': 7.0} # Needed for _calculate_score_from_fields

        results = lab.get_results()

        assert results['final_stats'] == {'stat': 1}
        assert results['time_hours'] == [10.0]
        assert results['cell_counts'] == [100]
        assert results['viabilities'] == [0.8]
        assert 'ph' in results['field_data']

    def test_create_ech0_protocol(self):
        """Test creation of the ECH0 protocol."""
        protocol = create_ech0_three_stage_protocol()
        assert protocol.name == "ECH0 Three-Stage Protocol"
        assert len(protocol.interventions) > 0
