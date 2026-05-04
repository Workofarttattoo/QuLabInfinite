"""
Tests for the XGBoost Hail Prediction Model.

Covers preprocessing, training, inference, validation, and the
integration with the existing hail_lab physics engine.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hail_model.predict import HailPredictor
from hail_model.preprocess import HailDataPreprocessor, load_config
from hail_model.train import HailModelTrainer
from hail_model.validate import validate_model


@pytest.fixture(scope="module")
def config():
    config_path = Path(__file__).resolve().parent.parent / "hail_model" / "config.yaml"
    return load_config(config_path)


@pytest.fixture(scope="module")
def preprocessor(config):
    return HailDataPreprocessor(config)


@pytest.fixture(scope="module")
def synthetic_data(preprocessor):
    return preprocessor.generate_synthetic_data(n_samples=500, hail_fraction=0.2)


# ------------------------------------------------------------------
# Preprocessing tests
# ------------------------------------------------------------------

class TestPreprocessor:
    def test_load_config(self, config):
        assert "xgboost" in config
        assert "data" in config
        assert config["data"]["target_column"] == "hail_occurred"

    def test_generate_synthetic_data(self, synthetic_data):
        assert len(synthetic_data) == 500
        assert "hail_occurred" in synthetic_data.columns
        assert "reflectivity_max" in synthetic_data.columns
        assert "latitude" in synthetic_data.columns
        hail_frac = synthetic_data["hail_occurred"].mean()
        assert 0.1 <= hail_frac <= 0.3

    def test_add_derived_features(self, preprocessor):
        df = pd.DataFrame({
            "latitude": [35.0],
            "longitude": [-97.0],
            "time": pd.to_datetime(["2024-06-15 14:00:00"]),
            "reflectivity": [55.0],
        })
        result = preprocessor.add_derived_features(df)
        assert "hour" in result.columns
        assert "month" in result.columns
        assert "reflectivity_max" in result.columns
        assert "cape" in result.columns
        assert result["hour"].iloc[0] == 14
        assert result["month"].iloc[0] == 6

    def test_clean_data_clips_outliers(self, preprocessor):
        df = pd.DataFrame({
            "latitude": [35.0],
            "longitude": [-97.0],
            "reflectivity_max": [120.0],
            "cape": [8000.0],
        })
        result = preprocessor.clean_data(df)
        assert result["reflectivity_max"].iloc[0] == 80.0
        assert result["cape"].iloc[0] == 6000.0

    def test_preprocess_dataframe(self, preprocessor, synthetic_data):
        processed = preprocessor.preprocess_dataframe(synthetic_data.copy())
        assert "hour" in processed.columns
        assert not processed["latitude"].isna().any()


# ------------------------------------------------------------------
# Training tests
# ------------------------------------------------------------------

class TestTrainer:
    def test_train_pipeline_synthetic(self, config):
        trainer = HailModelTrainer(config)
        metrics = trainer.train_pipeline(synthetic=True, n_synthetic=400)
        assert "accuracy" in metrics
        assert "roc_auc" in metrics
        assert metrics["roc_auc"] > 0.5
        assert trainer.model is not None

    def test_save_and_load_model(self, config):
        trainer = HailModelTrainer(config)
        trainer.train_pipeline(synthetic=True, n_synthetic=400)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = trainer.save_model(model_dir=tmpdir, model_name="test_model.json")
            assert path.exists()
            assert path.with_suffix(".meta.json").exists()

            trainer2 = HailModelTrainer(config)
            trainer2.load_model(path)
            assert trainer2.model is not None

    def test_feature_importance(self, config):
        trainer = HailModelTrainer(config)
        trainer.train_pipeline(synthetic=True, n_synthetic=400)
        importance = trainer.feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_prepare_features(self, config):
        trainer = HailModelTrainer(config)
        pp = HailDataPreprocessor(config)
        data = pp.generate_synthetic_data(100)
        data = pp.preprocess_dataframe(data)
        X, y = trainer.prepare_features(data)
        assert len(X) == len(y)
        assert X.shape[1] == len(config["data"]["feature_columns"])


# ------------------------------------------------------------------
# Prediction tests
# ------------------------------------------------------------------

class TestPredictor:
    @pytest.fixture(scope="class")
    def trained_model_path(self, config):
        trainer = HailModelTrainer(config)
        trainer.train_pipeline(synthetic=True, n_synthetic=400)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = trainer.save_model(model_dir=tmpdir, model_name="pred_test.json")
            yield path

    def test_predict_single_dict(self, trained_model_path, config):
        predictor = HailPredictor(model_path=trained_model_path, config=config)
        sample = {
            "reflectivity_max": 62.0,
            "reflectivity_mean": 55.0,
            "reflectivity_std": 8.0,
            "differential_reflectivity": 3.5,
            "correlation_coefficient": 0.91,
            "specific_differential_phase": 2.5,
            "velocity_max": 30.0,
            "velocity_mean": 15.0,
            "spectrum_width_mean": 9.0,
            "cape": 3200.0,
            "shear_0_6km": 55.0,
            "temp_500mb": -18.0,
            "freezing_level_m": 3000.0,
            "vil": 48.0,
            "echo_top_km": 14.0,
            "storm_relative_helicity": 280.0,
            "latitude": 35.5,
            "longitude": -97.0,
            "time": "2026-05-02T14:30:00",
        }
        proba = predictor.predict_proba(sample)
        assert 0.0 <= proba[0] <= 1.0

    def test_predict_full(self, trained_model_path, config):
        predictor = HailPredictor(model_path=trained_model_path, config=config)
        result = predictor.predict_full({
            "reflectivity_max": 62.0,
            "latitude": 35.5,
            "longitude": -97.0,
        })
        assert "hail_probability" in result
        assert "risk_level" in result
        assert result["risk_level"] in ("MINIMAL", "LOW", "MODERATE", "HIGH", "EXTREME")

    def test_predict_batch(self, trained_model_path, config):
        predictor = HailPredictor(model_path=trained_model_path, config=config)
        records = [
            {"reflectivity_max": 62.0, "latitude": 35.5, "longitude": -97.0, "cape": 3200},
            {"reflectivity_max": 25.0, "latitude": 40.0, "longitude": -90.0, "cape": 800},
        ]
        results = predictor.predict_batch(records)
        assert len(results) == 2
        assert all("hail_probability" in r for r in results)


# ------------------------------------------------------------------
# Validation tests
# ------------------------------------------------------------------

class TestValidation:
    def test_validate_with_synthetic(self, config):
        trainer = HailModelTrainer(config)
        trainer.train_pipeline(synthetic=True, n_synthetic=400)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = trainer.save_model(model_dir=tmpdir, model_name="val_test.json")
            report = validate_model(path, config=config)

        assert "metrics" in report
        assert report["metrics"]["roc_auc"] > 0.5
        assert "confusion_matrix" in report
        assert "feature_importance" in report
        assert "threshold_sweep" in report

    def test_threshold_sweep_monotonic_recall(self, config):
        trainer = HailModelTrainer(config)
        trainer.train_pipeline(synthetic=True, n_synthetic=600)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = trainer.save_model(model_dir=tmpdir, model_name="sweep_test.json")
            report = validate_model(path, config=config)

        sweeps = report["threshold_sweep"]
        recalls = [s["recall"] for s in sweeps]
        # Recall should generally decrease as threshold increases
        assert recalls[0] >= recalls[-1]


# ------------------------------------------------------------------
# Integration with existing hail_lab
# ------------------------------------------------------------------

class TestHailLabIntegration:
    def test_nowcaster_probability(self):
        from qulab.labs.earth_science.hail_lab.nowcaster import HailNowcaster
        nc = HailNowcaster()
        prob = nc.get_hail_probability(ship_index=2.0, cape=2500)
        assert 0.0 <= prob <= 1.0
        assert prob >= 0.5

    def test_physics_terminal_velocity(self):
        pytest.importorskip("trimesh")
        from qulab.labs.earth_science.hail_lab.hail_physics import HailPhysicsEngine
        engine = HailPhysicsEngine()
        v = engine.calculate_terminal_velocity(0.05)
        assert v > 0
        assert 30 < v < 40

    def test_physics_impact_force(self):
        pytest.importorskip("trimesh")
        from qulab.labs.earth_science.hail_lab.hail_physics import HailPhysicsEngine
        engine = HailPhysicsEngine()
        f = engine.calculate_impact_force(mass=0.06, velocity=60.0, theta_rad=0.0)
        assert f > 0
