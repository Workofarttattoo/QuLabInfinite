"""
Tests for the Ensemble model (XGBoost + RF) with classification and regression.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hail_model.ensemble import EnsemblePredictor, EnsembleTrainer
from hail_model.preprocess import load_config


@pytest.fixture(scope="module")
def config():
    return load_config(Path(__file__).resolve().parent.parent / "hail_model" / "config.yaml")


# ------------------------------------------------------------------
# Classification ensemble
# ------------------------------------------------------------------

class TestEnsembleClassification:
    def test_train_stacking(self, config):
        trainer = EnsembleTrainer(config)
        metrics = trainer.train_pipeline(synthetic=True, n_synthetic=500, method="stacking")
        assert "accuracy" in metrics
        assert "roc_auc" in metrics
        assert metrics["roc_auc"] > 0.8

    def test_train_voting(self, config):
        trainer = EnsembleTrainer(config)
        metrics = trainer.train_pipeline(synthetic=True, n_synthetic=500, method="voting")
        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0.8

    def test_save_and_load(self, config):
        trainer = EnsembleTrainer(config)
        trainer.train_pipeline(synthetic=True, n_synthetic=500)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save(tmpdir)
            loaded = EnsembleTrainer.load(tmpdir, config)
            assert loaded.xgb_model is not None
            assert loaded.rf_model is not None
            assert loaded.is_classification is True

    def test_predictor_classify(self, config):
        trainer = EnsembleTrainer(config)
        trainer.train_pipeline(synthetic=True, n_synthetic=500)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save(tmpdir)
            predictor = EnsemblePredictor(tmpdir, config)
            storm = {
                "reflectivity_max": 65.0, "reflectivity_mean": 58.0,
                "cape": 3200.0, "shear_0_6km": 55.0,
                "latitude": 35.5, "longitude": -97.0,
            }
            result = predictor.predict_full(storm)
            assert "hail_probability" in result
            assert result["mode"] == "classification"
            assert 0.0 <= result["hail_probability"] <= 1.0

    def test_predictor_calm(self, config):
        trainer = EnsembleTrainer(config)
        trainer.train_pipeline(synthetic=True, n_synthetic=500)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save(tmpdir)
            predictor = EnsemblePredictor(tmpdir, config)
            calm = {
                "reflectivity_max": 12.0, "reflectivity_mean": 8.0,
                "cape": 300.0, "shear_0_6km": 8.0,
                "latitude": 37.0, "longitude": -122.0,
            }
            result = predictor.predict_full(calm)
            assert result["hail_predicted"] is False


# ------------------------------------------------------------------
# Regression ensemble
# ------------------------------------------------------------------

class TestEnsembleRegression:
    @pytest.fixture()
    def regression_config(self, config):
        cfg = dict(config)
        cfg["data"] = dict(config["data"])
        cfg["data"]["target_column"] = "hail_size_inches"
        return cfg

    def test_train_regression_stacking(self, regression_config):
        trainer = EnsembleTrainer(regression_config)
        metrics = trainer.train_pipeline(synthetic=True, n_synthetic=500, method="stacking")
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["r2"] > 0.0

    def test_train_regression_voting(self, regression_config):
        trainer = EnsembleTrainer(regression_config)
        metrics = trainer.train_pipeline(synthetic=True, n_synthetic=500, method="voting")
        assert "rmse" in metrics
        assert "mae" in metrics

    def test_predictor_regression(self, regression_config):
        trainer = EnsembleTrainer(regression_config)
        trainer.train_pipeline(synthetic=True, n_synthetic=500)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save(tmpdir)
            predictor = EnsemblePredictor(tmpdir, regression_config)
            storm = {
                "reflectivity_max": 65.0, "reflectivity_mean": 58.0,
                "cape": 3200.0, "shear_0_6km": 55.0,
                "latitude": 35.5, "longitude": -97.0,
            }
            result = predictor.predict_full(storm)
            assert result["mode"] == "regression"
            assert "predicted_size_inches" in result
            assert result["predicted_size_inches"] >= 0.0

    def test_regression_calm_is_small(self, regression_config):
        trainer = EnsembleTrainer(regression_config)
        trainer.train_pipeline(synthetic=True, n_synthetic=500)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save(tmpdir)
            predictor = EnsemblePredictor(tmpdir, regression_config)
            calm = {
                "reflectivity_max": 12.0, "reflectivity_mean": 8.0,
                "cape": 300.0, "shear_0_6km": 8.0,
                "latitude": 37.0, "longitude": -122.0,
            }
            result = predictor.predict_full(calm)
            assert result["predicted_size_inches"] < 1.0
