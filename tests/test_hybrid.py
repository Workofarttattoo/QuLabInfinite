"""
Tests for the Hybrid Predictor (quantile XGBoost + MC bootstrap RF + stacking).

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hail_model.hybrid_predictor import HailForecast, HybridPredictor, HybridTrainer
from hail_model.preprocess import load_config


@pytest.fixture(scope="module")
def config():
    return load_config(Path(__file__).resolve().parent.parent / "hail_model" / "config.yaml")


@pytest.fixture(scope="module")
def trained_dir(config):
    trainer = HybridTrainer(config)
    trainer.train_pipeline(synthetic=True, n_synthetic=1000, n_bootstrap=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.save(tmpdir)
        yield tmpdir


class TestHybridTrainer:
    def test_train_and_metrics(self, config):
        trainer = HybridTrainer(config)
        metrics = trainer.train_pipeline(synthetic=True, n_synthetic=800, n_bootstrap=3)
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "mean_ci_width_80" in metrics
        assert metrics["r2"] > 0.0

    def test_quantile_models_trained(self, config):
        trainer = HybridTrainer(config)
        trainer.train_pipeline(synthetic=True, n_synthetic=500, n_bootstrap=3)
        assert len(trainer.quantile_models) == 5
        assert 0.50 in trainer.quantile_models

    def test_save_and_load(self, config):
        trainer = HybridTrainer(config)
        trainer.train_pipeline(synthetic=True, n_synthetic=500, n_bootstrap=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save(tmpdir)
            loaded = HybridTrainer.load(tmpdir, config)
            assert len(loaded.quantile_models) == 5
            assert len(loaded.bootstrap_forests) == 3
            assert loaded.meta_model is not None


class TestHybridPredictor:
    def test_predict_storm(self, trained_dir, config):
        predictor = HybridPredictor(trained_dir, config)
        storm = {
            "reflectivity_max": 65.0, "reflectivity_mean": 58.0,
            "reflectivity_std": 8.0, "cape": 3200.0, "shear_0_6km": 55.0,
            "differential_reflectivity": 0.5, "correlation_coefficient": 0.89,
            "specific_differential_phase": 2.0, "vil": 50.0, "echo_top_km": 14.0,
            "storm_relative_helicity": 280.0, "temp_500mb": -18.0,
            "freezing_level_m": 3000.0,
            "latitude": 35.5, "longitude": -97.0,
        }
        forecast = predictor.predict(storm)
        assert isinstance(forecast, HailForecast)
        assert forecast.predicted_size_inches >= 0.0
        assert 0.0 <= forecast.confidence_score <= 1.0
        assert len(forecast.quantiles) == 5
        assert forecast.confidence_interval_95[0] <= forecast.confidence_interval_95[1]

    def test_predict_calm(self, trained_dir, config):
        predictor = HybridPredictor(trained_dir, config)
        calm = {
            "reflectivity_max": 12.0, "reflectivity_mean": 8.0,
            "cape": 300.0, "shear_0_6km": 8.0,
            "latitude": 37.0, "longitude": -122.0,
        }
        forecast = predictor.predict(calm)
        assert forecast.predicted_size_inches < 1.0
        assert forecast.alert_level.startswith("NONE") or forecast.alert_level.startswith("MINOR")
        assert forecast.hail_predicted is False

    def test_quantile_ordering(self, trained_dir, config):
        predictor = HybridPredictor(trained_dir, config)
        storm = {
            "reflectivity_max": 60.0, "cape": 2500.0,
            "latitude": 35.0, "longitude": -97.0,
        }
        forecast = predictor.predict(storm)
        q = forecast.quantiles
        assert q[0.10] <= q[0.50] <= q[0.90]

    def test_to_dict(self, trained_dir, config):
        predictor = HybridPredictor(trained_dir, config)
        forecast = predictor.predict({
            "reflectivity_max": 55.0, "latitude": 35.0, "longitude": -97.0,
        })
        d = forecast.to_dict()
        assert "predicted_size_inches" in d
        assert "confidence_interval_95" in d
        assert "quantiles" in d

    def test_predict_batch(self, trained_dir, config):
        predictor = HybridPredictor(trained_dir, config)
        records = [
            {"reflectivity_max": 65.0, "cape": 3200.0, "latitude": 35.5, "longitude": -97.0},
            {"reflectivity_max": 12.0, "cape": 300.0, "latitude": 37.0, "longitude": -122.0},
        ]
        results = predictor.predict_batch(records)
        assert len(results) == 2
        assert results[0].predicted_size_inches > results[1].predicted_size_inches
