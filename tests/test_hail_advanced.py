"""
Tests for NEXRAD fetcher, Dual-Pol algorithms, and Roof Hunter bridge.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hail_model.dual_pol import (
    DualPolObservation,
    HydrometeorType,
    classify_hydrometeor,
    compute_mesh,
    compute_posh,
    estimate_hail_size,
)
from hail_model.nexrad_fetcher import (
    NEXRADFetcher,
    RadarObservation,
    nearest_station,
)
from hail_model.roof_hunter_bridge import HailIntelligence, PropertyAssessment

# ------------------------------------------------------------------
# NEXRAD Fetcher tests
# ------------------------------------------------------------------

class TestNEXRADFetcher:
    def test_nearest_station_oklahoma_city(self):
        station = nearest_station(35.47, -97.52)
        assert station == "KTLX"

    def test_nearest_station_wichita(self):
        station = nearest_station(37.69, -97.34)
        assert station == "KICT"

    def test_nearest_station_dallas(self):
        station = nearest_station(32.78, -96.80)
        assert station == "KFWS"

    def test_radar_observation_to_dict(self):
        obs = RadarObservation(
            latitude=35.0, longitude=-97.0, time="2026-05-01T12:00:00Z",
            station_id="KTLX", reflectivity_max=55.0,
        )
        d = obs.to_dict()
        assert d["latitude"] == 35.0
        assert d["station_id"] == "KTLX"
        assert d["reflectivity_max"] == 55.0
        assert "correlation_coefficient" in d

    def test_list_nexrad_files_returns_list(self):
        result = NEXRADFetcher.list_nexrad_files("KTLX")
        assert isinstance(result, list)

    def test_nexrad_download_url_format(self):
        from datetime import datetime
        url = NEXRADFetcher.nexrad_download_url("KTLX", datetime(2024, 6, 15), "file.gz")
        assert "2024/06/15/KTLX/file.gz" in url


# ------------------------------------------------------------------
# Dual-Pol HCA tests
# ------------------------------------------------------------------

class TestHydrometeorClassification:
    def test_giant_hail(self):
        obs = DualPolObservation(
            reflectivity_h=72.0, differential_reflectivity=0.3,
            correlation_coefficient=0.88, specific_differential_phase=0.5,
            temperature_c=-15.0,
        )
        assert classify_hydrometeor(obs) == HydrometeorType.GIANT_HAIL

    def test_large_hail(self):
        obs = DualPolObservation(
            reflectivity_h=63.0, differential_reflectivity=0.5,
            correlation_coefficient=0.90, specific_differential_phase=0.8,
            temperature_c=-12.0,
        )
        assert classify_hydrometeor(obs) == HydrometeorType.LARGE_HAIL

    def test_hail_rain(self):
        obs = DualPolObservation(
            reflectivity_h=55.0, differential_reflectivity=1.5,
            correlation_coefficient=0.93, specific_differential_phase=1.0,
            temperature_c=-8.0,
        )
        assert classify_hydrometeor(obs) == HydrometeorType.HAIL_RAIN

    def test_heavy_rain(self):
        obs = DualPolObservation(
            reflectivity_h=52.0, differential_reflectivity=3.0,
            correlation_coefficient=0.97, specific_differential_phase=2.5,
            temperature_c=10.0,
        )
        assert classify_hydrometeor(obs) == HydrometeorType.HEAVY_RAIN

    def test_light_rain(self):
        obs = DualPolObservation(
            reflectivity_h=25.0, differential_reflectivity=0.5,
            correlation_coefficient=0.99, specific_differential_phase=0.1,
            temperature_c=15.0,
        )
        assert classify_hydrometeor(obs) == HydrometeorType.LIGHT_RAIN

    def test_dry_snow(self):
        obs = DualPolObservation(
            reflectivity_h=20.0, differential_reflectivity=0.2,
            correlation_coefficient=0.99, specific_differential_phase=0.0,
            temperature_c=-15.0,
        )
        assert classify_hydrometeor(obs) == HydrometeorType.DRY_SNOW


# ------------------------------------------------------------------
# MESH / POSH tests
# ------------------------------------------------------------------

class TestMESH:
    def test_no_hail_below_freezing(self):
        profile_z = [30.0, 35.0, 30.0]
        heights = [2.0, 3.0, 4.0]
        mesh = compute_mesh(profile_z, heights, freezing_level_km=3.5)
        assert mesh == 0.0

    def test_severe_hail_profile(self):
        profile_z = [65.0, 68.0, 65.0, 60.0, 55.0, 50.0, 45.0]
        heights = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        mesh = compute_mesh(profile_z, heights, freezing_level_km=3.0)
        assert mesh > 0.0

    def test_mismatched_arrays_raises(self):
        with pytest.raises(ValueError):
            compute_mesh([50, 55], [3.0, 4.0, 5.0], 3.0)


class TestPOSH:
    def test_no_hail_weak_echo(self):
        posh = compute_posh(30.0, 8.0, 3.5)
        assert posh == 0.0

    def test_high_posh_strong_storm(self):
        posh = compute_posh(65.0, 14.0, 3.5)
        assert posh > 50.0

    def test_posh_clamped_to_100(self):
        posh = compute_posh(75.0, 18.0, 2.0)
        assert 0.0 <= posh <= 100.0


# ------------------------------------------------------------------
# Full hail estimation
# ------------------------------------------------------------------

class TestHailEstimation:
    def test_giant_hail_detected(self):
        obs = DualPolObservation(
            reflectivity_h=72.0, differential_reflectivity=0.2,
            correlation_coefficient=0.87, specific_differential_phase=0.5,
            temperature_c=-15.0,
        )
        est = estimate_hail_size(obs)
        assert est.hail_detected is True
        assert est.hydrometeor_class == HydrometeorType.GIANT_HAIL
        assert est.estimated_diameter_inches > 1.0
        assert est.confidence >= 0.9

    def test_no_hail_light_rain(self):
        obs = DualPolObservation(
            reflectivity_h=25.0, differential_reflectivity=0.5,
            correlation_coefficient=0.99, specific_differential_phase=0.1,
            temperature_c=15.0,
        )
        est = estimate_hail_size(obs)
        assert est.hail_detected is False
        assert est.estimated_diameter_inches == 0.0


# ------------------------------------------------------------------
# Roof Hunter Bridge tests
# ------------------------------------------------------------------

class TestHailIntelligence:
    @pytest.fixture(scope="class")
    def trained_model_path(self):
        from hail_model.train import HailModelTrainer
        trainer = HailModelTrainer()
        trainer.train_pipeline(synthetic=True, n_synthetic=400)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = trainer.save_model(model_dir=tmpdir, model_name="bridge_test.json")
            yield path

    def test_assess_with_model(self, trained_model_path):
        intel = HailIntelligence(
            model_path=str(trained_model_path),
            qualify_threshold=0.5,
        )
        storm_obs = RadarObservation(
            latitude=35.5, longitude=-97.0,
            time="2026-05-01T14:00:00Z", station_id="KTLX",
            reflectivity_max=65.0, reflectivity_mean=58.0,
            reflectivity_std=8.0, velocity_max=30.0,
            velocity_mean=15.0, spectrum_width_mean=9.0,
            differential_reflectivity=0.5,
            correlation_coefficient=0.89,
            specific_differential_phase=2.0,
            vil=50.0, echo_top_km=14.0,
        )
        result = intel.assess_property(35.5, -97.0, radar_obs=storm_obs, include_alerts=False)
        assert isinstance(result, PropertyAssessment)
        assert 0.0 <= result.hail_probability <= 1.0
        assert result.action in ("QUALIFY", "MONITOR", "SKIP")
        assert result.radar_station == "KTLX"

    def test_assess_without_model(self):
        intel = HailIntelligence(model_path=None)
        storm_obs = RadarObservation(
            latitude=35.5, longitude=-97.0,
            time="2026-05-01T14:00:00Z", station_id="KTLX",
            reflectivity_max=65.0, reflectivity_mean=58.0,
            differential_reflectivity=0.3,
            correlation_coefficient=0.88,
            specific_differential_phase=1.0,
        )
        result = intel.assess_property(35.5, -97.0, radar_obs=storm_obs, include_alerts=False)
        assert result.hail_predicted is True
        assert result.hydrometeor_class in ("LH", "GH", "HA")

    def test_assess_calm_conditions(self, trained_model_path):
        intel = HailIntelligence(model_path=str(trained_model_path))
        calm_obs = RadarObservation(
            latitude=40.0, longitude=-90.0,
            time="2026-05-01T14:00:00Z", station_id="KOAX",
            reflectivity_max=15.0, reflectivity_mean=10.0,
            differential_reflectivity=0.5,
            correlation_coefficient=0.99,
            specific_differential_phase=0.1,
        )
        result = intel.assess_property(40.0, -90.0, radar_obs=calm_obs, include_alerts=False)
        assert result.action == "SKIP"
        assert result.hail_predicted is False

    def test_assess_batch(self, trained_model_path):
        intel = HailIntelligence(model_path=str(trained_model_path))
        properties = [
            {"lat": 35.5, "lon": -97.0},
            {"lat": 40.0, "lon": -90.0},
        ]
        results = intel.assess_batch(properties, include_alerts=False)
        assert len(results) == 2
        assert results[0].hail_probability >= results[1].hail_probability

    def test_property_assessment_to_dict(self, trained_model_path):
        intel = HailIntelligence(model_path=str(trained_model_path))
        obs = RadarObservation(
            latitude=35.5, longitude=-97.0,
            time="2026-05-01T14:00:00Z", station_id="KTLX",
            reflectivity_max=55.0, differential_reflectivity=1.0,
            correlation_coefficient=0.95,
        )
        result = intel.assess_property(35.5, -97.0, radar_obs=obs, include_alerts=False)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "hail_probability" in d
        assert "action" in d
        assert "estimated_hail_size_inches" in d
