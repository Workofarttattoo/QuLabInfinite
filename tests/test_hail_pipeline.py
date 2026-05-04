import pytest
import numpy as np
import os
from unittest.mock import MagicMock, patch
from qulab.labs.earth_science.hail_lab.nowcaster import HailNowcaster
from qulab.labs.earth_science.hail_lab.spatial_validator import SpatialValidator
from qulab.labs.earth_science.hail_lab.hail_physics import HailPhysicsEngine
from qulab.labs.earth_science.hail_lab.master_orchestrator import HailDigitalTwinOrchestrator

def test_lightning_jump():
    nowcaster = HailNowcaster(flash_rate_threshold=2.0)
    # Build history
    for rate in [10, 10, 10, 10, 10]:
        nowcaster.detect_lightning_jump(rate)

    assert nowcaster.detect_lightning_jump(20) is True
    assert nowcaster.detect_lightning_jump(10) is False

def test_spatial_validator_haversine():
    validator = SpatialValidator()
    # Centered on prop_001 with small radius
    results = validator.find_at_risk_properties(40.7128, -74.0060, radius_km=0.1)
    assert len(results) >= 1
    assert any(p["id"] == "prop_001" for p in results)

def test_spatial_validator_path_extrapolation():
    validator = SpatialValidator()
    # Storm moving NE (45 deg) right through the cluster
    at_risk = validator.predict_storm_path(40.71, -74.01, 20.0, 45.0, duration_hours=0.5)
    assert len(at_risk) > 0
    # Should find properties in the main cluster
    assert any(p["id"] in ["prop_001", "prop_002", "prop_003"] for p in at_risk)

def test_hail_physics_terminal_velocity():
    engine = HailPhysicsEngine()
    v = engine.calculate_terminal_velocity(0.05) # 50mm
    # Physics formula: v_t = sqrt((4 * 917 * 9.81 * 0.05) / (3 * 0.45 * 1.225))
    # v_t approx 32.98
    assert 32.0 < v < 34.0

@pytest.mark.asyncio
async def test_orchestrator_reflectivity_trigger():
    with patch("qulab.labs.earth_science.hail_lab.satellite_fetcher.SatelliteFetcher.fetch_roof_top") as mock_fetch,          patch("qulab.labs.earth_science.hail_lab.trellis_bridge.TrellisBridge.generate_3d_roof") as mock_bridge,          patch("qulab.labs.earth_science.hail_lab.hail_physics.HailPhysicsEngine.simulate_strike_zone") as mock_physics:

        mock_fetch.return_value = "mock_image.png"
        mock_bridge.return_value = "mock_model.glb"
        mock_physics.return_value = {
            "success": True,
            "damage_ratio": 0.25,
            "total_damaged": 25,
            "terminal_velocity": 32.0
        }

        orchestrator = HailDigitalTwinOrchestrator(google_api_key="fake_key")

        # Run pipeline with high reflectivity but NO lightning jump
        result = await orchestrator.run_pipeline(
            current_flash_rate=10,
            storm_lat=40.7128,
            storm_lon=-74.0060,
            reflectivity_max=55.0
        )

        assert result["high_reflectivity"] is True
        assert result["lightning_jump"] is False
        assert len(result["properties_at_risk"]) > 0
        assert result["pipeline_success"] is True

@pytest.mark.asyncio
async def test_orchestrator_ml_probability():
    with patch("qulab.labs.earth_science.hail_lab.satellite_fetcher.SatelliteFetcher.fetch_roof_top") as mock_fetch,          patch("qulab.labs.earth_science.hail_lab.trellis_bridge.TrellisBridge.generate_3d_roof") as mock_bridge,          patch("qulab.labs.earth_science.hail_lab.hail_physics.HailPhysicsEngine.simulate_strike_zone") as mock_physics:

        mock_fetch.return_value = "mock_image.png"
        mock_bridge.return_value = "mock_model.glb"
        mock_physics.return_value = {"success": True, "damage_ratio": 0.1, "total_damaged": 10, "terminal_velocity": 30.0}

        # Use the real model if it exists, or mock the predictor
        model_path = "models/xgboost_hail.json"
        orchestrator = HailDigitalTwinOrchestrator(model_path=model_path if os.path.exists(model_path) else None)

        storm_features = {
            "reflectivity_max": 60.0,
            "correlation_coefficient": 0.85,
            "cape": 3000.0
        }

        result = await orchestrator.run_pipeline(
            current_flash_rate=10,
            storm_lat=40.7128,
            storm_lon=-74.0060,
            reflectivity_max=55.0,
            storm_features=storm_features
        )

        assert "predicted_hail_probability" in result
        assert result["predicted_hail_probability"] > 0

if __name__ == "__main__":
    import asyncio
    pytest.main([__file__])
