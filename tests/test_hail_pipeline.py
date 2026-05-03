import pytest
import numpy as np
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

def test_spatial_validator():
    validator = SpatialValidator()
    # Centered on prop_001 with VERY small radius
    results = validator.find_at_risk_properties(40.7128, -74.0060, radius_km=0.01)
    assert len(results) == 1
    assert results[0]["id"] == "prop_001"

def test_hail_physics_terminal_velocity():
    engine = HailPhysicsEngine()
    v = engine.calculate_terminal_velocity(0.05) # 50mm
    # v = 9 * sqrt(50) approx 63.6
    assert 63.0 < v < 64.0

@pytest.mark.asyncio
async def test_orchestrator_pipeline_flow():
    with patch("qulab.labs.earth_science.hail_lab.satellite_fetcher.SatelliteFetcher.fetch_roof_top") as mock_fetch,          patch("qulab.labs.earth_science.hail_lab.trellis_bridge.TrellisBridge.generate_3d_roof") as mock_bridge,          patch("qulab.labs.earth_science.hail_lab.hail_physics.HailPhysicsEngine.simulate_strike_zone") as mock_physics:

        mock_fetch.return_value = "mock_image.png"
        mock_bridge.return_value = "mock_model.glb"
        mock_physics.return_value = {
            "success": True,
            "damage_ratio": 0.25,
            "total_damaged": 25
        }

        orchestrator = HailDigitalTwinOrchestrator(google_api_key="fake_key")

        # Setup history
        for _ in range(5):
            orchestrator.nowcaster.detect_lightning_jump(10)

        # Run pipeline with a jump
        result = await orchestrator.run_pipeline(50, 40.7128, -74.0060)

        assert result["lightning_jump"] is True
        assert len(result["properties_at_risk"]) > 0
        assert result["pipeline_success"] is True
        assert result["damage_reports"][0]["success"] is True
        assert result["damage_reports"][0]["damage_metrics"]["ratio"] == 0.25

if __name__ == "__main__":
    pytest.main([__file__])
