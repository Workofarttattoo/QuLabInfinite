import json

from .hail_physics import HailPhysicsEngine
from .nowcaster import HailNowcaster
from .satellite_fetcher import SatelliteFetcher
from .spatial_validator import SpatialValidator
from .trellis_bridge import TrellisBridge


class HailDigitalTwinOrchestrator:
    def __init__(self, google_api_key: str | None = None):
        self.nowcaster = HailNowcaster()
        self.spatial = SpatialValidator()
        self.fetcher = SatelliteFetcher(google_api_key)
        self.bridge = TrellisBridge()
        self.physics = HailPhysicsEngine()

    async def run_pipeline(self, current_flash_rate: float, storm_lat: float, storm_lon: float) -> dict:
        """
        Runs the full Hail Digital Twin pipeline.
        1. Monitor: Detect lightning jump
        2. Target: Identify at-risk property
        3. Capture: Fetch satellite imagery
        4. Model: Generate 3D reconstruction
        5. Sim: Run Monte Carlo physics strikes
        """
        results = {
            "lightning_jump": False,
            "properties_at_risk": [],
            "pipeline_success": False,
            "damage_reports": []
        }

        # 1. Monitor
        if not self.nowcaster.detect_lightning_jump(current_flash_rate):
            return results

        results["lightning_jump"] = True

        # 2. Target
        at_risk = self.spatial.find_at_risk_properties(storm_lat, storm_lon)
        results["properties_at_risk"] = at_risk

        if not at_risk:
            return results

        # 3. Process each property
        for prop in at_risk:
            report = {"property_id": prop["id"], "success": False}

            # 3. Capture
            image_path = self.fetcher.fetch_roof_top(prop["lat"], prop["lon"])
            if not image_path:
                report["error"] = "Failed to fetch satellite image"
                results["damage_reports"].append(report)
                continue

            report["image_path"] = image_path

            # 4. Model
            # In a real scenario, this would call the TRELLIS API
            # For the demo, we might need to mock this if the Space is slow/restricted
            glb_path = self.bridge.generate_3d_roof(image_path)
            if not glb_path:
                report["error"] = "Failed to generate 3D model"
                results["damage_reports"].append(report)
                continue

            report["glb_path"] = glb_path

            # 5. Final Sim
            sim_result = self.physics.simulate_strike_zone(glb_path)
            if sim_result["success"]:
                report["success"] = True
                report["damage_metrics"] = {
                    "ratio": sim_result["damage_ratio"],
                    "total_damaged": sim_result["total_damaged"]
                }
            else:
                report["error"] = f"Physics sim failed: {sim_result.get('error')}"

            results["damage_reports"].append(report)

        results["pipeline_success"] = any(r["success"] for r in results["damage_reports"])
        return results

if __name__ == "__main__":
    import asyncio

    async def demo():
        orchestrator = HailDigitalTwinOrchestrator()
        # Simulate lightning jump
        for rate in [10, 12, 11, 10, 12]:
            orchestrator.nowcaster.detect_lightning_jump(rate)

        print("Running pipeline for lightning jump at 40.7128, -74.0060...")
        # Current rate = 50 (huge jump)
        result = await orchestrator.run_pipeline(50, 40.7128, -74.0060)
        print(json.dumps(result, indent=2))

    asyncio.run(demo())
