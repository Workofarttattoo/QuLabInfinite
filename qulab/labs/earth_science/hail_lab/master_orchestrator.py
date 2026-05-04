import json
import logging
from typing import Optional, Dict, Any

from .hail_physics import HailPhysicsEngine
from .nowcaster import HailNowcaster
from .satellite_fetcher import SatelliteFetcher
from .spatial_validator import SpatialValidator
from .trellis_bridge import TrellisBridge

logger = logging.getLogger(__name__)

class HailDigitalTwinOrchestrator:
    def __init__(self, google_api_key: str | None = None, model_path: Optional[str] = None):
        self.nowcaster = HailNowcaster(model_path=model_path)
        self.spatial = SpatialValidator()
        self.fetcher = SatelliteFetcher(google_api_key)
        self.bridge = TrellisBridge()
        self.physics = HailPhysicsEngine()

    async def run_pipeline(self,
                           current_flash_rate: float,
                           storm_lat: float,
                           storm_lon: float,
                           reflectivity_max: Optional[float] = None,
                           storm_features: Optional[Dict[str, Any]] = None,
                           velocity_kmh: float = 0.0,
                           heading_deg: float = 0.0) -> dict:
        """
        Runs the full Hail Digital Twin pipeline.
        Triggers on lightning jump OR high reflectivity (>50 dBZ).
        """
        results = {
            "lightning_jump": False,
            "high_reflectivity": False,
            "properties_at_risk": [],
            "pipeline_success": False,
            "damage_reports": []
        }

        # 1. Monitor Triggers
        # If reflectivity_max is in storm_features, use it
        if reflectivity_max is None and storm_features:
            reflectivity_max = storm_features.get("reflectivity_max", 0.0)

        effective_reflectivity = reflectivity_max or 0.0

        is_jump = self.nowcaster.detect_lightning_jump(current_flash_rate)
        is_high_ref = effective_reflectivity >= 50.0 # 50 dBZ threshold

        results["lightning_jump"] = is_jump
        results["high_reflectivity"] = is_high_ref

        if not (is_jump or is_high_ref):
            logger.info(f"No hail triggers detected (Ref={effective_reflectivity}, Jump={is_jump}). Pipeline standby.")
            return results

        # 2. Target Identification
        if velocity_kmh > 0:
            # Use path extrapolation if motion data is available
            at_risk = self.spatial.predict_storm_path(storm_lat, storm_lon, velocity_kmh, heading_deg)
            logger.info(f"Extrapolated storm path. Found {len(at_risk)} properties.")
        else:
            at_risk = self.spatial.find_at_risk_properties(storm_lat, storm_lon)
            logger.info(f"Checking vicinity of {storm_lat}, {storm_lon}. Found {len(at_risk)} properties.")

        results["properties_at_risk"] = at_risk

        if not at_risk:
            return results

        # Calculate probability once if we have features
        hail_prob = 0.0
        if storm_features:
            hail_prob = self.nowcaster.get_hail_probability(
                ship_index=storm_features.get("ship", 0.0),
                cape=storm_features.get("cape", 0.0),
                features=storm_features
            )
            results["predicted_hail_probability"] = hail_prob

        # 3. Process each property
        for prop in at_risk:
            report = {"property_id": prop["id"], "name": prop["name"], "success": False}

            # 3. Capture
            image_path = self.fetcher.fetch_roof_top(prop["lat"], prop["lon"])
            if not image_path:
                report["error"] = "Failed to fetch satellite image"
                results["damage_reports"].append(report)
                continue

            report["image_path"] = image_path

            # 4. Model Reconstruction
            glb_path = self.bridge.generate_3d_roof(image_path)
            if not glb_path:
                report["error"] = "Failed to generate 3D model"
                results["damage_reports"].append(report)
                continue

            report["glb_path"] = glb_path

            # 5. Physics Simulation
            # Determine hail size from features if available, otherwise default to 5cm
            hail_size = storm_features.get("hail_size_inches", 1.5) * 0.0254 if storm_features else 0.05

            sim_result = self.physics.simulate_strike_zone(glb_path, hail_diameter_m=hail_size)
            if sim_result["success"]:
                report["success"] = True
                report["damage_metrics"] = {
                    "ratio": sim_result["damage_ratio"],
                    "total_damaged": sim_result["total_damaged"],
                    "terminal_velocity": sim_result["terminal_velocity"]
                }
            else:
                report["error"] = f"Physics sim failed: {sim_result.get('error')}"

            results["damage_reports"].append(report)

        results["pipeline_success"] = any(r["success"] for r in results["damage_reports"])
        return results

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)

    async def demo():
        orchestrator = HailDigitalTwinOrchestrator(model_path="models/xgboost_hail.json")

        print("\n--- Scenario A: High Reflectivity Trigger (No lightning jump) ---")
        result_a = await orchestrator.run_pipeline(
            current_flash_rate=10,
            storm_lat=40.7128,
            storm_lon=-74.0060,
            reflectivity_max=55.0
        )
        print(f"Pipeline Success: {result_a['pipeline_success']}")
        print(f"Properties found: {len(result_a['properties_at_risk'])}")

        print("\n--- Scenario B: Predicted Path with ML Probabilities ---")
        # Prime the history for lightning jump
        for _ in range(5):
            orchestrator.nowcaster.detect_lightning_jump(10)

        storm_features = {
            "reflectivity_max": 60.0,
            "correlation_coefficient": 0.85,
            "cape": 3000.0,
            "hail_size_inches": 2.0
        }
        result_b = await orchestrator.run_pipeline(
            current_flash_rate=50, # Lightning jump!
            storm_lat=40.70,
            storm_lon=-74.03,
            velocity_kmh=30.0,
            heading_deg=45.0,
            storm_features=storm_features
        )
        if "predicted_hail_probability" in result_b:
            print(f"Predicted Hail Probability: {result_b['predicted_hail_probability']:.4f}")
        print(f"Pipeline Success: {result_b['pipeline_success']}")
        print(f"Lightning Jump: {result_b['lightning_jump']}")
        print(f"High Reflectivity: {result_b['high_reflectivity']}")

    asyncio.run(demo())
