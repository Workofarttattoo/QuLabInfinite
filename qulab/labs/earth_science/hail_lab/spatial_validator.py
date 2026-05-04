import numpy as np
from typing import List, Dict, Any

class SpatialValidator:
    """
    Identifies specific properties/roofs at risk within a target area.
    """
    def __init__(self):
        # Mock database of properties (expanded)
        self.properties = [
            {"id": "prop_001", "lat": 40.7128, "lon": -74.0060, "name": "Standard Residential"},
            {"id": "prop_002", "lat": 40.7130, "lon": -74.0065, "name": "Large Warehouse"},
            {"id": "prop_003", "lat": 40.7125, "lon": -74.0055, "name": "Suburban Home"},
            {"id": "prop_004", "lat": 40.7200, "lon": -73.9900, "name": "Northside Commercial"},
            {"id": "prop_005", "lat": 40.7000, "lon": -74.0200, "name": "Westside Waterfront"},
            {"id": "prop_006", "lat": 40.7500, "lon": -73.9500, "name": "Uptown Estate"}
        ]

    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculates the Haversine distance between two points on Earth in km.
        """
        R = 6371.0  # Earth radius in km

        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)

        a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return float(R * c)

    def find_at_risk_properties(self, storm_lat: float, storm_lon: float, radius_km: float = 2.0) -> List[Dict[str, Any]]:
        """
        Finds properties within the storm's impact radius using Haversine distance.
        """
        at_risk = []
        for prop in self.properties:
            dist = self.haversine_distance(prop["lat"], prop["lon"], storm_lat, storm_lon)
            if dist <= radius_km:
                at_risk.append(prop)
        return at_risk

    def predict_storm_path(self, current_lat: float, current_lon: float, velocity_kmh: float, heading_deg: float, duration_hours: float = 1.0) -> List[Dict[str, Any]]:
        """
        Extrapolates storm position and identifies properties along the predicted path.
        """
        # Convert heading to radians (0 deg is North, 90 is East)
        heading_rad = np.radians(heading_deg)

        # Calculate displacement
        distance = velocity_kmh * duration_hours

        # Approximate new coordinates (valid for small distances)
        # 1 degree lat approx 111km
        dlat = (distance * np.cos(heading_rad)) / 111.0
        # 1 degree lon depends on latitude
        dlon = (distance * np.sin(heading_rad)) / (111.0 * np.cos(np.radians(current_lat)))

        predicted_lat = current_lat + dlat
        predicted_lon = current_lon + dlon

        # Identify properties near the start, end, and middle of the path
        path_samples = 5
        affected_properties = []
        seen_ids = set()

        for i in range(path_samples + 1):
            fraction = i / path_samples
            lat = current_lat + fraction * dlat
            lon = current_lon + fraction * dlon

            # Use a slightly wider radius for path prediction (e.g., 3km)
            at_risk = self.find_at_risk_properties(lat, lon, radius_km=3.0)
            for p in at_risk:
                if p["id"] not in seen_ids:
                    affected_properties.append(p)
                    seen_ids.add(p["id"])

        return affected_properties

if __name__ == "__main__":
    validator = SpatialValidator()
    # Storm at 40.70, -74.03 moving NE (45 deg) at 40km/h
    at_risk = validator.predict_storm_path(40.70, -74.03, 40.0, 45.0, duration_hours=0.5)
    print(f"Properties in predicted path: {[p['name'] for p in at_risk]}")
