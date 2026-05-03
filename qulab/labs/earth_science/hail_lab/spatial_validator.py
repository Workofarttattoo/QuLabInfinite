
class SpatialValidator:
    """
    Identifies specific properties/roofs at risk within a target area.
    """
    def __init__(self):
        # Mock database of properties
        self.properties = [
            {"id": "prop_001", "lat": 40.7128, "lon": -74.0060, "name": "Standard Residential"},
            {"id": "prop_002", "lat": 40.7130, "lon": -74.0065, "name": "Large Warehouse"},
            {"id": "prop_003", "lat": 40.7125, "lon": -74.0055, "name": "Suburban Home"}
        ]

    def find_at_risk_properties(self, storm_lat: float, storm_lon: float, radius_km: float = 1.0) -> list[dict]:
        """
        Finds properties within the storm's impact radius.
        """
        at_risk = []
        for prop in self.properties:
            # Simplified distance calculation
            dist = ((prop["lat"] - storm_lat)**2 + (prop["lon"] - storm_lon)**2)**0.5 * 111 # ~km per deg
            if dist <= radius_km:
                at_risk.append(prop)
        return at_risk
