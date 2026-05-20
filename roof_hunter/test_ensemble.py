import sys
import os

# Add the parent QuLabInfinite directory to the path so we can import roof_hunter modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.ml_models import ensemble_forecast, track_storm_cells
from integrations.radar_engine import NEXRADLevel2
from integrations.vulnerability_engine import calculate_roof_vulnerability
import pprint

def run_test():
    print("=== Testing Ensemble Forecast ===")
    target_location = {'lat': 35.33, 'lon': -97.27}  # Norman, OK area (near KTLX)
    
    # Let's test the ensemble forecast (which triggers HRRR and XGBoost)
    result = ensemble_forecast(target_location, horizon_hours=2)
    pprint.pprint(result)

    print("\n=== Testing Vulnerability Engine ===")
    property_lead = {'latitude': 35.33, 'longitude': -97.27, 'hail_probability': result.get('hail_probability', 0.5)}
    vuln_score = calculate_roof_vulnerability(property_lead)
    print(f"Property Lead Details: {property_lead}")
    print(f"Calculated Vulnerability Score (0-100): {vuln_score:.2f}")

    print("\n=== Testing Storm Cell Tracking (Dynamic Polygons) ===")
    cells = track_storm_cells("KTLX")
    if cells:
        print(f"Found {len(cells)} cells.")
        for c in cells[:3]: # print top 3
            print(f"Cell ID: {c.get('storm_cell_id')}, Max DBZ: {c.get('max_reflectivity_dbz')}")
            print(f"   Polygon coords count: {len(c.get('polygon_geojson', []))}")
    else:
        print("No storm cells returned (might be clear skies or missing data).")

if __name__ == "__main__":
    run_test()
