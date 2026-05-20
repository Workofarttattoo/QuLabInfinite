import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Fix path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roof_hunter.models import ForecastState
from roof_hunter.roof_hunter_digital_twin import RoofHunterWeatherTwin

def run_funnel_test():
    print("="*70)
    print("  ROOF HUNTER — THE FUNNEL OF CERTAINTY")
    print("  Testing 21-Day lead time vs Real-Time Snap")
    print("="*70)

    # Target: Oklahoma City in Peak May
    target_lat, target_lon = 35.4676, -97.5164
    target_date = datetime(2026, 5, 15, 20, 0, tzinfo=timezone.utc)

    horizons = [
        {"name": "21 Days Out (Pattern)", "days": 21, "has_radar": False, "has_hrrr": False},
        {"name": "7 Days Out (Ensemble)", "days": 7, "has_radar": False, "has_hrrr": False},
        {"name": "1 Day Out (Mesoscale)", "days": 1, "has_radar": False, "has_hrrr": True},
        {"name": "30 Min Out (Radar Lock)", "days": 0, "has_radar": True, "has_hrrr": True}
    ]

    for h in horizons:
        print(f"\n[HORIZON] {h['name']}")
        
        # 1. Prepare State
        # In a real forecast, environmental certainty increases as days decrease
        state = ForecastState(
            timestamp=target_date,
            latitude=target_lat, longitude=target_lon,
            surface_temp_c=28.0, relative_humidity=0.65,
            wind_speed_m_s=12.0 if h['has_hrrr'] else 5.0,
            precip_mm=0.5 if h['has_radar'] else 0.0
        )

        # 2. Configure Twin
        twin = RoofHunterWeatherTwin([state])
        
        # Inject Radar if Horizon = 0
        if h['has_radar']:
            twin.active_cells = [{
                "storm_cell_id": "CELL_ALPHA",
                "center_lat": target_lat + 0.01,
                "center_lon": target_lon - 0.01,
                "max_reflectivity_dbz": 65.0
            }]

        # 3. Simulate
        history = twin.simulate()
        result = history[0]
        
        print(f"  Pattern Baseline:  {twin.s2s_baseline:.1%}")
        print(f"  Final Hail Prob:   {result['hail_probability']:.1%}")
        
        if result['hail_probability'] > 0.8:
            status = "🔥 DISPATCH HIGH-PRIORITY"
        elif result['hail_probability'] > 0.4:
            status = "⚠️ MONITOR REGION"
        else:
            status = "⚪️ STANDBY"
            
        print(f"  Decision Status:   {status}")
        if "RADAR_LOCKED" in result.get('note', ''):
            print(f"  Note:              {result['note']}")

    print("\n" + "="*70)
    print("  CONCLUSION: We get to 'weeks ahead' by matching the pattern baseline.")
    print("  Even with NO clouds 21 days out, we have a 25% baseline risk in May.")
    print("  This allows 'Sales War-Room' prep before the storm even forms.")
    print("="*70)

if __name__ == "__main__":
    run_funnel_test()
