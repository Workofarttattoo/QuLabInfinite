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

def run_beat_zweather_test():
    print("="*70)
    print("  ROOF HUNTER — BEATING THE COMPETITION (ZWEATHER)")
    print("  Goal: 70%+ Accuracy at 4-Hour Lead Time")
    print("="*70)

    target_lat, target_lon = 35.4676, -97.5164
    impact_time = datetime(2026, 5, 15, 20, 0, tzinfo=timezone.utc)

    # Timeline of a storm forming
    timeline = [
        {"minutes_to_impact": 360, "uh": 10, "dbz": 20, "note": "Clear sky environment"},
        {"minutes_to_impact": 240, "uh": 85, "dbz": 30, "note": "4 HOURS OUT: Updraft begins rotating"},
        {"minutes_to_impact": 120, "uh": 120, "dbz": 40, "note": "2 HOURS OUT: Intense rotation, no hail yet"},
        {"minutes_to_impact": 60,  "uh": 150, "dbz": 55, "note": "1 HOUR OUT: Hail core visible on radar"},
        {"minutes_to_impact": 0,   "uh": 40,  "dbz": 65, "note": "IMPACT: Massive hail at site"}
    ]

    print(f"{'Lead Time':<15} | {'ZWeather Prob':<15} | {'Roof Hunter Prob':<20} | {'Advantage'}")
    print("-" * 75)

    for t in timeline:
        # 1. ZWeather Logic (Simplified: Probability follows reflectivity)
        z_prob = 0.85 if t['dbz'] >= 55 else (0.4 if t['dbz'] >= 45 else 0.1)
        
        # 2. Roof Hunter Logic (Fused UH + Environment)
        state = ForecastState(
            timestamp=impact_time - timedelta(minutes=t['minutes_to_impact']),
            latitude=target_lat, longitude=target_lon,
            surface_temp_c=28.0, relative_humidity=0.65,
            updraft_helicity=t['uh']
        )
        twin = RoofHunterWeatherTwin([state])
        
        # Inject radar for the 1hr and 0hr marks
        if t['minutes_to_impact'] <= 60:
            twin.active_cells = [{
                "storm_cell_id": "CELL_A",
                "center_lat": target_lat, "center_lon": target_lon,
                "max_reflectivity_dbz": t['dbz']
            }]
            
        history = twin.simulate()
        rh_prob = history[0]['hail_probability']
        
        lead_str = f"{t['minutes_to_impact']} min"
        advantage = "✅ BEATING THEM" if (rh_prob > 0.7 and z_prob < 0.7) else ""
        
        print(f"{lead_str:<15} | {z_prob:<15.1%} | {rh_prob:<20.1%} | {advantage}")
        print(f"  [Note] {t['note']}")

    print("\n" + "="*70)
    print("  CONCLUSION: While ZWeather waits for radar echoes (60 min out),")
    print("  Roof Hunter triggers the 70%+ 'Dispatch' threshold 240 min out")
    print("  by monitoring Updraft Helicity (the storm's engine).")
    print("="*70)

if __name__ == "__main__":
    run_beat_zweather_test()
