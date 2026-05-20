"""12-Month S2S Backtest and Tuning Engine for Roof Hunter."""

import sys
from pathlib import Path
from datetime import date, timedelta
import json
import logging

# Syncing paths to QuLabInfinite
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validate_last_week import (
    fetch_last_week_weather, 
    build_forecast_payload, 
    fetch_spc_reports, 
    match_reports_to_forecast
)
from roof_hunter.roof_hunter_digital_twin import RoofHunterWeatherTwin
from roof_hunter.models import ForecastState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BacktestEngine")

def run_yearly_backtest(lat: float, lon: float):
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=365)
    
    logger.info(f"Starting 12-Month Backtest for {lat}, {lon}")
    logger.info(f"Period: {start_date} to {end_date}")
    
    # We run in monthly batches to manage memory and API limits
    all_summaries = []
    current_start = start_date
    
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=30), end_date)
        logger.info(f"Processing Batch: {current_start} -> {current_end}")
        
        try:
            # 1. Fetch Historical Weather
            weather = fetch_last_week_weather(lat, lon, current_end, current_start)
            payload = build_forecast_payload(weather, lat, lon)
            
            # 2. Simulate Digital Twin
            twin = RoofHunterWeatherTwin([ForecastState.from_dict(item) for item in payload['forecast']])
            history = twin.simulate()
            
            # 3. Fetch Ground Truth (SPC Reports)
            reports = fetch_spc_reports(lat, lon, current_start, current_end, radius_km=50.0)
            
            # 4. Match and Score
            metrics = match_reports_to_forecast(history, reports)
            all_summaries.append(metrics['summary'])
            
            logger.info(f"Batch Result: {metrics['summary']['match_rate']*100}% Strike Match Rate")
            
        except Exception as e:
            logger.error(f"Error in batch {current_start}: {e}")
            
        current_start = current_end + timedelta(days=1)
    
    # Final Aggregation
    final_report = {
        "location": {"lat": lat, "lon": lon},
        "period": {"start": str(start_date), "end": str(end_date)},
        "batches": all_summaries,
        "aggregate": {
            "avg_precision": sum(s['precision'] for s in all_summaries) / len(all_summaries) if all_summaries else 0,
            "avg_recall": sum(s['recall'] for s in all_summaries) / len(all_summaries) if all_summaries else 0,
            "total_matches": sum(s['matched_report_count'] for s in all_summaries)
        }
    }
    
    report_path = Path("backtest_report_yearly.json")
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)
    
    print("\n" + "="*50)
    print("12-MONTH BACKTEST COMPLETE")
    print(f"Total Matches Identified: {final_report['aggregate']['total_matches']}")
    print(f"Final Precision: {final_report['aggregate']['avg_precision']*100}%")
    print(f"Final Recall: {final_report['aggregate']['avg_recall']*100}%")
    print("="*50)

if __name__ == "__main__":
    # Test against Moore, OK (Historical Hotspot)
    run_yearly_backtest(35.3394, -97.4867)
