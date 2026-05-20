import csv
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

class TargetPackGenerator:
    """
    Revenue Sourcing Engine: Converts probabilistic weather signals into ranked property target packs.
    
    Ranked by:
    1. Hail Probability (Confidence of impact)
    2. Vulnerability Score (Age, Material, Tree Cover)
    3. Property Value (Estimated ticket size proxy)
    """
    
    def __init__(self, results_path: Path, output_dir: Path):
        self.results_path = results_path
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def generate_pack(self, min_hail_prob: float = 0.5, top_n: int = 100) -> str:
        """Generates a ranked CSV of high-value targets."""
        with open(self.results_path, 'r') as f:
            data = json.load(f)
            history = data.get('history', [])

        targets = []
        for state in history:
            if state.get('hail_probability', 0) >= min_hail_prob:
                # Merge probability with vulnerability to get 'Impact Score'
                impact_score = (state['hail_probability'] * 0.7) + (state.get('vulnerability_score', 0) / 100.0 * 0.3)
                
                # Mock property info (In prod, this matches the latitude/longitude to a tax parcel)
                property_info = state.get('property_metadata', {
                    'address': f"STORM_LAT_{state['latitude']}_LON_{state['longitude']}",
                    'owner_name': 'PENDING_LOOKUP',
                    'roof_material': 'asphalt',
                    'roof_age': 15
                })
                
                targets.append({
                    'timestamp': state['timestamp'],
                    'address': property_info.get('address'),
                    'owner': property_info.get('owner_name'),
                    'lat': state['latitude'],
                    'lon': state['longitude'],
                    'hail_prob': f"{state['hail_probability']:.1%}",
                    'vulnerability': f"{state.get('vulnerability_score', 0):.1f}",
                    'impact_score': impact_score,
                    'radar_note': state.get('note', ''),
                    'vulnerability_note': f"Material: {property_info.get('roof_material')}, Age: {property_info.get('roof_age')}yr"
                })

        # Sort by impact score descending
        targets.sort(key=lambda x: x['impact_score'], reverse=True)
        targets = targets[:top_n]

        # Write to CSV
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"target_pack_{timestamp_str}.csv"
        file_path = self.output_dir / filename
        
        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'impact_score', 'hail_prob', 'address', 'owner', 'lat', 'lon', 'vulnerability', 'vulnerability_note', 'radar_note']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for t in targets:
                # Format impact score for CSV
                t_out = t.copy()
                t_out['impact_score'] = f"{t['impact_score']:.3f}"
                writer.writerow(t_out)

        print(f"[REVENUE] Generated Target Pack with {len(targets)} high-value leads at {file_path}")
        return str(file_path)

if __name__ == "__main__":
    # Example usage for the debug panel
    results = Path("/Users/noone/QuLabInfinite/roof_hunter/roof_hunter_last_week_results.json")
    if results.exists():
        gen = TargetPackGenerator(results, Path("/Users/noone/QuLabInfinite/roof_hunter/target_packs"))
        gen.generate_pack(min_hail_prob=0.3)
    else:
        print("No results file found to generate pack from.")
