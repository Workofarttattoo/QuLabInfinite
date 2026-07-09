import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Ensure we can import from the root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hail_model.roof_hunter_bridge import HailIntelligence
from hail_model.preprocess import HailDataPreprocessor, load_config
from hail_model.nexrad_fetcher import RadarObservation

logger = logging.getLogger(__name__)

def generate_2025_dataset(n_samples: int = 5000):
    """Generates a synthetic 2025 dataset for evaluation."""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)
    preprocessor = HailDataPreprocessor(config)

    # Generate data
    df = preprocessor.generate_synthetic_data(n_samples=n_samples)

    # Update years to 2025
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'].apply(lambda x: x.replace(year=2025))

    return df

def evaluate_accuracy_2025():
    """Runs Roof Hunter against 2025 data and evaluates performance."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model_path = os.path.join(os.path.dirname(__file__), "models", "xgboost_hail.json")
    if not os.path.exists(model_path):
        # Check alternative path
        alt_path = os.path.join("hail_model", "models", "xgboost_hail.json")
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            logger.error(f"Model not found at {model_path}. Please train it first.")
            return

    intel = HailIntelligence(model_path=model_path)
    data_2025 = generate_2025_dataset(10000)

    results = []
    logger.info("Evaluating 10000 samples from 2025...")

    for _, row in data_2025.iterrows():
        # Create a RadarObservation from the row
        obs = RadarObservation(
            latitude=row['latitude'],
            longitude=row['longitude'],
            time=row['time'].isoformat(),
            reflectivity_max=row['reflectivity_max'],
            reflectivity_mean=row['reflectivity_mean'],
            reflectivity_std=row['reflectivity_std'],
            velocity_max=row['velocity_max'],
            velocity_mean=row['velocity_mean'],
            spectrum_width_mean=row['spectrum_width_mean'],
            differential_reflectivity=row['differential_reflectivity'],
            correlation_coefficient=row['correlation_coefficient'],
            specific_differential_phase=row['specific_differential_phase'],
            vil=row['vil'],
            echo_top_km=row['echo_top_km']
        )

        # Assess property
        assessment = intel.assess_property(row['latitude'], row['longitude'], radar_obs=obs, include_alerts=False)

        results.append({
            "time": row['time'],
            "actual_hail": row['hail_occurred'],
            "predicted_hail": 1 if assessment.action == "QUALIFY" else 0,
            "probability": assessment.hail_probability,
            "action": assessment.action
        })

    eval_df = pd.DataFrame(results)

    # Calculate Metrics
    tp = ((eval_df['actual_hail'] == 1) & (eval_df['predicted_hail'] == 1)).sum()
    tn = ((eval_df['actual_hail'] == 0) & (eval_df['predicted_hail'] == 0)).sum()
    fp = ((eval_df['actual_hail'] == 0) & (eval_df['predicted_hail'] == 1)).sum()
    fn = ((eval_df['actual_hail'] == 1) & (eval_df['predicted_hail'] == 0)).sum()

    accuracy = (tp + tn) / len(eval_df)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n=== Roof Hunter 2025 Accuracy Report ===")
    print(f"Total Samples: {len(eval_df)}")
    print(f"Accuracy:      {accuracy:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")

    # 90-Day Lead Validation
    print("\n=== 90-Day Lead Persistence Evaluation ===")

    eval_df = eval_df.sort_values('time')
    qualifications = eval_df[eval_df['action'] == "QUALIFY"].copy()
    validated_leads = 0

    for idx, qual in qualifications.iterrows():
        start_time = qual['time']
        end_time = start_time + timedelta(days=90)

        hits_in_window = eval_df[(eval_df['time'] > start_time) & (eval_df['time'] <= end_time) & (eval_df['actual_hail'] == 1)]

        if not hits_in_window.empty:
            validated_leads += 1

    lead_accuracy_90d = validated_leads / len(qualifications) if len(qualifications) > 0 else 0
    print(f"Total Leads Qualified:      {len(qualifications)}")
    print(f"Leads Struck within 90 Days: {validated_leads}")
    print(f"90-Day Lead Success Rate:    {lead_accuracy_90d:.2%}")

def evaluate_outlook_2025():
    """Evaluates the 90-day seasonal outlook logic."""
    intel = HailIntelligence()

    locations = [
        {"name": "Oklahoma City, OK", "lat": 35.47, "lon": -97.52},
        {"name": "Dallas, TX", "lat": 32.78, "lon": -96.80},
        {"name": "Denver, CO", "lat": 39.74, "lon": -104.99},
        {"name": "New York, NY", "lat": 40.71, "lon": -74.01},
        {"name": "Miami, FL", "lat": 25.76, "lon": -80.19}
    ]

    print("\n=== 90-Day Seasonal Outlook Evaluation (Climatological) ===")
    for loc in locations:
        outlook = intel.predict_90d_outlook(loc['lat'], loc['lon'])
        print(f"Location: {loc['name']:<20} | Prob: {outlook['hail_probability_90d']:.2%} | Risk: {outlook['risk_level']}")

if __name__ == "__main__":
    evaluate_accuracy_2025()
    evaluate_outlook_2025()
