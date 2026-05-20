import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add QuLabInfinite to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.ml_models import load_or_train_xgboost, predict_hail_xgboost

def generate_seasonal_profile(month: int, lat: float) -> dict:
    """Generate synthetic weather parameters based on seasonality and latitude."""
    # Peak severe weather season in US Plains is May (5) to July (7)
    seasonality = np.exp(-0.5 * ((month - 6) / 1.5) ** 2) # Gaussian peaking in June
    
    # Adjust for latitude (lower lat = longer season, higher lat = shifted season)
    lat_factor = 1.0 - abs(lat - 35.0) / 15.0 # Max at 35N (OKC area)

    # Base values
    temp = 10 + 20 * seasonality * lat_factor + np.random.normal(0, 5)
    cape = 500 + 3500 * seasonality * lat_factor + np.random.normal(0, 500)
    shear = 10 + 20 * seasonality + np.random.normal(0, 5)
    dbz = 20 + 45 * seasonality * lat_factor + np.random.normal(0, 10)
    
    return {
        'temp': max(-10, min(45, temp)), # Celsius
        'cape': max(0, cape),            # J/kg
        'shear': max(0, shear),          # m/s
        'dbz': max(0, min(75, dbz))      # dBZ
    }

def run_annual_simulation():
    model = load_or_train_xgboost()
    if not model:
        print("Could not load XGBoost model.")
        return

    cities = {
        "Oklahoma City, OK": {'lat': 35.46, 'lon': -97.51},
        "Dallas, TX": {'lat': 32.77, 'lon': -96.79},
        "Denver, CO": {'lat': 39.73, 'lon': -104.99},
        "Wichita, KS": {'lat': 37.68, 'lon': -97.33},
        "Minneapolis, MN": {'lat': 44.97, 'lon': -93.26}
    }

    results = []
    np.random.seed(42) # For reproducible simulations

    print("=== Running 1-Year Forward Storm Simulation ===")
    
    for city, coords in cities.items():
        for month in range(1, 13):
            # Simulate 10 storm events per month for this city
            for event in range(10):
                profile = generate_seasonal_profile(month, coords['lat'])
                prob = predict_hail_xgboost(profile['dbz'], profile['cape'], profile['shear'], profile['temp'])
                
                results.append({
                    'City': city,
                    'Month': month,
                    'Lat': coords['lat'],
                    'Lon': coords['lon'],
                    'Temp_C': round(profile['temp'], 1),
                    'CAPE': round(profile['cape'], 0),
                    'Shear': round(profile['shear'], 1),
                    'Max_DBZ': round(profile['dbz'], 1),
                    'Hail_Probability': prob
                })

    df = pd.DataFrame(results)
    
    # Aggregate to find the highest risk areas/months
    monthly_risk = df.groupby(['City', 'Month'])['Hail_Probability'].mean().reset_index()
    
    # Print the top 10 highest risk months/cities
    print("\n--- Top 10 Highest Risk Severe Hail Windows (1-Year Outlook) ---")
    top_risks = monthly_risk.sort_values(by='Hail_Probability', ascending=False).head(10)
    for idx, row in top_risks.iterrows():
        month_name = datetime(2026, int(row['Month']), 1).strftime('%B')
        print(f"{row['City']} - {month_name}: {row['Hail_Probability']:.1%} expected storm hail probability")

    # Overall city risk
    print("\n--- Annual Total Catastrophic Risk Ranking ---")
    annual_risk = df.groupby('City')['Hail_Probability'].mean().sort_values(ascending=False)
    for city, risk in annual_risk.items():
        print(f"{city}: {risk:.1%} annualized hail probability per storm")
        
    output_path = "/Users/noone/QuLabInfinite/roof_hunter/data/annual_simulation_results.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nDetailed simulation dataset (600 storm profiles) saved to {output_path}")

if __name__ == "__main__":
    run_annual_simulation()
