"""
Historical Backtest Engine v2
-----------------------------
1. Pulls 765 REAL confirmed hail events from NOAA Storm Events (2024 Oklahoma)
2. Generates negative samples (clear days at same locations)
3. Fetches REAL weather from Open-Meteo archive for ALL samples
4. Retrains XGBoost on features Open-Meteo ACTUALLY provides:
   - temperature_2m, windspeed_10m, windspeed_100m, precipitation, 
     surface_pressure, cloudcover
5. 80/20 train/test split — backtests on held-out REAL events
6. Reports precision, recall, F1 with exact dates/times/hours
"""
import os
import sys
import io
import gzip
import re
import pickle
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from time import sleep

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          "integrations", "xgboost_roof_model_v2.pkl")

# ─────────────────────────────────────────────────
# 1. NOAA: Fetch real confirmed hail events
# ─────────────────────────────────────────────────
def fetch_noaa_hail_events(state="OKLAHOMA", year=2024):
    base_url = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
    print(f"[NOAA] Fetching Storm Events for {state}, {year}...")
    
    try:
        index_resp = requests.get(base_url, timeout=15)
        index_resp.raise_for_status()
        pattern = rf'StormEvents_details-ftp_v1\.0_d{year}_c\d+\.csv\.gz'
        matches = re.findall(pattern, index_resp.text)
        if not matches:
            raise ValueError("No matching file found")
        
        filename = matches[-1]
        print(f"[NOAA] Downloading {filename}...")
        resp = requests.get(base_url + filename, timeout=30)
        resp.raise_for_status()
        
        csv_data = gzip.decompress(resp.content)
        df = pd.read_csv(io.BytesIO(csv_data), low_memory=False)
        df = df[df['EVENT_TYPE'].str.upper() == 'HAIL']
        df = df[df['STATE'].str.upper() == state.upper()]
        
        events = df[['BEGIN_DATE_TIME', 'STATE', 'CZ_NAME',
                      'BEGIN_LAT', 'BEGIN_LON', 'MAGNITUDE']].copy()
        events.columns = ['datetime', 'state', 'county', 'lat', 'lon', 'hail_size']
        events['datetime'] = pd.to_datetime(events['datetime'], format='mixed', errors='coerce')
        events = events.dropna(subset=['lat', 'lon', 'datetime'])
        events['hail'] = 1  # positive label
        
        print(f"[NOAA] Loaded {len(events)} confirmed hail events")
        return events
    except Exception as e:
        print(f"[NOAA] Error: {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────────
# 2. Generate negative (clear-sky) samples
# ─────────────────────────────────────────────────
def generate_negatives(positives: pd.DataFrame, n_per_location=1) -> pd.DataFrame:
    """For each hail location, pick a date 30 days later (likely clear) as a negative sample."""
    negatives = []
    unique_locs = positives.drop_duplicates(subset=['lat', 'lon']).head(100)
    
    for _, row in unique_locs.iterrows():
        for offset in [30, 60]:  # Two clear-day offsets
            neg_dt = row['datetime'] + timedelta(days=offset)
            # Keep it in 2024
            if neg_dt.year > 2024:
                neg_dt = row['datetime'] - timedelta(days=offset)
            negatives.append({
                'datetime': neg_dt,
                'state': row['state'],
                'county': row['county'],
                'lat': row['lat'],
                'lon': row['lon'],
                'hail_size': 0.0,
                'hail': 0
            })
    
    df = pd.DataFrame(negatives)
    print(f"[NEG] Generated {len(df)} negative (clear-sky) samples")
    return df

# ─────────────────────────────────────────────────
# 3. Batch fetch Open-Meteo weather for all samples
# ─────────────────────────────────────────────────
def fetch_weather_batch(samples: pd.DataFrame) -> pd.DataFrame:
    """Fetch real historical weather for each sample from Open-Meteo archive API."""
    features_list = []
    total = len(samples)
    
    # Deduplicate API calls: group by (lat, lon, date)
    samples['date_str'] = samples['datetime'].dt.strftime('%Y-%m-%d')
    samples['hour'] = samples['datetime'].dt.hour
    unique_queries = samples.drop_duplicates(subset=['lat', 'lon', 'date_str'])
    
    print(f"[Open-Meteo] Fetching weather for {len(unique_queries)} unique location-dates...")
    
    weather_cache = {}
    for i, (_, row) in enumerate(unique_queries.iterrows()):
        key = (round(row['lat'], 2), round(row['lon'], 2), row['date_str'])
        if key in weather_cache:
            continue
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": row['lat'],
            "longitude": row['lon'],
            "start_date": row['date_str'],
            "end_date": row['date_str'],
            "hourly": "temperature_2m,windspeed_10m,windspeed_100m,precipitation,surface_pressure,cloudcover",
            "timezone": "America/Chicago"
        }
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json().get('hourly', {})
            weather_cache[key] = data
        except Exception as e:
            weather_cache[key] = None
        
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(unique_queries)}] fetched...")
        sleep(0.15)  # Rate limit
    
    print(f"[Open-Meteo] Cached {len(weather_cache)} weather lookups")
    
    # Now map weather back to each sample
    for _, row in samples.iterrows():
        key = (round(row['lat'], 2), round(row['lon'], 2), row['date_str'])
        data = weather_cache.get(key)
        hour = int(row['hour'])
        
        if data is None:
            continue
        
        def safe_get(arr, idx):
            if arr and idx < len(arr) and arr[idx] is not None:
                return arr[idx]
            return np.nan
        
        temp = safe_get(data.get('temperature_2m'), hour)
        ws10 = safe_get(data.get('windspeed_10m'), hour)
        ws100 = safe_get(data.get('windspeed_100m'), hour)
        precip = safe_get(data.get('precipitation'), hour)
        pressure = safe_get(data.get('surface_pressure'), hour)
        cloud = safe_get(data.get('cloudcover'), hour)
        
        # Derived features
        shear = abs(ws100 - ws10) if not np.isnan(ws100) and not np.isnan(ws10) else np.nan
        wind_max = max(ws10, ws100) if not np.isnan(ws10) and not np.isnan(ws100) else np.nan
        
        features_list.append({
            'datetime': row['datetime'],
            'lat': row['lat'],
            'lon': row['lon'],
            'county': row['county'],
            'hail_size': row['hail_size'],
            'hail': row['hail'],
            'temp_2m': temp,
            'wind_10m': ws10,
            'wind_100m': ws100,
            'shear': shear,
            'wind_max': wind_max,
            'precipitation': precip,
            'pressure': pressure,
            'cloudcover': cloud,
            'month': row['datetime'].month,
            'hour': hour
        })
    
    df = pd.DataFrame(features_list).dropna()
    print(f"[Features] Built {len(df)} samples with complete weather features")
    return df

# ─────────────────────────────────────────────────
# 4. Train & Backtest
# ─────────────────────────────────────────────────
def train_and_backtest():
    print("=" * 70)
    print("  ROOF HUNTER — Historical Backtest Engine v2")
    print("  Training XGBoost on REAL NOAA + Open-Meteo data")
    print("=" * 70)
    
    # Step 1: Get real hail events
    positives = fetch_noaa_hail_events("OKLAHOMA", 2024)
    if positives.empty:
        print("FATAL: No NOAA data available.")
        return
    
    # Subsample positives to keep API calls reasonable (use 150 hail events)
    pos_sample = positives.sample(n=min(150, len(positives)), random_state=42)
    
    # Step 2: Generate negatives
    negatives = generate_negatives(pos_sample)
    
    # Step 3: Combine
    all_samples = pd.concat([pos_sample, negatives], ignore_index=True)
    print(f"\nTotal samples before weather fetch: {len(all_samples)} "
          f"({len(pos_sample)} hail, {len(negatives)} clear)")
    
    # Step 4: Fetch real weather
    dataset = fetch_weather_batch(all_samples)
    
    pos_count = dataset['hail'].sum()
    neg_count = len(dataset) - pos_count
    print(f"\nDataset with weather: {len(dataset)} samples ({pos_count} hail, {neg_count} clear)")
    
    if len(dataset) < 20:
        print("FATAL: Not enough data with weather. Exiting.")
        return
    
    # Step 5: Train/test split (80/20)
    feature_cols = ['temp_2m', 'wind_10m', 'wind_100m', 'shear', 'wind_max',
                    'precipitation', 'pressure', 'cloudcover', 'month', 'hour']
    
    from sklearn.model_selection import train_test_split
    X = dataset[feature_cols]
    y = dataset['hail']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Get test metadata for reporting
    test_indices = X_test.index
    test_meta = dataset.loc[test_indices]
    
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    
    # Step 6: Train XGBoost
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=neg_count / max(pos_count, 1),  # Handle class imbalance
        eval_metric='logloss', random_state=42
    )
    model.fit(X_train, y_train)
    
    # Save the new model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"[MODEL] Saved retrained model to {MODEL_PATH}")
    
    # Step 7: Predict on test set
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Step 8: Report per-event results
    print("\n" + "=" * 70)
    print("  BACKTEST — EVENT-LEVEL RESULTS (Hold-out Test Set)")
    print("=" * 70)
    
    for i, (idx, row) in enumerate(test_meta.iterrows()):
        pred_prob = y_pred_proba[i]
        pred_label = y_pred[i]
        actual = int(row['hail'])
        
        hit = "✅" if pred_label == actual else "❌"
        actual_str = f"HAIL {row['hail_size']}\"" if actual else "CLEAR"
        
        print(f"  {hit} {row['datetime'].strftime('%Y-%m-%d %H:%M')} | {str(row['county']):12s} | "
              f"Actual: {actual_str:12s} | Predicted: {pred_prob:.0%} | "
              f"Wind={row['wind_max']:.0f}km/h Precip={row['precipitation']:.1f}mm "
              f"Shear={row['shear']:.0f}")
    
    # Step 9: Metrics
    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    tn = int(((y_pred == 0) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    
    print("\n" + "=" * 70)
    print("  FINAL ACCURACY REPORT")
    print("=" * 70)
    print(f"  Total Test Events:    {total}")
    print(f"  True Positives (TP):  {tp}  — Model said HAIL, hail happened")
    print(f"  True Negatives (TN):  {tn}  — Model said CLEAR, was clear")
    print(f"  False Positives (FP): {fp}  — Model said HAIL, was actually clear")
    print(f"  False Negatives (FN): {fn}  — Model said CLEAR, hail actually happened")
    print()
    print(f"  Accuracy:   {accuracy:.1%}")
    print(f"  Precision:  {precision:.1%}")
    print(f"  Recall:     {recall:.1%}")
    print(f"  F1 Score:   {f1:.1%}")
    
    # Feature importance
    print("\n  Feature Importance:")
    importances = model.feature_importances_
    for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"    {feat:15s} {imp:.3f} {bar}")
    
    # Save results
    output_path = "/Users/noone/QuLabInfinite/roof_hunter/data/backtest_v2_results.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    test_meta = test_meta.copy()
    test_meta['predicted_prob'] = y_pred_proba
    test_meta['predicted_hail'] = y_pred
    test_meta.to_csv(output_path, index=False)
    print(f"\n  Full results saved to {output_path}")


if __name__ == "__main__":
    train_and_backtest()
