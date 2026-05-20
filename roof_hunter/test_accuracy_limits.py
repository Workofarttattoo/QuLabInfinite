"""
Roof Hunter — Accuracy & Limits Stress Test
=============================================
Tests the full prediction pipeline against REAL NOAA Storm Events data across:
  • Multiple states (OK, TX, KS, NE, CO)
  • Multiple years (2023, 2024)
  • Multiple forecast horizons (0h → 72h simulated lead time)
  • Hail size tiers (any hail, ≥1", ≥2", ≥3")

Metrics: Accuracy, Precision, Recall, F1, AUC-ROC, Brier Skill Score
Output: JSON report + terminal summary with degradation curves
"""

import os, sys, io, gzip, re, json, math, pickle
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, timezone
from time import sleep
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# 1. NOAA Storm Events fetcher (real confirmed hail)
# ---------------------------------------------------------------------------
NOAA_BASE = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"

def fetch_noaa_hail_events(state: str, year: int, max_events: int = 300) -> pd.DataFrame:
    print(f"[NOAA] Fetching Storm Events: {state} {year}...")
    try:
        idx = requests.get(NOAA_BASE, timeout=20)
        idx.raise_for_status()
        pat = rf'StormEvents_details-ftp_v1\.0_d{year}_c\d+\.csv\.gz'
        hits = re.findall(pat, idx.text)
        if not hits:
            print(f"  ⚠ No file for {year}")
            return pd.DataFrame()
        resp = requests.get(NOAA_BASE + hits[-1], timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.BytesIO(gzip.decompress(resp.content)), low_memory=False)
        df = df[df['EVENT_TYPE'].str.upper() == 'HAIL']
        df = df[df['STATE'].str.upper() == state.upper()]
        ev = df[['BEGIN_DATE_TIME','STATE','CZ_NAME',
                 'BEGIN_LAT','BEGIN_LON','MAGNITUDE']].copy()
        ev.columns = ['datetime','state','county','lat','lon','hail_size']
        ev['datetime'] = pd.to_datetime(ev['datetime'], format='mixed', errors='coerce')
        ev = ev.dropna(subset=['lat','lon','datetime'])
        ev['hail'] = 1
        if len(ev) > max_events:
            ev = ev.sample(n=max_events, random_state=42)
        print(f"  ✓ {len(ev)} hail events")
        return ev
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return pd.DataFrame()


def generate_negatives(positives: pd.DataFrame, ratio: float = 1.0) -> pd.DataFrame:
    n = int(len(positives) * ratio)
    locs = positives.drop_duplicates(subset=['lat','lon']).head(n)
    negs = []
    for _, r in locs.iterrows():
        offset = 30
        neg_dt = r['datetime'] + timedelta(days=offset)
        if neg_dt.year > r['datetime'].year:
            neg_dt = r['datetime'] - timedelta(days=offset)
        negs.append({'datetime': neg_dt, 'state': r['state'], 'county': r['county'],
                     'lat': r['lat'], 'lon': r['lon'], 'hail_size': 0.0, 'hail': 0})
    df = pd.DataFrame(negs)
    print(f"[NEG] {len(df)} clear-sky samples")
    return df


# ---------------------------------------------------------------------------
# 2. Open-Meteo weather fetch
# ---------------------------------------------------------------------------
def fetch_weather_batch(samples: pd.DataFrame) -> pd.DataFrame:
    samples = samples.copy()
    samples['date_str'] = samples['datetime'].dt.strftime('%Y-%m-%d')
    samples['hour'] = samples['datetime'].dt.hour
    uniq = samples.drop_duplicates(subset=['lat','lon','date_str'])
    print(f"[Open-Meteo] {len(uniq)} unique queries...")

    cache: Dict[tuple, Any] = {}
    for i, (_, r) in enumerate(uniq.iterrows()):
        key = (round(r['lat'],2), round(r['lon'],2), r['date_str'])
        if key in cache:
            continue
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": r['lat'], "longitude": r['lon'],
            "start_date": r['date_str'], "end_date": r['date_str'],
            "hourly": "temperature_2m,windspeed_10m,windspeed_100m,precipitation,"
                      "surface_pressure,cloudcover,relativehumidity_2m,dewpoint_2m",
            "timezone": "America/Chicago"
        }
        try:
            resp = requests.get(url, params=params, timeout=12)
            resp.raise_for_status()
            cache[key] = resp.json().get('hourly', {})
        except:
            cache[key] = None
        if (i+1) % 30 == 0:
            print(f"  [{i+1}/{len(uniq)}]")
        sleep(0.12)

    rows = []
    for _, r in samples.iterrows():
        key = (round(r['lat'],2), round(r['lon'],2), r['date_str'])
        d = cache.get(key)
        h = int(r['hour'])
        if d is None:
            continue
        def sg(arr, idx):
            if arr and idx < len(arr) and arr[idx] is not None:
                return arr[idx]
            return np.nan
        t = sg(d.get('temperature_2m'), h)
        ws10 = sg(d.get('windspeed_10m'), h)
        ws100 = sg(d.get('windspeed_100m'), h)
        pr = sg(d.get('precipitation'), h)
        sp = sg(d.get('surface_pressure'), h)
        cc = sg(d.get('cloudcover'), h)
        rh = sg(d.get('relativehumidity_2m'), h)
        dp = sg(d.get('dewpoint_2m'), h)
        shear = abs(ws100 - ws10) if not (np.isnan(ws100) or np.isnan(ws10)) else np.nan
        wmax = max(ws10, ws100) if not (np.isnan(ws10) or np.isnan(ws100)) else np.nan
        rows.append({
            'datetime': r['datetime'], 'lat': r['lat'], 'lon': r['lon'],
            'county': r['county'], 'state': r['state'],
            'hail_size': r['hail_size'], 'hail': r['hail'],
            'temp_2m': t, 'dewpoint_2m': dp, 'rh': rh,
            'wind_10m': ws10, 'wind_100m': ws100, 'shear': shear,
            'wind_max': wmax, 'precipitation': pr,
            'pressure': sp, 'cloudcover': cc,
            'month': r['datetime'].month, 'hour': h
        })
    df = pd.DataFrame(rows).dropna()
    print(f"[Features] {len(df)} complete samples")
    return df


# ---------------------------------------------------------------------------
# 3. Digital-twin hail scoring (uses the real pipeline)
# ---------------------------------------------------------------------------
def score_with_digital_twin(row: pd.Series) -> float:
    """Run a single sample through the full RoofHunterWeatherTwin pipeline."""
    try:
        from roof_hunter.roof_hunter_digital_twin import ForecastState, RoofHunterWeatherTwin
        state = ForecastState(
            timestamp=row['datetime'].to_pydatetime().replace(tzinfo=timezone.utc),
            latitude=row['lat'], longitude=row['lon'],
            surface_temp_c=row['temp_2m'],
            relative_humidity=row['rh'] / 100.0 if row['rh'] > 1 else row['rh'],
            surface_dewpoint_c=row['dewpoint_2m'],
            surface_pressure_hpa=row['pressure'],
            wind_speed_m_s=row['wind_max'] / 3.6,  # km/h → m/s
            precip_mm=row['precipitation'],
        )
        twin = RoofHunterWeatherTwin([state])
        history = twin.simulate()
        return history[0]['hail_probability']
    except Exception as e:
        return -1.0


# ---------------------------------------------------------------------------
# 4. XGBoost-only prediction (v1 model)
# ---------------------------------------------------------------------------
def score_with_xgboost_v1(row: pd.Series) -> float:
    try:
        from roof_hunter.integrations.ml_models import predict_hail_xgboost
        return predict_hail_xgboost(
            dbz=45.0,  # proxy
            cape=max(0, (row['temp_2m'] - 10) * 80 + (row['rh'] - 50) * 15),
            shear=row['shear'],
            temp=row['temp_2m']
        )
    except:
        return -1.0


# ---------------------------------------------------------------------------
# 5. Simulated lead-time degradation
# ---------------------------------------------------------------------------
def add_lead_time_noise(dataset: pd.DataFrame, lead_hours: int) -> pd.DataFrame:
    """Simulate forecast uncertainty at longer lead times by adding Gaussian noise."""
    if lead_hours == 0:
        return dataset.copy()
    df = dataset.copy()
    rng = np.random.default_rng(seed=lead_hours)
    # Noise scales with sqrt(lead_hours) — standard meteorological error growth
    scale = math.sqrt(lead_hours)
    df['temp_2m'] += rng.normal(0, 0.8 * scale, len(df))
    df['dewpoint_2m'] += rng.normal(0, 0.6 * scale, len(df))
    df['wind_10m'] += rng.normal(0, 1.5 * scale, len(df)).clip(-20)
    df['wind_100m'] += rng.normal(0, 2.0 * scale, len(df)).clip(-30)
    df['shear'] = np.abs(df['wind_100m'] - df['wind_10m'])
    df['wind_max'] = np.maximum(df['wind_10m'], df['wind_100m'])
    df['precipitation'] = (df['precipitation'] + rng.normal(0, 0.5 * scale, len(df))).clip(0)
    df['pressure'] += rng.normal(0, 0.3 * scale, len(df))
    df['rh'] = (df['rh'] + rng.normal(0, 3 * scale, len(df))).clip(0, 100)
    return df


# ---------------------------------------------------------------------------
# 6. Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = 0.3) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    # AUC-ROC (manual trapezoidal)
    auc = _manual_auc(y_true, y_prob)

    # Brier score
    brier = float(np.mean((y_prob - y_true) ** 2))
    # Climatological Brier (predict base rate always)
    climo = y_true.mean()
    brier_climo = float(np.mean((climo - y_true) ** 2))
    bss = 1.0 - brier / brier_climo if brier_climo > 0 else 0.0

    return {
        'accuracy': round(acc, 4), 'precision': round(prec, 4),
        'recall': round(rec, 4), 'f1': round(f1, 4),
        'auc_roc': round(auc, 4), 'brier_score': round(brier, 4),
        'brier_skill_score': round(bss, 4),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'threshold': threshold, 'n_samples': total
    }


def _manual_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Trapezoidal AUC-ROC without sklearn."""
    order = np.argsort(-y_prob)
    y_s = y_true[order]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tpr_prev, fpr_prev, auc = 0.0, 0.0, 0.0
    tp_count, fp_count = 0, 0
    for i in range(len(y_s)):
        if y_s[i] == 1:
            tp_count += 1
        else:
            fp_count += 1
        tpr = tp_count / n_pos
        fpr = fp_count / n_neg
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
        tpr_prev, fpr_prev = tpr, fpr
    return float(auc)


# ---------------------------------------------------------------------------
# 7. Main test runner
# ---------------------------------------------------------------------------
STATES = ['OKLAHOMA', 'TEXAS', 'KANSAS', 'NEBRASKA', 'COLORADO']
YEARS = [2023, 2024]
# Extended lead-time horizon: 0h → 90 days (2160h)
LEAD_HOURS = [0, 3, 6, 12, 18, 24, 36, 48, 72, 120, 168, 240, 336, 504, 720, 1080, 1440, 2160]
#             0h 3h 6h 12h 18h 1d  1.5d 2d  3d   5d   7d   10d  14d  21d  30d  45d   60d   90d
HAIL_TIERS = {
    'any_hail': 0.0,
    'severe_1in': 1.0,
    'significant_2in': 2.0,
    'giant_3in': 3.0,
}
EVENTS_PER_STATE = 40  # 40 × 5 states × 2 years = up to 400 hail events + ~400 negatives


def run_full_accuracy_test():
    print("=" * 72)
    print("  ROOF HUNTER — ACCURACY & LIMITS STRESS TEST")
    print("  Testing prediction pipeline against REAL NOAA Storm Events")
    print("=" * 72)

    # ── Phase 1: Collect data ──────────────────────────────────
    all_pos = []
    for state in STATES:
        for year in YEARS:
            ev = fetch_noaa_hail_events(state, year, max_events=EVENTS_PER_STATE)
            if not ev.empty:
                all_pos.append(ev)
            sleep(0.5)

    if not all_pos:
        print("FATAL: No NOAA data retrieved. Check network.")
        return

    positives = pd.concat(all_pos, ignore_index=True)
    negatives = generate_negatives(positives, ratio=1.0)
    all_samples = pd.concat([positives, negatives], ignore_index=True)
    print(f"\n[DATA] Total samples: {len(all_samples)} "
          f"({len(positives)} hail, {len(negatives)} clear)")

    # ── Phase 2: Fetch weather ─────────────────────────────────
    dataset = fetch_weather_batch(all_samples)
    if len(dataset) < 30:
        print("FATAL: Insufficient weather data.")
        return

    # ── Phase 3: Score every sample with Digital Twin ──────────
    print("\n[SCORING] Running Digital Twin on all samples...")
    probs_twin = []
    probs_xgb = []
    for i, (_, row) in enumerate(dataset.iterrows()):
        p_twin = score_with_digital_twin(row)
        p_xgb = score_with_xgboost_v1(row)
        probs_twin.append(p_twin)
        probs_xgb.append(p_xgb)
        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{len(dataset)}] scored")

    dataset['prob_twin'] = probs_twin
    dataset['prob_xgb'] = probs_xgb
    # Remove failures
    dataset = dataset[(dataset['prob_twin'] >= 0) & (dataset['prob_xgb'] >= 0)]

    y_true = dataset['hail'].values.astype(int)
    y_twin = dataset['prob_twin'].values
    y_xgb = dataset['prob_xgb'].values

    # ── Phase 4: Baseline metrics ──────────────────────────────
    results: Dict[str, Any] = {'timestamp': datetime.now(timezone.utc).isoformat()}

    print("\n" + "=" * 72)
    print("  BASELINE ACCURACY (Lead Time = 0h, real observed weather)")
    print("=" * 72)

    for thresh in [0.2, 0.3, 0.4, 0.5]:
        m_twin = compute_metrics(y_true, y_twin, threshold=thresh)
        m_xgb = compute_metrics(y_true, y_xgb, threshold=thresh)
        print(f"\n  Threshold {thresh:.0%}:")
        print(f"    Digital Twin  → Acc={m_twin['accuracy']:.1%}  P={m_twin['precision']:.1%}"
              f"  R={m_twin['recall']:.1%}  F1={m_twin['f1']:.1%}  AUC={m_twin['auc_roc']:.3f}"
              f"  BSS={m_twin['brier_skill_score']:.3f}")
        print(f"    XGBoost-only  → Acc={m_xgb['accuracy']:.1%}  P={m_xgb['precision']:.1%}"
              f"  R={m_xgb['recall']:.1%}  F1={m_xgb['f1']:.1%}  AUC={m_xgb['auc_roc']:.3f}"
              f"  BSS={m_xgb['brier_skill_score']:.3f}")

    results['baseline'] = {
        'twin': compute_metrics(y_true, y_twin, 0.3),
        'xgboost': compute_metrics(y_true, y_xgb, 0.3),
    }

    # ── Phase 5: Lead-time degradation ─────────────────────────
    print("\n" + "=" * 72)
    print("  LEAD-TIME DEGRADATION CURVE")
    print("  How far out can we predict accurately?")
    print("=" * 72)

    lead_results = []
    for lh in LEAD_HOURS:
        noisy = add_lead_time_noise(dataset, lh)
        p_list = []
        for _, row in noisy.iterrows():
            p_list.append(score_with_digital_twin(row))
        noisy['prob_lead'] = p_list
        noisy = noisy[noisy['prob_lead'] >= 0]
        yt = noisy['hail'].values.astype(int)
        yp = noisy['prob_lead'].values
        m = compute_metrics(yt, yp, 0.3)
        lead_results.append({'lead_hours': lh, **m})

        lead_label = f"+{lh}h" if lh < 24 else f"+{lh}h ({lh//24}d)"
        bar_f1 = "█" * int(m['f1'] * 40)
        status = "✅" if m['f1'] >= 0.4 else ("⚠️" if m['f1'] >= 0.2 else "❌")
        print(f"  {status} {lead_label:12s} │ F1={m['f1']:.1%}  AUC={m['auc_roc']:.3f}"
              f"  P={m['precision']:.1%}  R={m['recall']:.1%} │ {bar_f1}")

    results['lead_time_curve'] = lead_results

    # ── Phase 6: Per-state breakdown ───────────────────────────
    print("\n" + "=" * 72)
    print("  GEOGRAPHIC ACCURACY (per state)")
    print("=" * 72)

    state_results = {}
    for state in STATES:
        subset = dataset[dataset['state'].str.upper() == state]
        if len(subset) < 10:
            print(f"  {state:12s} │ Insufficient data ({len(subset)} samples)")
            continue
        yt = subset['hail'].values.astype(int)
        yp = subset['prob_twin'].values
        m = compute_metrics(yt, yp, 0.3)
        state_results[state] = m
        n_hail = int(yt.sum())
        print(f"  {state:12s} │ n={len(subset):3d} ({n_hail} hail) │"
              f" F1={m['f1']:.1%}  AUC={m['auc_roc']:.3f}  BSS={m['brier_skill_score']:.3f}")

    results['by_state'] = state_results

    # ── Phase 7: Hail size tiers ───────────────────────────────
    print("\n" + "=" * 72)
    print("  HAIL SIZE TIER DETECTION RATES")
    print("=" * 72)

    tier_results = {}
    for tier_name, min_size in HAIL_TIERS.items():
        hail_mask = dataset['hail_size'] >= min_size
        clear_mask = dataset['hail'] == 0
        subset = dataset[hail_mask | clear_mask].copy()
        subset['tier_label'] = (subset['hail_size'] >= min_size).astype(int)
        if subset['tier_label'].sum() < 5:
            print(f"  {tier_name:20s} │ Too few events ({subset['tier_label'].sum()})")
            continue
        yt = subset['tier_label'].values
        yp = subset['prob_twin'].values
        m = compute_metrics(yt, yp, 0.3)
        tier_results[tier_name] = m
        print(f"  {tier_name:20s} │ n={len(subset):3d} ({int(yt.sum())} events) │"
              f" F1={m['f1']:.1%}  R={m['recall']:.1%}  AUC={m['auc_roc']:.3f}")

    results['by_hail_tier'] = tier_results

    # ── Phase 8: Find the limits ───────────────────────────────
    print("\n" + "=" * 72)
    print("  ⚡ LIMITS ASSESSMENT")
    print("=" * 72)

    # Find max useful lead time (F1 ≥ 0.25)
    useful_leads = [r for r in lead_results if r['f1'] >= 0.25]
    max_useful = max((r['lead_hours'] for r in useful_leads), default=0)

    # Find best state
    best_state = max(state_results.items(), key=lambda x: x[1]['f1'])[0] if state_results else "N/A"
    worst_state = min(state_results.items(), key=lambda x: x[1]['f1'])[0] if state_results else "N/A"

    # Smallest detectable hail
    detectable = [t for t, m in tier_results.items() if m['recall'] >= 0.3]

    print(f"  Max useful lead time (F1≥0.25):  {max_useful}h")
    print(f"  Best geography:                  {best_state}")
    print(f"  Worst geography:                 {worst_state}")
    print(f"  Detectable hail tiers (R≥0.3):   {', '.join(detectable) if detectable else 'None'}")
    print(f"  Baseline AUC-ROC (Twin):         {results['baseline']['twin']['auc_roc']:.3f}")
    print(f"  Baseline F1 (Twin @ 0.3):        {results['baseline']['twin']['f1']:.1%}")
    print(f"  Total samples tested:            {len(dataset)}")

    results['limits'] = {
        'max_useful_lead_hours': max_useful,
        'best_state': best_state,
        'worst_state': worst_state,
        'detectable_tiers': detectable,
        'total_samples': len(dataset),
    }

    # ── Save ───────────────────────────────────────────────────
    out_path = Path(__file__).parent / "data" / "accuracy_limits_report.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Full report saved to {out_path}")

    csv_path = Path(__file__).parent / "data" / "accuracy_limits_dataset.csv"
    dataset.to_csv(csv_path, index=False)
    print(f"  Dataset saved to {csv_path}")

    print("\n" + "=" * 72)
    print("  TEST COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    run_full_accuracy_test()
