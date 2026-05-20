# Roof Hunter

A lightweight weather forecast digital twin scaffold for hail and tornado lead scoring.

## QuLab atmospheric engines (proprietary)

All logic under [`roof_hunter/integrations/`](./integrations/) is **first-party**: transparent formulas, no bundled
external model weights, and JSON fields you can audit end-to-end.

| Engine | Module | What it replaces | Role |
| --- | --- | --- | --- |
| Global outlook | `global_outlook_engine.py` | Global neural NWP outlook bridges | Mixed-layer CAPE uplift, heuristic CIN, shear/PWAT environment index → `qu_global_*`, `qu_outlook_*`; mirrors into legacy `graphcast_*` keys when unset. |
| Satellite nowcast | `satellite_nowcast_engine.py` | GOES + CNN hail classifiers | IR cooling / cold-top / vis–IR glaciation + GLM fusion → `qu_satellite_hail_nowcast_0_1` (optional JSON inputs below). |
| Mesoscale refine | `mesoscale_downscale_engine.py` | Diffusion downscalers | Deterministic sub-grid T/Td/RH refinement + `apply_mesoscale_patch_to_state` (alias `apply_corrdiff_patch_to_state`). |
| HRRR ingest | `nomads_hrrr_ingest.py` | Herbie-style GRIB helpers | AWS HRRR open data + `.idx` **single-range** byte download, `cfgrib` filters per field → `ForecastState` list. |

Optional deps: [`requirements-atmos-ingest.txt`](./requirements-atmos-ingest.txt) (`numpy`, `xarray`, `cfgrib` / eccodes).

**HRRR CLI**

```bash
pip install -r roof_hunter/requirements-atmos-ingest.txt
python -m roof_hunter.integrations.nomads_hrrr_ingest --lat 35 --lon -97 --run "2025-05-01 12:00" --output roof_hunter_forecast.json
```

**GOES-style JSON keys (optional)** for satellite nowcast: `goes_ch13_bt_c`, `goes_ch13_bt_prior_c`, `goes_pair_dt_minutes`, `goes_ch02_reflectance_norm_0_1`, plus existing `lightning_flashes_per_hour`.

**Enrichment before simulation**

```python
import json
from pathlib import Path
from roof_hunter.integrations import enrich_forecast_payload
from roof_hunter.roof_hunter_digital_twin import RoofHunterWeatherTwin

payload = json.loads(Path("roof_hunter_forecast.json").read_text())
twin = RoofHunterWeatherTwin.load_forecast_from_payload(enrich_forecast_payload(payload))
```

You may still supply legacy `graphcast_*` or `lightning_severe_hail_prob_0_1` explicitly; they override inferred outlook/nowcast when present.

## What it does

- reads a time series of forecast states
- uses `AtmosphericScienceLab` to compute basic atmospheric analysis
- scores hail/tornado risk with a simple heuristic
- writes a time-stepped result history for later ingestion by lead scoring or call center workflows

## Usage

1. Create a JSON forecast file named `roof_hunter_forecast.json`.
2. Run:

```bash
python -m roof_hunter.roof_hunter_digital_twin --forecast roof_hunter_forecast.json --output roof_hunter_results.json
```

## Forecast schema

```json
{
  "forecast": [
    {
      "timestamp": "2026-05-03T12:00:00",
      "latitude": 35.0,
      "longitude": -97.0,
      "surface_temp_c": 28.0,
      "relative_humidity": 0.72,
      "surface_dewpoint_c": 22.0,
      "surface_pressure_hpa": 1008.0,
      "surface_pressure_trend_hpa_per_hour": -1.2,
      "precipitable_water_mm": 42.0,
      "low_level_moisture_g_m3": 16.5,
      "wind_speed_m_s": 14.0,
      "wind_direction_deg": 220.0,
      "precip_mm": 10.0
    }
  ]
}
```

## Hail core output

The twin now produces a localized hail core proxy:
- `hail_core_confidence` — confidence of a storm core in the forecast cell
- `hail_core_radius_ft` — approximate core radius in feet
- `hail_core_note` — guidance on how strongly the source supports a core

This is a short-range, high-risk proxy. It is not a precise radar-derived 10-foot location, but it does let your system validate and prioritize the strongest local hail cores using moisture and pressure input.

## Realistic horizon

- A 1-2 day outlook is much more reliable for hail/tornado risk.
- A 5-7 day horizon can still be simulated, but should be treated as scenario analysis rather than operational warning.
- For production, combine this twin with radar, NWP ensemble output, and real storm-tracking layers.

## Last-week validation

A helper is now available to fetch and replay last week’s actual weather for a location.

Use Open-Meteo by default or NOAA station data directly:

```bash
python roof_hunter/validate_last_week.py --source noaa
```

For a longer validation window, add `--window-days`:

```bash
python roof_hunter/validate_last_week.py --source open-meteo --window-days 14 --compare-reports
```

To compare the forecast with real SPC hail/tornado reports, add report matching. For a single-point validation run, use a wider match radius to reflect the forecast footprint:

```bash
python roof_hunter/validate_last_week.py --source noaa --compare-reports --report-match-radius-km 40
```

This writes:
- `roof_hunter/roof_hunter_last_week_forecast.json`
- `roof_hunter/roof_hunter_last_week_results.json`
- `roof_hunter/roof_hunter_last_week_report_matches.json`

Use this to validate predicted hail/tornado risk against observed SPC reports from the same period.

## Lightning / LPI feeds

The twin can use **Open-Meteo `lightning_potential`**, **GOES GLM** flash counts (via `--lightning-glm`, requires `netCDF4`), and/or a **JSON sidecar** for your own strike or LPI time series. See [LIGHTNING_DATA_FEEDS.md](./LIGHTNING_DATA_FEEDS.md) for what each source represents and official references.

```bash
pip install -r roof_hunter/requirements-lightning.txt
python roof_hunter/validate_last_week.py --source open-meteo --window-days 7 --lightning-glm
```

## Implementation Roadmap (Extracted from QULAB:Roof Hunter)

| Phase | Timeline | Focus Area | Key Deliverables |
| --- | --- | --- | --- |
| 1 | 2 Weeks | Data Pipeline Upgrade | NEXRAD, GOES-16, Mesonet integration |
| 2 | 3 Weeks | ML Model Development | XGBoost, LSTM, CNN models |
| 3 | 2 Weeks | Geospatial Enhancements | TITAN tracking, dynamic polygons |
| 4 | 1 Week | Property-Level Precision | Building footprints, roof material DB |
| 5 | 2 Weeks | Real-Time Validation | mPING integration, feedback loop |
| 6 | 3 Weeks | Extended Forecast | HRRR, CAMs, ensemble modeling |
| 7 | 2 Weeks | Scalability & Deployment | TimescaleDB, Kafka, Dask cluster |
| 8 | Ongoing | Continuous Improvement | Weekly model retraining, A/B testing |

## Architecture Diagram

```text
[NOAA/Gov APIs] → [Kafka] → [Dask Cluster] → [TimescaleDB]
                                      ↓
[Property Data] → [Redis Cache] → [FastAPI] → [Web Dashboard]
                                      ↓
[ML Models] → [S3] (Training Data)
```

## Key Metrics to Track

| Metric | Current (Est.) | Target | Industry Avg. |
| --- | --- | --- | --- |
| **Hail Detection Accuracy** | 75% | **95%** | 80% |
| **False Positive Rate** | 20% | **<5%** | 15% |
| **Lead Time (Hail >1.0")** | 0h (reactive) | **2h** | 0.5h |
| **Spatial Resolution** | 5km | **500m** | 2km |
| **Hail Size Error** | ±0.5" | **±0.125"** | ±0.375" |
| **Property-Level Precision** | Zip-code | **Rooftop** | Block-group |
