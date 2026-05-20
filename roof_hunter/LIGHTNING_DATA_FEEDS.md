# Lightning and LPI data feeds (Roof Hunter)

This document explains what each wired feed **is**, where it comes from, and how it maps into the digital twin (`lightning_potential_j_kg`, `lightning_flashes_per_hour`).

## 1. Open-Meteo `lightning_potential` (archive / forecast)

- **What it is:** A model-derived **Lightning Potential Index (LPI)**-style field (energy-like units, often labeled J/kg in APIs). It indicates how conducive the column is to lightning in the **assimilated / forecast model**, not a direct strike observation.
- **Where:** Open-Meteo [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) via `hourly=lightning_potential` (already requested in `validate_last_week.fetch_last_week_weather`).
- **Reality check:** For many CONUS archive points, values come back **all `null`**; the twin then falls back to a **CAPE + precip + wind** proxy inside `_lightning_hail_coupling_boost`. Keep requesting LPI anyway for regions or model mixes where it is populated.
- **Twin field:** `lightning_potential_j_kg`

## 2. GOES-R GLM Level 2 LCFA (Geostationary Lightning Mapper)

- **What it is:** NOAA’s **optical** lightning mapper on GOES-16/17/18. **Level 2 LCFA** files list **flashes** (with group/event detail) in ~**20 second** chunks, in NetCDF, on public cloud buckets. Each flash has a latitude/longitude (optical energy–weighted centroid).
- **Where (open data):**
  - **GOES-16 (East):** `s3://noaa-goes16/GLM-L2-LCFA/YYYY/DDD/HH/` (HTTPS: `https://noaa-goes16.s3.amazonaws.com/...`)
  - **GOES-18 (West):** `s3://noaa-goes18/GLM-L2-LCFA/...`
  - Registry: [AWS Open Data — NOAA GOES](https://registry.opendata.aws/noaa-goes/)
- **Product guide:** [NCEI GLM L2 LCFA](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C01527) (format, variables, science context).
- **How we use it:** `roof_hunter/lightning_feeds.py` lists all `.nc` granules for each **UTC hour**, downloads them, counts flashes whose lat/lon falls within **`--glm-radius-km`** of your `--lat`/`--lon`, and sets **`lightning_flashes_per_hour`** on each forecast row (same hour, UTC).
- **Cost / performance:** One UTC hour can be **~180** small files. Full **14-day** replay is feasible but **slow** on first run; results are **cached** under `roof_hunter/.lightning_cache/` (or `--glm-cache-dir`).
- **Dependency:** `pip install netCDF4` (see `requirements-lightning.txt`).
- **Twin field:** `lightning_flashes_per_hour`

## 3. Sidecar JSON (your own strikes / LPI)

Use this for **commercial networks** (Earth Networks, Vaisala, etc.), **research CSVs**, or **blended** products you compute offline.

- **Path:** `--lightning-sidecar path/to.json`
- **Formats** (see `load_lightning_sidecar` in `lightning_feeds.py`):

```json
{
  "version": 1,
  "hours": {
    "2026-04-18T00:00": {
      "lightning_flashes_per_hour": 120.0,
      "lightning_potential_j_kg": 800.0
    }
  }
}
```

or a list of rows:

```json
[
  {"timestamp": "2026-04-18T00:00", "lightning_flashes_per_hour": 120},
  {"timestamp": "2026-04-18T01:00", "lightning_potential_j_kg": 400}
]
```

Timestamps are normalized to UTC hour keys to match Open-Meteo’s `2026-04-18T00:00` style. **Sidecar values override** the same hour when both GLM and sidecar are used.

## CLI wiring (`validate_last_week.py`)

```bash
pip install -r roof_hunter/requirements-lightning.txt

python roof_hunter/validate_last_week.py --source open-meteo --window-days 3 \
  --lightning-glm --glm-radius-km 25 --glm-satellite goes16

python roof_hunter/validate_last_week.py --source open-meteo \
  --lightning-sidecar path/to/lightning_sidecar.json
```

## What the twin does with these numbers

In `roof_hunter_digital_twin.py`, `_lightning_hail_coupling_boost` prefers:

1. Positive `lightning_potential_j_kg` (strongest capped boost),
2. Else positive `lightning_flashes_per_hour`,
3. Else a **proxy** from CAPE, precipitation, and wind.

So **linking lightning to hail** in your stack means: keep GLM or provider flash rates and/or LPI **on the same hourly timeline** as the surface/archive weather that drives the twin.
