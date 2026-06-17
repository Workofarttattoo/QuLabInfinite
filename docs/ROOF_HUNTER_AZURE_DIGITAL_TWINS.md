# Roof Hunter Azure Digital Twins Weather Simulator

This integration models Roof Hunter properties, weather cells, stations, and
simulation runs as Azure Digital Twins. It improves Roof Hunter by making
weather assumptions explicit, replayable, and calibratable against local sensor,
radar, and roof outcome data.

Azure Digital Twins does not make weather more accurate by itself. Accuracy
improves when the twins are fed higher-quality local observations and the
simulator is recalibrated against measured roof outcomes.

## What was added

- `hail_model.azure_digital_twins`
  - DTDL model factory for Roof, WeatherCell, WeatherStation, and SimulationRun.
  - Local roof weather simulator that uses the existing hail intelligence bridge.
  - Optional Azure Digital Twins publisher.
- API route:
  - `GET /api/v1/roof-hunter/digital-twin-models`
  - `POST /api/v1/roof-hunter/simulate`
  - `POST /api/v1/roof-hunter/backtest`
- Deployment helper:
  - `PYTHONPATH=/workspace python3 scripts/deploy_roof_hunter_digital_twins.py`
- Backtest helper:
  - `PYTHONPATH=/workspace python3 scripts/backtest_roof_hunter_hail.py <input.csv>`

## Local simulation

```python
from hail_model.azure_digital_twins import (
    RoofHunterWeatherSimulator,
    RoofProfile,
    WeatherSnapshot,
)

roof = RoofProfile(
    property_id="okc-001",
    latitude=35.47,
    longitude=-97.52,
    material="asphalt_shingle",
    age_years=18,
)
weather = WeatherSnapshot(
    latitude=35.47,
    longitude=-97.52,
    reflectivity_dbz=62.0,
    differential_reflectivity=0.4,
    correlation_coefficient=0.88,
    precipitation_rate_mm_hr=25.0,
    gust_mps=24.0,
)

result = RoofHunterWeatherSimulator().simulate_roof(roof, weather)
print(result.to_dict())
```

## API usage

```bash
curl -X POST http://localhost:8000/api/v1/roof-hunter/simulate \
  -H 'Content-Type: application/json' \
  -d '{
    "roof": {
      "property_id": "okc-001",
      "latitude": 35.47,
      "longitude": -97.52,
      "material": "asphalt_shingle",
      "age_years": 18
    },
    "weather": {
      "latitude": 35.47,
      "longitude": -97.52,
      "reflectivity_dbz": 62.0,
      "differential_reflectivity": 0.4,
      "correlation_coefficient": 0.88,
      "precipitation_rate_mm_hr": 25.0,
      "gust_mps": 24.0
    }
  }'
```

## Azure deployment

Install the Azure SDK packages in the deployment environment:

```bash
pip install azure-digitaltwins-core azure-identity
```

Authenticate with Azure CLI or managed identity, then set the ADT endpoint:

```bash
export AZURE_DIGITAL_TWINS_ENDPOINT="https://<name>.api.<region>.digitaltwins.azure.net"
PYTHONPATH=/workspace python3 scripts/deploy_roof_hunter_digital_twins.py --sample
```

Without an endpoint, the script writes the DTDL model JSON files to
`build/roof_hunter_dtdl` and skips Azure upload.

## Data sources to improve accuracy

Use Azure Digital Twins as the state graph and feed it with:

- NOAA/NWS forecasts and active alerts.
- NEXRAD dual-pol radar features.
- Local roof or neighborhood IoT stations.
- Satellite-derived roof material, tree cover, and solar exposure.
- Roof inspection or claim outcomes for calibration.

## Backtesting with known hail outcomes

Use the backtest CLI to compare a coarse NOAA/MOAA-only baseline against
enriched high-quality radar plus NOAA/MOAA inputs:

```bash
PYTHONPATH=/workspace python3 scripts/backtest_roof_hunter_hail.py \
  data/samples/roof_hunter_known_hail_backtest.csv \
  --summary-only
```

The input can be CSV, JSON, or JSONL. Required fields:

- `latitude`/`longitude` or aliases like `lat`/`lon`
- known outcome: `hail_occurred`, `observed_hail`, `target`, or `label`

Recommended enriched radar fields:

- `radar_reflectivity_dbz`
- `radar_differential_reflectivity`
- `radar_correlation_coefficient`
- `radar_specific_differential_phase`
- `cape_j_kg`
- `shear_0_6km_kt`
- `freezing_level_m`

Recommended baseline NOAA/MOAA fields:

- `noaa_reflectivity_dbz`
- `noaa_cape_j_kg`
- `noaa_wind_speed_mps`
- `noaa_gust_mps`
- `noaa_precipitation_rate_mm_hr`

The report includes baseline metrics, enriched metrics, and deltas for
accuracy, precision, recall, F1, false-positive rate, Brier score, and ROC AUC.

