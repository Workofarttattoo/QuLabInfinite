from __future__ import annotations

from pathlib import Path

from hail_model.azure_digital_twins import (
    RoofHunterWeatherSimulator,
    RoofProfile,
    TwinModelFactory,
    WeatherSnapshot,
)


def test_digital_twin_models_are_valid_dtdl_interfaces(tmp_path: Path):
    models = TwinModelFactory.all_models()

    assert len(models) == 4
    assert {model["@type"] for model in models} == {"Interface"}
    assert any(model["@id"].endswith(":Roof;1") for model in models)
    assert any(model["@id"].endswith(":WeatherCell;1") for model in models)

    written = TwinModelFactory.write_models(tmp_path)
    assert len(written) == 4
    assert all(path.exists() for path in written)


def test_weather_snapshot_converts_to_hail_radar_observation():
    weather = WeatherSnapshot(
        latitude=35.47,
        longitude=-97.52,
        reflectivity_dbz=62.0,
        differential_reflectivity=0.4,
        correlation_coefficient=0.88,
        gust_mps=24.0,
    )

    radar = weather.to_radar_observation()

    assert radar.reflectivity_max == 62.0
    assert radar.correlation_coefficient == 0.88
    assert radar.station_id
    assert radar.echo_top_km > 3.0


def test_roof_weather_simulator_flags_high_hail_risk():
    roof = RoofProfile(
        property_id="okc-demo-001",
        latitude=35.47,
        longitude=-97.52,
        material="asphalt_shingle",
        age_years=22.0,
        drainage_score=0.35,
    )
    weather = WeatherSnapshot(
        latitude=35.47,
        longitude=-97.52,
        air_temp_c=31.0,
        humidity_percent=78.0,
        precipitation_rate_mm_hr=35.0,
        reflectivity_dbz=68.0,
        differential_reflectivity=0.3,
        correlation_coefficient=0.86,
        specific_differential_phase=2.4,
        cape_j_kg=3200.0,
        gust_mps=28.0,
    )

    result = RoofHunterWeatherSimulator().simulate_roof(roof, weather)

    assert result.hail_probability >= 0.5
    assert result.combined_climate_risk_score > 0.3
    assert result.action in {"MONITOR", "QUALIFY"}
    assert "hail" in result.drivers
    assert result.roof_twin_id.startswith("roof-okc-demo-001")
