from __future__ import annotations

from pathlib import Path

from hail_model.backtest import RoofHunterBacktester, load_backtest_records, record_from_mapping


def test_record_from_mapping_builds_baseline_and_enriched_weather():
    record = record_from_mapping(
        {
            "record_id": "case-1",
            "lat": 35.47,
            "lon": -97.52,
            "hail_occurred": "true",
            "noaa_reflectivity_dbz": 28,
            "radar_reflectivity_dbz": 66,
            "radar_differential_reflectivity": 0.4,
            "radar_correlation_coefficient": 0.88,
        }
    )

    assert record.hail_occurred is True
    assert record.baseline_weather.reflectivity_dbz == 28
    assert record.enriched_weather.reflectivity_dbz == 66
    assert record.enriched_weather.correlation_coefficient == 0.88


def test_backtest_shows_enriched_radar_improves_sample_fixture():
    fixture = Path(__file__).resolve().parent.parent / "data" / "samples" / "roof_hunter_known_hail_backtest.csv"
    records = load_backtest_records(fixture)

    report = RoofHunterBacktester().backtest(records)

    assert report.baseline.count == 8
    assert report.enriched.recall > report.baseline.recall
    assert report.enriched.f1 > report.baseline.f1
    assert report.improvement["recall_delta"] > 0
    assert report.improvement["brier_score_delta"] < 0

