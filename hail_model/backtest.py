"""
Backtesting utilities for Roof Hunter hail and climate predictions.

The backtester compares two prediction modes against known outcomes:
  - baseline: coarse NOAA/MOAA-style weather fields only
  - enriched: high-quality radar + dual-pol + NOAA/MOAA fields

Input files may be CSV, JSON, or JSONL. Required columns are latitude,
longitude, and a known hail outcome such as hail_occurred. Roof metadata and
radar fields are optional but strongly recommended for meaningful backtests.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .azure_digital_twins import RoofHunterWeatherSimulator, RoofProfile, WeatherSnapshot
from .roof_hunter_bridge import HailIntelligence

_TRUE_VALUES = {"1", "true", "t", "yes", "y", "hail", "severe hail"}
_OUTCOME_ALIASES = ("hail_occurred", "hail", "observed_hail", "target", "label")
_LAT_ALIASES = ("latitude", "lat", "begin_lat", "event_latitude")
_LON_ALIASES = ("longitude", "lon", "lng", "begin_lon", "event_longitude")


@dataclass
class BacktestRecord:
    """One historical roof/weather/outcome record."""

    record_id: str
    roof: RoofProfile
    baseline_weather: WeatherSnapshot
    enriched_weather: WeatherSnapshot
    hail_occurred: bool
    observed_hail_size_inches: float = 0.0


@dataclass
class PredictionComparison:
    """Predictions for one record under baseline and enriched inputs."""

    record_id: str
    hail_occurred: bool
    observed_hail_size_inches: float
    baseline_probability: float
    enriched_probability: float
    baseline_action: str
    enriched_action: str
    baseline_risk_level: str
    enriched_risk_level: str
    probability_delta: float
    enriched_drivers: list[str]


@dataclass
class BacktestMetrics:
    """Binary classification and calibration metrics."""

    count: int
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    brier_score: float
    roc_auc: float | None
    confusion_matrix: dict[str, int]


@dataclass
class BacktestReport:
    """Backtest report comparing baseline and enriched prediction quality."""

    baseline: BacktestMetrics
    enriched: BacktestMetrics
    improvement: dict[str, float | None]
    comparisons: list[PredictionComparison]

    def to_dict(self, include_records: bool = True) -> dict[str, Any]:
        payload = {
            "baseline": asdict(self.baseline),
            "enriched": asdict(self.enriched),
            "improvement": self.improvement,
        }
        if include_records:
            payload["comparisons"] = [asdict(item) for item in self.comparisons]
        return payload


class RoofHunterBacktester:
    """Run historical backtests for Roof Hunter weather simulations."""

    def __init__(
        self,
        simulator: RoofHunterWeatherSimulator | None = None,
        model_path: str | Path | None = None,
    ):
        if simulator is not None:
            self.simulator = simulator
        elif model_path is not None:
            self.simulator = RoofHunterWeatherSimulator(
                hail_intelligence=HailIntelligence(model_path=model_path)
            )
        else:
            self.simulator = RoofHunterWeatherSimulator()

    def backtest(
        self,
        records: list[BacktestRecord],
        threshold: float = 0.5,
    ) -> BacktestReport:
        """Compare baseline and enriched predictions against known outcomes."""

        comparisons: list[PredictionComparison] = []
        for record in records:
            baseline = self.simulator.simulate_roof(record.roof, record.baseline_weather)
            enriched = self.simulator.simulate_roof(record.roof, record.enriched_weather)
            comparisons.append(
                PredictionComparison(
                    record_id=record.record_id,
                    hail_occurred=record.hail_occurred,
                    observed_hail_size_inches=record.observed_hail_size_inches,
                    baseline_probability=baseline.hail_probability,
                    enriched_probability=enriched.hail_probability,
                    baseline_action=baseline.action,
                    enriched_action=enriched.action,
                    baseline_risk_level=baseline.risk_level,
                    enriched_risk_level=enriched.risk_level,
                    probability_delta=round(
                        enriched.hail_probability - baseline.hail_probability,
                        4,
                    ),
                    enriched_drivers=enriched.drivers,
                )
            )

        baseline_metrics = _metrics(
            [item.hail_occurred for item in comparisons],
            [item.baseline_probability for item in comparisons],
            threshold,
        )
        enriched_metrics = _metrics(
            [item.hail_occurred for item in comparisons],
            [item.enriched_probability for item in comparisons],
            threshold,
        )

        return BacktestReport(
            baseline=baseline_metrics,
            enriched=enriched_metrics,
            improvement=_improvement(baseline_metrics, enriched_metrics),
            comparisons=comparisons,
        )


def load_backtest_records(path: str | Path) -> list[BacktestRecord]:
    """Load backtest records from CSV, JSON array, or JSONL."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    elif suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows = payload["records"] if isinstance(payload, dict) and "records" in payload else payload
    else:
        raise ValueError(f"Unsupported backtest file type: {source.suffix}")

    if not isinstance(rows, list):
        raise ValueError("Backtest file must contain a list of records")
    return [record_from_mapping(row, index=i) for i, row in enumerate(rows, start=1)]


def record_from_mapping(row: dict[str, Any], index: int = 1) -> BacktestRecord:
    """Normalize one flat record from NOAA/MOAA/radar exports."""

    normalized = {_normalize_key(k): v for k, v in row.items()}
    lat = _float(_first(normalized, _LAT_ALIASES), "latitude")
    lon = _float(_first(normalized, _LON_ALIASES), "longitude")
    outcome = _parse_bool(_first(normalized, _OUTCOME_ALIASES, default=False))
    record_id = str(
        _first(
            normalized,
            ("record_id", "event_id", "episode_id", "property_id"),
            default=f"record-{index}",
        )
    )

    roof = RoofProfile(
        property_id=str(_first(normalized, ("property_id",), default=record_id)),
        latitude=lat,
        longitude=lon,
        material=str(_first(normalized, ("roof_material", "material"), default="asphalt_shingle")),
        area_sqft=_float(_first(normalized, ("area_sqft",), default=1800.0), "area_sqft"),
        slope_degrees=_float(_first(normalized, ("slope_degrees",), default=25.0), "slope_degrees"),
        age_years=_float(_first(normalized, ("age_years",), default=8.0), "age_years"),
        albedo=_float(_first(normalized, ("albedo",), default=0.18), "albedo"),
        emissivity=_float(_first(normalized, ("emissivity",), default=0.90), "emissivity"),
        drainage_score=_float(
            _first(normalized, ("drainage_score",), default=0.75),
            "drainage_score",
        ),
        tree_cover_percent=_float(
            _first(normalized, ("tree_cover_percent", "tree_cover_pct"), default=10.0),
            "tree_cover_percent",
        ),
        elevation_m=_float(_first(normalized, ("elevation_m",), default=300.0), "elevation_m"),
    )

    timestamp = str(_first(normalized, ("timestamp", "time", "begin_time"), default=""))
    if not timestamp:
        timestamp = WeatherSnapshot(latitude=lat, longitude=lon).timestamp

    baseline_weather = _baseline_weather(normalized, lat, lon, timestamp)
    enriched_weather = _enriched_weather(normalized, lat, lon, timestamp)
    return BacktestRecord(
        record_id=record_id,
        roof=roof,
        baseline_weather=baseline_weather,
        enriched_weather=enriched_weather,
        hail_occurred=outcome,
        observed_hail_size_inches=_float(
            _first(
                normalized,
                ("hail_size_inches", "magnitude", "hail_size", "max_hail_size_inches"),
                default=0.0,
            ),
            "observed_hail_size_inches",
        ),
    )


def _baseline_weather(row: dict[str, Any], lat: float, lon: float, timestamp: str) -> WeatherSnapshot:
    """Build a coarse NOAA/MOAA-only weather snapshot."""

    precip = _float(
        _first(row, ("noaa_precipitation_rate_mm_hr", "precipitation_rate_mm_hr"), default=0.0),
        "precipitation_rate_mm_hr",
    )
    cape = _float(_first(row, ("noaa_cape_j_kg", "cape_j_kg", "cape"), default=800.0), "cape")
    # Baseline intentionally uses coarse reflectivity so high-quality radar can
    # show its incremental value during backtests.
    coarse_reflectivity = _float(
        _first(row, ("noaa_reflectivity_dbz", "forecast_reflectivity_dbz"), default=28.0),
        "noaa_reflectivity_dbz",
    )
    return WeatherSnapshot(
        latitude=lat,
        longitude=lon,
        timestamp=timestamp,
        air_temp_c=_float(_first(row, ("noaa_air_temp_c", "air_temp_c"), default=24.0), "air_temp_c"),
        dewpoint_c=_float(_first(row, ("noaa_dewpoint_c", "dewpoint_c"), default=16.0), "dewpoint_c"),
        humidity_percent=_float(
            _first(row, ("noaa_humidity_percent", "humidity_percent"), default=65.0),
            "humidity_percent",
        ),
        wind_speed_mps=_float(
            _first(row, ("noaa_wind_speed_mps", "wind_speed_mps"), default=6.0),
            "wind_speed_mps",
        ),
        wind_direction_degrees=_float(
            _first(row, ("noaa_wind_direction_degrees", "wind_direction_degrees"), default=220.0),
            "wind_direction_degrees",
        ),
        gust_mps=_float(_first(row, ("noaa_gust_mps", "gust_mps"), default=10.0), "gust_mps"),
        pressure_hpa=_float(_first(row, ("noaa_pressure_hpa", "pressure_hpa"), default=1010.0), "pressure_hpa"),
        precipitation_rate_mm_hr=precip,
        cloud_cover_percent=_float(
            _first(row, ("noaa_cloud_cover_percent", "cloud_cover_percent"), default=45.0),
            "cloud_cover_percent",
        ),
        solar_radiation_w_m2=_float(
            _first(row, ("noaa_solar_radiation_w_m2", "solar_radiation_w_m2"), default=650.0),
            "solar_radiation_w_m2",
        ),
        reflectivity_dbz=coarse_reflectivity,
        differential_reflectivity=1.4,
        correlation_coefficient=0.97,
        specific_differential_phase=0.4,
        cape_j_kg=cape,
        shear_0_6km_kt=_float(
            _first(row, ("noaa_shear_0_6km_kt", "shear_0_6km_kt", "shear_0_6km"), default=25.0),
            "shear_0_6km_kt",
        ),
        freezing_level_m=_float(
            _first(row, ("noaa_freezing_level_m", "freezing_level_m"), default=3500.0),
            "freezing_level_m",
        ),
    )


def _enriched_weather(row: dict[str, Any], lat: float, lon: float, timestamp: str) -> WeatherSnapshot:
    """Build an enriched snapshot from high-quality radar + NOAA/MOAA fields."""

    baseline = _baseline_weather(row, lat, lon, timestamp)
    return WeatherSnapshot(
        latitude=lat,
        longitude=lon,
        timestamp=timestamp,
        air_temp_c=baseline.air_temp_c,
        dewpoint_c=baseline.dewpoint_c,
        humidity_percent=baseline.humidity_percent,
        wind_speed_mps=baseline.wind_speed_mps,
        wind_direction_degrees=baseline.wind_direction_degrees,
        gust_mps=_float(_first(row, ("radar_gust_mps", "gust_mps"), default=baseline.gust_mps), "gust_mps"),
        pressure_hpa=baseline.pressure_hpa,
        precipitation_rate_mm_hr=baseline.precipitation_rate_mm_hr,
        cloud_cover_percent=baseline.cloud_cover_percent,
        solar_radiation_w_m2=baseline.solar_radiation_w_m2,
        reflectivity_dbz=_float(
            _first(row, ("radar_reflectivity_dbz", "reflectivity_dbz", "reflectivity_max"), default=baseline.reflectivity_dbz),
            "radar_reflectivity_dbz",
        ),
        differential_reflectivity=_float(
            _first(row, ("radar_differential_reflectivity", "differential_reflectivity"), default=baseline.differential_reflectivity),
            "differential_reflectivity",
        ),
        correlation_coefficient=_float(
            _first(row, ("radar_correlation_coefficient", "correlation_coefficient"), default=baseline.correlation_coefficient),
            "correlation_coefficient",
        ),
        specific_differential_phase=_float(
            _first(row, ("radar_specific_differential_phase", "specific_differential_phase"), default=baseline.specific_differential_phase),
            "specific_differential_phase",
        ),
        cape_j_kg=_float(_first(row, ("cape_j_kg", "cape"), default=baseline.cape_j_kg), "cape_j_kg"),
        shear_0_6km_kt=_float(
            _first(row, ("shear_0_6km_kt", "shear_0_6km"), default=baseline.shear_0_6km_kt),
            "shear_0_6km_kt",
        ),
        freezing_level_m=_float(
            _first(row, ("freezing_level_m",), default=baseline.freezing_level_m),
            "freezing_level_m",
        ),
    )


def _metrics(labels: list[bool], probabilities: list[float], threshold: float) -> BacktestMetrics:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length")
    if not labels:
        raise ValueError("at least one backtest record is required")

    predictions = [prob >= threshold for prob in probabilities]
    tp = sum(1 for y, pred in zip(labels, predictions, strict=True) if y and pred)
    tn = sum(1 for y, pred in zip(labels, predictions, strict=True) if not y and not pred)
    fp = sum(1 for y, pred in zip(labels, predictions, strict=True) if not y and pred)
    fn = sum(1 for y, pred in zip(labels, predictions, strict=True) if y and not pred)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return BacktestMetrics(
        count=len(labels),
        threshold=threshold,
        accuracy=round(_safe_div(tp + tn, len(labels)), 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        false_positive_rate=round(_safe_div(fp, fp + tn), 4),
        brier_score=round(
            sum((prob - float(label)) ** 2 for label, prob in zip(labels, probabilities, strict=True))
            / len(labels),
            4,
        ),
        roc_auc=_roc_auc(labels, probabilities),
        confusion_matrix={"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    )


def _improvement(baseline: BacktestMetrics, enriched: BacktestMetrics) -> dict[str, float | None]:
    auc_delta = None
    if baseline.roc_auc is not None and enriched.roc_auc is not None:
        auc_delta = round(enriched.roc_auc - baseline.roc_auc, 4)
    return {
        "accuracy_delta": round(enriched.accuracy - baseline.accuracy, 4),
        "precision_delta": round(enriched.precision - baseline.precision, 4),
        "recall_delta": round(enriched.recall - baseline.recall, 4),
        "f1_delta": round(enriched.f1 - baseline.f1, 4),
        "false_positive_rate_delta": round(
            enriched.false_positive_rate - baseline.false_positive_rate,
            4,
        ),
        "brier_score_delta": round(enriched.brier_score - baseline.brier_score, 4),
        "roc_auc_delta": auc_delta,
    }


def _roc_auc(labels: list[bool], probabilities: list[float]) -> float | None:
    positives = [prob for label, prob in zip(labels, probabilities, strict=True) if label]
    negatives = [prob for label, prob in zip(labels, probabilities, strict=True) if not label]
    if not positives or not negatives:
        return None

    wins = 0.0
    total = len(positives) * len(negatives)
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return round(wins / total, 4)


def _first(row: dict[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] not in ("", None):
            return row[name]
    return default


def _float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric, got {value!r}") from exc


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    return str(value).strip().lower() in _TRUE_VALUES


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_").replace("-", "_")


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
