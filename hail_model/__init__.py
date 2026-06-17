"""
XGBoost Hail Prediction Model for QuLabInfinite.

Production-ready hail occurrence and size prediction using dual-pol radar
features, environmental parameters, and NOAA historical data.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from .azure_digital_twins import (
    AzureDigitalTwinsPublisher,
    RoofHunterWeatherSimulator,
    RoofProfile,
    TwinModelFactory,
    WeatherSimulationResult,
    WeatherSnapshot,
)
from .backtest import (
    BacktestMetrics,
    BacktestRecord,
    BacktestReport,
    PredictionComparison,
    RoofHunterBacktester,
    load_backtest_records,
    record_from_mapping,
)
from .dual_pol import (
    DualPolObservation,
    HailEstimate,
    HydrometeorType,
    classify_hydrometeor,
    compute_mesh,
    compute_posh,
    estimate_hail_size,
)
from .ensemble import EnsemblePredictor as EnsemblePredictor
from .ensemble import EnsembleTrainer as EnsembleTrainer
from .nexrad_fetcher import NEXRADFetcher, RadarObservation, get_radar_features_for_property
from .predict import HailPredictor
from .preprocess import HailDataPreprocessor
from .roof_hunter_bridge import HailIntelligence, PropertyAssessment
from .train import HailModelTrainer
from .validate import validate_model

__all__ = [
    "EnsemblePredictor",
    "EnsembleTrainer",
    "AzureDigitalTwinsPublisher",
    "BacktestMetrics",
    "BacktestRecord",
    "BacktestReport",
    "DualPolObservation",
    "HailDataPreprocessor",
    "HailEstimate",
    "HailIntelligence",
    "HailModelTrainer",
    "HailPredictor",
    "HydrometeorType",
    "NEXRADFetcher",
    "PropertyAssessment",
    "PredictionComparison",
    "RadarObservation",
    "RoofHunterBacktester",
    "RoofHunterWeatherSimulator",
    "RoofProfile",
    "TwinModelFactory",
    "WeatherSimulationResult",
    "WeatherSnapshot",
    "classify_hydrometeor",
    "compute_mesh",
    "compute_posh",
    "estimate_hail_size",
    "get_radar_features_for_property",
    "load_backtest_records",
    "record_from_mapping",
    "validate_model",
]
