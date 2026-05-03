"""
XGBoost Hail Prediction Model for QuLabInfinite.

Production-ready hail occurrence and size prediction using dual-pol radar
features, environmental parameters, and NOAA historical data.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from .predict import HailPredictor
from .preprocess import HailDataPreprocessor
from .train import HailModelTrainer
from .validate import validate_model

__all__ = [
    "HailDataPreprocessor",
    "HailModelTrainer",
    "HailPredictor",
    "validate_model",
]
