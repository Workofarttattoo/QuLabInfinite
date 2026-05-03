"""
Hail Predictor — real-time inference for single properties or batches.

Loads a trained XGBoost model and returns hail probability, binary
prediction, and estimated hail size.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from .preprocess import HailDataPreprocessor, load_config

logger = logging.getLogger(__name__)


class HailPredictor:
    """Lightweight inference wrapper around a trained XGBoost model."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        config: dict[str, Any] | None = None,
        config_path: str | None = None,
    ):
        if config is None:
            if config_path is None:
                config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
            config = load_config(config_path)
        self.config = config
        self.preprocessor = HailDataPreprocessor(config)
        self.model: xgb.Booster | None = None
        self.feature_columns: list[str] = list(config["data"]["feature_columns"])
        self.target_column: str = config["data"]["target_column"]

        if model_path is not None:
            self.load_model(model_path)

    # ------------------------------------------------------------------
    # Model I/O
    # ------------------------------------------------------------------

    def load_model(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))

        meta_path = model_path.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.feature_columns = meta.get("feature_columns", self.feature_columns)
            self.target_column = meta.get("target_column", self.target_column)

        logger.info("Loaded model from %s (%d features)", model_path, len(self.feature_columns))

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def _to_dmatrix(self, data: dict[str, Any] | pd.DataFrame) -> xgb.DMatrix:
        """Convert raw input into an XGBoost DMatrix."""
        df = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()

        df = self.preprocessor.add_derived_features(df)
        df = self.preprocessor.clean_data(df)

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0

        X = df[self.feature_columns].astype(np.float32)
        return xgb.DMatrix(X, feature_names=self.feature_columns)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, data: dict[str, Any] | pd.DataFrame) -> np.ndarray:
        """Return raw hail probability (0–1) for each sample."""
        if self.model is None:
            raise RuntimeError("No model loaded — call load_model() first")
        dm = self._to_dmatrix(data)
        return self.model.predict(dm)

    def predict(
        self, data: dict[str, Any] | pd.DataFrame, threshold: float = 0.5
    ) -> np.ndarray:
        """Return binary hail prediction (0 or 1)."""
        proba = self.predict_proba(data)
        return (proba >= threshold).astype(int)

    def predict_full(
        self, data: dict[str, Any] | pd.DataFrame, threshold: float = 0.5
    ) -> dict[str, Any]:
        """Return a rich prediction dict with probability, binary, and risk level."""
        proba = self.predict_proba(data)
        binary = (proba >= threshold).astype(int)

        def _risk_level(p: float) -> str:
            if p >= 0.8:
                return "EXTREME"
            if p >= 0.6:
                return "HIGH"
            if p >= 0.4:
                return "MODERATE"
            if p >= 0.2:
                return "LOW"
            return "MINIMAL"

        if proba.ndim == 0 or len(proba) == 1:
            p = float(proba.item() if proba.ndim == 0 else proba[0])
            return {
                "hail_probability": round(p, 4),
                "hail_predicted": bool(binary.item() if binary.ndim == 0 else binary[0]),
                "risk_level": _risk_level(p),
            }

        return {
            "predictions": [
                {
                    "hail_probability": round(float(p), 4),
                    "hail_predicted": bool(b),
                    "risk_level": _risk_level(float(p)),
                }
                for p, b in zip(proba, binary, strict=True)
            ]
        }

    def predict_batch(
        self, records: list[dict[str, Any]], threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        """Convenience wrapper for multiple single-record predictions."""
        df = pd.DataFrame(records)
        proba = self.predict_proba(df)
        binary = (proba >= threshold).astype(int)

        def _risk_level(p: float) -> str:
            if p >= 0.8:
                return "EXTREME"
            if p >= 0.6:
                return "HIGH"
            if p >= 0.4:
                return "MODERATE"
            if p >= 0.2:
                return "LOW"
            return "MINIMAL"

        return [
            {
                "hail_probability": round(float(p), 4),
                "hail_predicted": bool(b),
                "risk_level": _risk_level(float(p)),
            }
            for p, b in zip(proba, binary, strict=True)
        ]


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model_dir = os.path.join(os.path.dirname(__file__), "models")
    model_file = os.path.join(model_dir, "xgboost_hail.json")

    if not os.path.exists(model_file):
        print("No trained model found. Run train.py first.")
        raise SystemExit(1)

    predictor = HailPredictor(model_path=model_file)

    sample = {
        "reflectivity_max": 62.0,
        "reflectivity_mean": 55.0,
        "reflectivity_std": 8.0,
        "differential_reflectivity": 3.5,
        "correlation_coefficient": 0.91,
        "specific_differential_phase": 2.5,
        "velocity_max": 30.0,
        "velocity_mean": 15.0,
        "spectrum_width_mean": 9.0,
        "cape": 3200.0,
        "shear_0_6km": 55.0,
        "temp_500mb": -18.0,
        "freezing_level_m": 3000.0,
        "vil": 48.0,
        "echo_top_km": 14.0,
        "storm_relative_helicity": 280.0,
        "latitude": 35.5,
        "longitude": -97.0,
        "time": "2026-05-02T14:30:00",
    }

    result = predictor.predict_full(sample)
    print(f"\nHail probability : {result['hail_probability']:.2%}")
    print(f"Hail predicted   : {result['hail_predicted']}")
    print(f"Risk level       : {result['risk_level']}")
