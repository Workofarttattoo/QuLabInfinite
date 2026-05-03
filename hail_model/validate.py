"""
Model Validation — evaluate a trained hail model against held-out data.

Reports classification metrics, feature importance, and optionally
generates a per-threshold precision-recall table.

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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .preprocess import HailDataPreprocessor, load_config

logger = logging.getLogger(__name__)


def validate_model(
    model_path: str | Path,
    test_data: pd.DataFrame | str | Path | None = None,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """Evaluate a saved model and return a comprehensive metrics report.

    Parameters
    ----------
    model_path : path to the saved XGBoost JSON model
    test_data : either a DataFrame or a path to a CSV with labelled test data.
        If *None*, synthetic test data is generated.
    config / config_path : model configuration
    thresholds : list of classification thresholds to sweep (default 0.3–0.9)
    """
    if config is None:
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        config = load_config(config_path)

    preprocessor = HailDataPreprocessor(config)
    feature_columns: list[str] = list(config["data"]["feature_columns"])
    target_column: str = config["data"]["target_column"]

    # Load model
    model = xgb.Booster()
    model.load_model(str(model_path))

    meta_path = Path(model_path).with_suffix(".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        feature_columns = meta.get("feature_columns", feature_columns)

    # Load or generate test data
    if test_data is None:
        logger.info("No test data supplied — generating synthetic data")
        full = preprocessor.generate_synthetic_data(n_samples=1000)
        test_df = preprocessor.preprocess_dataframe(full)
    elif isinstance(test_data, (str, Path)):
        test_df = pd.read_csv(test_data)
        test_df = preprocessor.preprocess_dataframe(test_df)
    else:
        test_df = preprocessor.preprocess_dataframe(test_data)

    for col in feature_columns:
        if col not in test_df.columns:
            test_df[col] = 0.0

    X_test = test_df[feature_columns].astype(np.float32)
    y_test = test_df[target_column].astype(np.float32)

    dtest = xgb.DMatrix(X_test, feature_names=feature_columns)
    y_proba = model.predict(dtest)

    report: dict[str, Any] = {"model_path": str(model_path)}

    if target_column == "hail_occurred":
        y_pred = (y_proba >= 0.5).astype(int)
        report["metrics"] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        }

        cm = confusion_matrix(y_test, y_pred)
        report["confusion_matrix"] = cm.tolist()

        report["classification_report"] = classification_report(
            y_test, y_pred, target_names=["no_hail", "hail"], output_dict=True
        )

        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        threshold_sweep = []
        for t in thresholds:
            yp = (y_proba >= t).astype(int)
            threshold_sweep.append({
                "threshold": t,
                "precision": float(precision_score(y_test, yp, zero_division=0)),
                "recall": float(recall_score(y_test, yp, zero_division=0)),
                "f1": float(f1_score(y_test, yp, zero_division=0)),
            })
        report["threshold_sweep"] = threshold_sweep
    else:
        report["metrics"] = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_proba))),
            "mae": float(mean_absolute_error(y_test, y_proba)),
        }

    # Feature importance
    importance = model.get_score(importance_type="gain")
    report["feature_importance"] = dict(
        sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    )

    return report


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

    report = validate_model(model_file)

    print("\n=== Validation Report ===")
    for k, v in report["metrics"].items():
        print(f"  {k}: {v:.4f}")

    print("\nConfusion Matrix:")
    for row in report.get("confusion_matrix", []):
        print(f"  {row}")

    print("\nThreshold Sweep:")
    for entry in report.get("threshold_sweep", []):
        print(
            f"  t={entry['threshold']:.1f}  "
            f"P={entry['precision']:.3f}  "
            f"R={entry['recall']:.3f}  "
            f"F1={entry['f1']:.3f}"
        )

    print("\nTop Features (gain):")
    for feat, score in list(report["feature_importance"].items())[:10]:
        print(f"  {feat}: {score:.2f}")
