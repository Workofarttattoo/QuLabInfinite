"""
Ensemble Hail Prediction — XGBoost + Random Forest with stacking and
weighted-voting meta-learners.

Supports both classification (hail_occurred: 0/1) and regression
(hail_size_inches: continuous).

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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .preprocess import HailDataPreprocessor, load_config

logger = logging.getLogger(__name__)


class EnsembleTrainer:
    """Train an ensemble of XGBoost + Random Forest with a meta-learner."""

    def __init__(self, config: dict[str, Any] | None = None, config_path: str | None = None):
        if config is None:
            if config_path is None:
                config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
            config = load_config(config_path)
        self.config = config
        self.preprocessor = HailDataPreprocessor(config)
        self.target_column: str = config["data"]["target_column"]
        self.feature_columns: list[str] = list(config["data"]["feature_columns"])
        self.is_classification = self.target_column == "hail_occurred"

        self.xgb_model: xgb.Booster | None = None
        self.rf_model: RandomForestClassifier | RandomForestRegressor | None = None
        self.meta_model: Any = None
        self.ensemble_method: str = "stacking"
        self.voting_weights: list[float] = [0.6, 0.4]

    # ------------------------------------------------------------------
    # Base model training
    # ------------------------------------------------------------------

    def _train_xgboost(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
    ) -> xgb.Booster:
        params = {
            k: v for k, v in self.config["xgboost"].items()
            if k not in ("n_estimators", "early_stopping_rounds")
        }
        if not self.is_classification:
            params["objective"] = "reg:squarederror"
            params["eval_metric"] = "rmse"
            params.pop("scale_pos_weight", None)

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=list(X_val.columns))

        self.xgb_model = xgb.train(
            params, dtrain,
            num_boost_round=self.config["xgboost"]["n_estimators"],
            evals=[(dtrain, "train"), (dval, "eval")],
            early_stopping_rounds=self.config["xgboost"]["early_stopping_rounds"],
            verbose_eval=False,
        )
        logger.info("XGBoost trained (best iter %d)", self.xgb_model.best_iteration)
        return self.xgb_model

    def _train_random_forest(
        self, X_train: pd.DataFrame, y_train: pd.Series,
    ) -> RandomForestClassifier | RandomForestRegressor:
        rf_params = {
            "n_estimators": 300,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "random_state": self.config["data"]["random_state"],
            "n_jobs": -1,
        }
        if self.is_classification:
            self.rf_model = RandomForestClassifier(class_weight="balanced", **rf_params)
        else:
            self.rf_model = RandomForestRegressor(**rf_params)

        self.rf_model.fit(X_train, y_train)
        logger.info("Random Forest trained (%d estimators)", rf_params["n_estimators"])
        return self.rf_model

    # ------------------------------------------------------------------
    # Meta-feature generation
    # ------------------------------------------------------------------

    def _xgb_predict(self, X: pd.DataFrame) -> np.ndarray:
        dm = xgb.DMatrix(X, feature_names=list(X.columns))
        return self.xgb_model.predict(dm)

    def _rf_predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.is_classification and hasattr(self.rf_model, "predict_proba"):
            return self.rf_model.predict_proba(X)[:, 1]
        return self.rf_model.predict(X)

    def _meta_features(self, X: pd.DataFrame) -> np.ndarray:
        xgb_pred = self._xgb_predict(X)
        rf_pred = self._rf_predict(X)
        return np.column_stack([xgb_pred, rf_pred])

    # ------------------------------------------------------------------
    # Meta-learner training
    # ------------------------------------------------------------------

    def _train_meta_model(
        self, X_meta: np.ndarray, y: pd.Series,
    ) -> Any:
        if self.is_classification:
            self.meta_model = LogisticRegression(max_iter=1000)
        else:
            self.meta_model = LinearRegression()
        self.meta_model.fit(X_meta, y)
        logger.info("Meta-model trained (%s)", type(self.meta_model).__name__)
        return self.meta_model

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series, method: str = "stacking",
    ) -> dict[str, float]:
        if method == "stacking":
            meta = self._meta_features(X_test)
            y_pred_raw = self.meta_model.predict(meta)
            if self.is_classification and hasattr(self.meta_model, "predict_proba"):
                y_proba = self.meta_model.predict_proba(meta)[:, 1]
            else:
                y_proba = y_pred_raw
        else:
            xgb_p = self._xgb_predict(X_test)
            rf_p = self._rf_predict(X_test)
            w = self.voting_weights
            y_proba = w[0] * xgb_p + w[1] * rf_p
            y_pred_raw = y_proba

        if self.is_classification:
            y_pred = (y_proba >= 0.5).astype(int)
            return {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, y_proba)),
            }
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_raw))),
            "mae": float(mean_absolute_error(y_test, y_pred_raw)),
            "r2": float(r2_score(y_test, y_pred_raw)),
        }

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def train_pipeline(
        self,
        synthetic: bool = True,
        n_synthetic: int = 2000,
        method: str = "stacking",
    ) -> dict[str, float]:
        self.ensemble_method = method

        if synthetic:
            full = self.preprocessor.generate_synthetic_data(n_synthetic)
            full = self.preprocessor.preprocess_dataframe(full)
        else:
            raise NotImplementedError("Provide nexrad_path for real data")

        for col in self.feature_columns:
            if col not in full.columns:
                full[col] = 0.0

        X = full[self.feature_columns].astype(np.float32)
        y = full[self.target_column].astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config["data"]["test_size"],
            random_state=self.config["data"]["random_state"],
            stratify=y if self.is_classification else None,
        )
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=0.15,
            random_state=self.config["data"]["random_state"],
            stratify=y_train if self.is_classification else None,
        )

        logger.info("Training XGBoost...")
        self._train_xgboost(X_tr, y_tr, X_val, y_val)

        logger.info("Training Random Forest...")
        self._train_random_forest(X_tr, y_tr)

        if method == "stacking":
            meta_train = self._meta_features(X_val)
            self._train_meta_model(meta_train, y_val)

        metrics = self.evaluate(X_test, y_test, method)
        logger.info("Ensemble (%s) metrics: %s", method, metrics)
        return metrics

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, model_dir: str | Path | None = None) -> Path:
        model_dir = Path(model_dir or self.config["paths"]["model_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)

        self.xgb_model.save_model(str(model_dir / "ensemble_xgb.json"))

        import joblib
        joblib.dump(self.rf_model, model_dir / "ensemble_rf.joblib")
        if self.meta_model is not None:
            joblib.dump(self.meta_model, model_dir / "ensemble_meta.joblib")

        meta = {
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "is_classification": self.is_classification,
            "ensemble_method": self.ensemble_method,
            "voting_weights": self.voting_weights,
        }
        (model_dir / "ensemble_meta.json").write_text(json.dumps(meta, indent=2))
        logger.info("Ensemble saved to %s", model_dir)
        return model_dir

    @classmethod
    def load(cls, model_dir: str | Path, config: dict[str, Any] | None = None) -> EnsembleTrainer:
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "ensemble_meta.json").read_text())

        if config is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
            config = load_config(config_path)

        obj = cls(config)
        obj.feature_columns = meta["feature_columns"]
        obj.target_column = meta["target_column"]
        obj.is_classification = meta["is_classification"]
        obj.ensemble_method = meta["ensemble_method"]
        obj.voting_weights = meta["voting_weights"]

        obj.xgb_model = xgb.Booster()
        obj.xgb_model.load_model(str(model_dir / "ensemble_xgb.json"))

        import joblib
        obj.rf_model = joblib.load(model_dir / "ensemble_rf.joblib")
        meta_path = model_dir / "ensemble_meta.joblib"
        if meta_path.exists():
            obj.meta_model = joblib.load(meta_path)

        logger.info("Ensemble loaded from %s", model_dir)
        return obj


class EnsemblePredictor:
    """Lightweight prediction wrapper around a trained ensemble."""

    def __init__(self, model_dir: str | Path, config: dict[str, Any] | None = None):
        self.ensemble = EnsembleTrainer.load(model_dir, config)
        self.preprocessor = self.ensemble.preprocessor

    def _prepare(self, data: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()
        df = self.preprocessor.add_derived_features(df)
        df = self.preprocessor.clean_data(df)
        for col in self.ensemble.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        return df[self.ensemble.feature_columns].astype(np.float32)

    def predict_proba(self, data: dict[str, Any] | pd.DataFrame) -> np.ndarray:
        X = self._prepare(data)
        e = self.ensemble
        if e.ensemble_method == "stacking" and e.meta_model is not None:
            meta = e._meta_features(X)
            if hasattr(e.meta_model, "predict_proba"):
                return e.meta_model.predict_proba(meta)[:, 1]
            return e.meta_model.predict(meta)
        xgb_p = e._xgb_predict(X)
        rf_p = e._rf_predict(X)
        w = e.voting_weights
        return w[0] * xgb_p + w[1] * rf_p

    def predict(self, data: dict[str, Any] | pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        raw = self.predict_proba(data)
        if self.ensemble.is_classification:
            return (raw >= threshold).astype(int)
        return raw

    def predict_full(self, data: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
        raw = self.predict_proba(data)
        if self.ensemble.is_classification:
            p = float(raw[0]) if raw.ndim else float(raw)
            return {
                "hail_probability": round(p, 4),
                "hail_predicted": p >= 0.5,
                "risk_level": _risk(p),
                "mode": "classification",
            }
        size = float(raw[0]) if raw.ndim else float(raw)
        return {
            "predicted_size_inches": round(max(0.0, size), 2),
            "hail_predicted": size >= 1.0,
            "risk_level": _risk_from_size(size),
            "mode": "regression",
        }


def _risk(p: float) -> str:
    if p >= 0.8:
        return "EXTREME"
    if p >= 0.6:
        return "HIGH"
    if p >= 0.4:
        return "MODERATE"
    if p >= 0.2:
        return "LOW"
    return "MINIMAL"


def _risk_from_size(inches: float) -> str:
    if inches >= 2.5:
        return "EXTREME"
    if inches >= 1.75:
        return "HIGH"
    if inches >= 1.0:
        return "MODERATE"
    if inches >= 0.5:
        return "LOW"
    return "MINIMAL"
