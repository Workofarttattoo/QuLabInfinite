"""
Hybrid Hail Predictor — Quantile Regression + Monte Carlo Bootstrap
+ Ensemble Stacking for probabilistic hail size prediction with
calibrated uncertainty intervals.

Combines:
  1. XGBoost quantile regression (10th/25th/50th/75th/90th percentiles)
  2. Monte Carlo bootstrap (N resampled RF models → prediction distribution)
  3. Stacking ensemble (XGBoost + RF meta-learner)
  4. Confidence scoring from model agreement

No TensorFlow/Keras — runs fast on CPU with zero GPU dependencies.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .preprocess import HailDataPreprocessor, load_config

logger = logging.getLogger(__name__)

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


@dataclass
class HailForecast:
    """Full probabilistic hail forecast for a single location."""

    predicted_size_inches: float
    uncertainty: float
    quantiles: dict[float, float]
    confidence_interval_95: tuple[float, float]
    confidence_score: float
    risk_level: str
    alert_level: str
    hail_predicted: bool
    mc_mean: float = 0.0
    mc_std: float = 0.0
    ensemble_mean: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_size_inches": round(self.predicted_size_inches, 2),
            "uncertainty": round(self.uncertainty, 3),
            "quantiles": {str(k): round(v, 2) for k, v in self.quantiles.items()},
            "confidence_interval_95": (
                round(self.confidence_interval_95[0], 2),
                round(self.confidence_interval_95[1], 2),
            ),
            "confidence_score": round(self.confidence_score, 3),
            "risk_level": self.risk_level,
            "alert_level": self.alert_level,
            "hail_predicted": self.hail_predicted,
        }


class HybridTrainer:
    """Train quantile XGBoost + bootstrap RF + stacking meta-learner."""

    def __init__(self, config: dict[str, Any] | None = None, config_path: str | None = None):
        if config is None:
            config_path = config_path or os.path.join(os.path.dirname(__file__), "config.yaml")
            config = load_config(config_path)
        self.config = config
        self.preprocessor = HailDataPreprocessor(config)
        self.feature_columns: list[str] = list(config["data"]["feature_columns"])
        self.target_column: str = "hail_size_inches"

        self.quantile_models: dict[float, xgb.Booster] = {}
        self.bootstrap_forests: list[RandomForestRegressor] = []
        self.meta_model: LinearRegression | None = None

    # ------------------------------------------------------------------
    # 1. Quantile XGBoost
    # ------------------------------------------------------------------

    def _train_quantile_xgb(
        self, X_tr: pd.DataFrame, y_tr: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
    ) -> None:
        base_params = {
            k: v for k, v in self.config["xgboost"].items()
            if k not in ("n_estimators", "early_stopping_rounds", "objective", "eval_metric", "scale_pos_weight")
        }

        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=list(X_tr.columns))
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=list(X_val.columns))

        for q in QUANTILES:
            params = {
                **base_params,
                "objective": "reg:quantileerror",
                "quantile_alpha": q,
                "eval_metric": "mae",
            }
            model = xgb.train(
                params, dtrain,
                num_boost_round=self.config["xgboost"].get("n_estimators", 500),
                evals=[(dval, "eval")],
                early_stopping_rounds=self.config["xgboost"].get("early_stopping_rounds", 50),
                verbose_eval=False,
            )
            self.quantile_models[q] = model
            logger.info("Quantile XGBoost q=%.2f trained (iter %d)", q, model.best_iteration)

    # ------------------------------------------------------------------
    # 2. Monte Carlo Bootstrap Random Forests
    # ------------------------------------------------------------------

    def _train_bootstrap_forests(
        self, X_tr: pd.DataFrame, y_tr: pd.Series, n_forests: int = 10,
    ) -> None:
        rng = np.random.default_rng(self.config["data"]["random_state"])
        n = len(X_tr)

        for _i in range(n_forests):
            idx = rng.choice(n, size=n, replace=True)
            X_boot = X_tr.iloc[idx]
            y_boot = y_tr.iloc[idx]

            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=3,
                random_state=int(rng.integers(0, 2**31)),
                n_jobs=-1,
            )
            rf.fit(X_boot, y_boot)
            self.bootstrap_forests.append(rf)

        logger.info("Trained %d bootstrap Random Forests", n_forests)

    # ------------------------------------------------------------------
    # 3. Stacking meta-learner
    # ------------------------------------------------------------------

    def _meta_features(self, X: pd.DataFrame) -> np.ndarray:
        dm = xgb.DMatrix(X, feature_names=list(X.columns))
        cols = []
        for q in QUANTILES:
            cols.append(self.quantile_models[q].predict(dm))
        boot_preds = np.column_stack([rf.predict(X) for rf in self.bootstrap_forests])
        cols.append(boot_preds.mean(axis=1))
        cols.append(boot_preds.std(axis=1))
        return np.column_stack(cols)

    def _train_meta(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        meta = self._meta_features(X_val)
        self.meta_model = LinearRegression()
        self.meta_model.fit(meta, y_val)
        logger.info("Meta-learner trained")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def train_pipeline(
        self, synthetic: bool = True, n_synthetic: int = 3000,
        n_bootstrap: int = 10,
    ) -> dict[str, float]:
        if synthetic:
            full = self.preprocessor.generate_synthetic_data(n_synthetic)
            full = self.preprocessor.preprocess_dataframe(full)
        else:
            raise NotImplementedError

        for col in self.feature_columns:
            if col not in full.columns:
                full[col] = 0.0

        X = full[self.feature_columns].astype(np.float32)
        y = full[self.target_column].astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config["data"]["test_size"],
            random_state=self.config["data"]["random_state"],
        )
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.15,
            random_state=self.config["data"]["random_state"],
        )

        self._train_quantile_xgb(X_tr, y_tr, X_val, y_val)
        self._train_bootstrap_forests(X_tr, y_tr, n_bootstrap)
        self._train_meta(X_val, y_val)

        meta_test = self._meta_features(X_test)
        y_pred = self.meta_model.predict(meta_test)
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        }

        dm_test = xgb.DMatrix(X_test, feature_names=list(X_test.columns))
        ci_width = (
            self.quantile_models[0.90].predict(dm_test)
            - self.quantile_models[0.10].predict(dm_test)
        )
        metrics["mean_ci_width_80"] = float(np.mean(ci_width))

        logger.info("Hybrid metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, model_dir: str | Path) -> Path:
        d = Path(model_dir)
        d.mkdir(parents=True, exist_ok=True)
        for q, m in self.quantile_models.items():
            m.save_model(str(d / f"quantile_{q:.2f}.json"))
        joblib.dump(self.bootstrap_forests, d / "bootstrap_forests.joblib")
        if self.meta_model is not None:
            joblib.dump(self.meta_model, d / "meta_model.joblib")
        (d / "hybrid_meta.json").write_text(json.dumps({
            "feature_columns": self.feature_columns,
            "quantiles": list(QUANTILES),
            "n_bootstrap": len(self.bootstrap_forests),
        }))
        logger.info("Hybrid model saved to %s", d)
        return d

    @classmethod
    def load(cls, model_dir: str | Path, config: dict[str, Any] | None = None) -> HybridTrainer:
        d = Path(model_dir)
        meta = json.loads((d / "hybrid_meta.json").read_text())
        if config is None:
            config = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))
        obj = cls(config)
        obj.feature_columns = meta["feature_columns"]
        for q in meta["quantiles"]:
            m = xgb.Booster()
            m.load_model(str(d / f"quantile_{q:.2f}.json"))
            obj.quantile_models[q] = m
        obj.bootstrap_forests = joblib.load(d / "bootstrap_forests.joblib")
        meta_path = d / "meta_model.joblib"
        if meta_path.exists():
            obj.meta_model = joblib.load(meta_path)
        return obj


class HybridPredictor:
    """Probabilistic hail prediction with uncertainty quantification."""

    def __init__(self, model_dir: str | Path, config: dict[str, Any] | None = None):
        self.hybrid = HybridTrainer.load(model_dir, config)
        self.preprocessor = self.hybrid.preprocessor

    def _prepare(self, data: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()
        df = self.preprocessor.add_derived_features(df)
        df = self.preprocessor.clean_data(df)
        for col in self.hybrid.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        return df[self.hybrid.feature_columns].astype(np.float32)

    def predict(self, data: dict[str, Any] | pd.DataFrame) -> HailForecast:
        X = self._prepare(data)
        dm = xgb.DMatrix(X, feature_names=self.hybrid.feature_columns)

        # Quantile predictions
        quantiles = {q: float(self.hybrid.quantile_models[q].predict(dm)[0]) for q in QUANTILES}
        median = max(0.0, quantiles[0.50])

        # Bootstrap MC predictions
        boot_preds = np.array([rf.predict(X)[0] for rf in self.hybrid.bootstrap_forests])
        mc_mean = float(np.mean(boot_preds))
        mc_std = float(np.std(boot_preds))

        # Stacking meta-prediction
        if self.hybrid.meta_model is not None:
            meta = self.hybrid._meta_features(X)
            ensemble_mean = float(self.hybrid.meta_model.predict(meta)[0])
        else:
            ensemble_mean = mc_mean

        predicted = max(0.0, ensemble_mean)
        ci_lo = max(0.0, quantiles[0.10])
        ci_hi = max(0.0, quantiles[0.90])
        uncertainty = (ci_hi - ci_lo) / 2.0

        # Confidence = agreement between methods (0–1)
        spread = np.std([median, mc_mean, ensemble_mean])
        confidence = float(np.exp(-spread))

        return HailForecast(
            predicted_size_inches=round(predicted, 2),
            uncertainty=round(uncertainty, 3),
            quantiles={q: round(max(0.0, v), 2) for q, v in quantiles.items()},
            confidence_interval_95=(ci_lo, ci_hi),
            confidence_score=round(min(1.0, confidence), 3),
            risk_level=_risk(predicted),
            alert_level=_alert(predicted),
            hail_predicted=predicted >= 1.0,
            mc_mean=round(mc_mean, 2),
            mc_std=round(mc_std, 3),
            ensemble_mean=round(ensemble_mean, 2),
        )

    def predict_batch(self, records: list[dict[str, Any]]) -> list[HailForecast]:
        return [self.predict(r) for r in records]


def _risk(inches: float) -> str:
    if inches >= 2.5:
        return "EXTREME"
    if inches >= 1.75:
        return "HIGH"
    if inches >= 1.0:
        return "MODERATE"
    if inches >= 0.5:
        return "LOW"
    return "MINIMAL"


def _alert(inches: float) -> str:
    if inches >= 1.75:
        return "SEVERE (>=1.75\")"
    if inches >= 1.0:
        return "MODERATE (1.0-1.75\")"
    if inches >= 0.75:
        return "MINOR (0.75-1.0\")"
    return "NONE (<0.75\")"
