"""
XGBoost Hail Model Trainer — training pipeline with early stopping and
optional Optuna hyperparameter tuning.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .preprocess import HailDataPreprocessor, load_config

logger = logging.getLogger(__name__)


class HailModelTrainer:
    """Trains, evaluates, and serialises an XGBoost hail-prediction model."""

    def __init__(self, config: dict[str, Any] | None = None, config_path: str | None = None):
        if config is None:
            if config_path is None:
                config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
            config = load_config(config_path)
        self.config = config
        self.preprocessor = HailDataPreprocessor(config)
        self.model: xgb.Booster | None = None
        self.feature_columns: list[str] = list(config["data"]["feature_columns"])
        self.target_column: str = config["data"]["target_column"]

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_features(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Select feature columns and target from a preprocessed DataFrame."""
        available = [c for c in self.feature_columns if c in df.columns]
        missing = set(self.feature_columns) - set(available)
        if missing:
            logger.warning("Missing feature columns (will be zero-filled): %s", missing)
            for col in missing:
                df[col] = 0.0
            available = self.feature_columns

        X = df[available].astype(np.float32)
        y = df[self.target_column].astype(np.float32)
        return X, y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> xgb.Booster:
        """Train the XGBoost model with early stopping."""
        params = {
            k: v
            for k, v in self.config["xgboost"].items()
            if k not in ("n_estimators", "early_stopping_rounds")
        }

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=list(X_test.columns))

        evals = [(dtrain, "train"), (dtest, "eval")]
        n_rounds = self.config["xgboost"]["n_estimators"]
        early_stop = self.config["xgboost"]["early_stopping_rounds"]

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_rounds,
            evals=evals,
            early_stopping_rounds=early_stop,
            verbose_eval=50,
        )

        logger.info("Training complete — best iteration: %d", self.model.best_iteration)
        return self.model

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5
    ) -> dict[str, float]:
        """Compute classification or regression metrics."""
        if self.model is None:
            raise RuntimeError("Model not trained — call train() first")

        dtest = xgb.DMatrix(X_test, feature_names=list(X_test.columns))
        y_proba = self.model.predict(dtest)

        if self.target_column == "hail_occurred":
            y_pred = (y_proba >= threshold).astype(int)
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, y_proba)),
            }
        else:
            metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_proba))),
                "mae": float(mean_absolute_error(y_test, y_proba)),
            }

        return metrics

    def feature_importance(self, importance_type: str = "gain") -> dict[str, float]:
        """Return feature-importance scores."""
        if self.model is None:
            raise RuntimeError("No model loaded")
        return dict(self.model.get_score(importance_type=importance_type))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save_model(
        self,
        model_dir: str | Path | None = None,
        model_name: str | None = None,
    ) -> Path:
        """Save the model as a portable XGBoost JSON file."""
        if self.model is None:
            raise RuntimeError("No model to save")

        model_dir = Path(model_dir or self.config["paths"]["model_dir"])
        model_name = model_name or self.config["paths"]["model_name"]
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / model_name
        self.model.save_model(str(path))

        import json
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps({
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "best_iteration": self.model.best_iteration,
        }))

        logger.info("Saved model to %s (+ metadata %s)", path, meta_path)
        return path

    def load_model(self, model_path: str | Path) -> None:
        """Load a previously saved model."""
        import json

        model_path = Path(model_path)
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))

        meta_path = model_path.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.feature_columns = meta.get("feature_columns", self.feature_columns)
            self.target_column = meta.get("target_column", self.target_column)

        logger.info("Loaded model from %s", model_path)

    # ------------------------------------------------------------------
    # End-to-end pipeline
    # ------------------------------------------------------------------

    def train_pipeline(
        self,
        nexrad_path: str | Path | None = None,
        spc_path: str | Path | None = None,
        mesonet_path: str | Path | None = None,
        synthetic: bool = False,
        n_synthetic: int = 2000,
    ) -> dict[str, float]:
        """Run the full train → evaluate → save pipeline.

        When *synthetic=True* (or no data paths given), generates
        synthetic training data for testing the pipeline end-to-end.
        """
        if synthetic or nexrad_path is None:
            logger.info("Generating %d synthetic samples", n_synthetic)
            full_df = self.preprocessor.generate_synthetic_data(n_synthetic)
            full_df = self.preprocessor.preprocess_dataframe(full_df)
            stratify = full_df[self.target_column]
            train_df, test_df = pd.DataFrame(), pd.DataFrame()
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(
                full_df,
                test_size=self.config["data"]["test_size"],
                random_state=self.config["data"]["random_state"],
                stratify=stratify,
            )
        else:
            train_df, test_df = self.preprocessor.preprocess(
                nexrad_path, spc_path, mesonet_path
            )

        X_train, y_train = self.prepare_features(train_df)
        X_test, y_test = self.prepare_features(test_df)

        self.train(X_train, y_train, X_test, y_test)

        metrics = self.evaluate(X_test, y_test)
        logger.info("Evaluation metrics: %s", metrics)

        self.save_model()
        return metrics

    # ------------------------------------------------------------------
    # Optuna hyperparameter tuning
    # ------------------------------------------------------------------

    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_trials: int = 50,
    ) -> dict[str, Any]:
        """Bayesian hyperparameter optimisation via Optuna."""
        import optuna

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=list(X_test.columns))

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
            }
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dtest, "eval")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            y_pred = model.predict(dtest)
            return float(roc_auc_score(y_test, y_pred))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_trial
        logger.info("Best AUC: %.4f  params=%s", best.value, best.params)
        return {"best_auc": best.value, "best_params": best.params}


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    trainer = HailModelTrainer()
    metrics = trainer.train_pipeline(synthetic=True, n_synthetic=3000)
    print("\n=== Training Complete ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
