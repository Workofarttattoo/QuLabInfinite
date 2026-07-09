"""Train a baseline roof replacement damage classifier from labeled leads.

This baseline is tabular (metadata + event context) and is intended as a fast
starting point while satellite chip CV labeling matures.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _keyword_count(text: str, words: Tuple[str, ...]) -> int:
    t = (text or "").lower()
    return sum(1 for w in words if w in t)


def _feature_frame(lead_df: pd.DataFrame) -> pd.DataFrame:
    df = lead_df.copy()
    df["report_type"] = df.get("report_type", "").fillna("").astype(str).str.lower()
    df["property_segment"] = df.get("property_segment", "").fillna("").astype(str).str.lower()
    df["geocode_precision"] = df.get("geocode_precision", "").fillna("").astype(str).str.lower()
    comments = df.get("spc_comments", "").fillna("").astype(str)
    df["kw_damage"] = comments.apply(lambda x: _keyword_count(x, ("damage", "destroyed", "roof", "torn")))
    df["kw_large_hail"] = comments.apply(lambda x: _keyword_count(x, ("baseball", "softball", "2 inch", "3 inch")))
    df["kw_tornado"] = comments.apply(lambda x: _keyword_count(x, ("tornado", "ef-")))
    numeric_cols = [
        "lead_rank_score",
        "severity_score_0_1",
        "image_evidence_score_0_3",
        "sentinel2_days_after_event",
        "sentinel2_cloud_cover_pct",
        "kw_damage",
        "kw_large_hail",
        "kw_tornado",
    ]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    out = df[
        [
            "report_type",
            "property_segment",
            "geocode_precision",
            *numeric_cols,
        ]
    ].copy()
    return out


def _build_pipeline() -> ColumnTransformer:
    cat_cols = ["report_type", "property_segment", "geocode_precision"]
    num_cols = [
        "lead_rank_score",
        "severity_score_0_1",
        "image_evidence_score_0_3",
        "sentinel2_days_after_event",
        "sentinel2_cloud_cover_pct",
        "kw_damage",
        "kw_large_hail",
        "kw_tornado",
    ]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )
    return pre


def train(lead_csv: Path, labels_csv: Path, out_model: Path, out_metrics: Path) -> Dict[str, Any]:
    leads = pd.read_csv(lead_csv)
    labels = pd.read_csv(labels_csv)
    labels["lead_id"] = labels["lead_id"].astype(str)
    leads["lead_id"] = leads["lead_id"].astype(str)
    labels = labels[labels["replacement_needed"].isin([0, 1, "0", "1", 0.0, 1.0])]
    labels["replacement_needed"] = labels["replacement_needed"].astype(float).astype(int)
    merged = leads.merge(labels[["lead_id", "replacement_needed"]], on="lead_id", how="inner")
    if len(merged) < 12:
        raise ValueError(f"Need at least 12 labeled rows to train; found {len(merged)}")

    X = _feature_frame(merged)
    y = merged["replacement_needed"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    pre = _build_pipeline()
    models = {
        "gb": GradientBoostingClassifier(random_state=42),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
    }
    best_name = None
    best_model = None
    best_auc = -1.0
    evals: Dict[str, Dict[str, Any]] = {}
    for name, est in models.items():
        pipe = Pipeline([("pre", pre), ("model", est)])
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else pipe.predict(X_test)
        pred = (prob >= 0.5).astype(int)
        try:
            auc = float(roc_auc_score(y_test, prob))
        except ValueError:
            auc = float("nan")
        f1 = float(f1_score(y_test, pred, zero_division=0))
        acc = float(accuracy_score(y_test, pred))
        evals[name] = {"auc": auc, "f1": f1, "accuracy": acc}
        if np.isnan(auc):
            score = f1
        else:
            score = auc
        if score > best_auc:
            best_auc = score
            best_name = name
            best_model = pipe

    out_model.parent.mkdir(parents=True, exist_ok=True)
    with out_model.open("wb") as f:
        pickle.dump(best_model, f)

    metrics = {
        "n_labeled_rows": int(len(merged)),
        "class_balance": merged["replacement_needed"].value_counts().to_dict(),
        "best_model": best_name,
        "model_metrics": evals,
    }
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Train baseline roof replacement model from labeled leads")
    p.add_argument("--lead-csv", type=Path, required=True, help="Sentinel-enriched lead CSV")
    p.add_argument("--labels-csv", type=Path, required=True, help="Filled labels template CSV")
    p.add_argument("--out-model", type=Path, default=Path("roof_hunter/models/roof_damage_baseline.pkl"))
    p.add_argument("--out-metrics", type=Path, default=Path("roof_hunter/output/roof_damage_training_metrics.json"))
    args = p.parse_args()
    metrics = train(args.lead_csv, args.labels_csv, args.out_model, args.out_metrics)
    print(
        "Training complete: "
        f"rows={metrics['n_labeled_rows']} best_model={metrics['best_model']} metrics_file={args.out_metrics}"
    )


if __name__ == "__main__":
    main()
