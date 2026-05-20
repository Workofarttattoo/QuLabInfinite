from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class OnlineLogisticBelief:
    """Lightweight Bayesian-ish logistic regression (ADF/EKF-style).

    This is the "digital state" D: a belief over fusion weights for multiple hail signals.
    """

    feature_names: List[str]
    w: List[float]
    # Diagonal covariance only (fast, stable, good-enough for online weighting).
    p_diag: List[float]
    started_at: str
    updated_at: Optional[str] = None
    observations_seen: int = 0
    positive_seen: int = 0

    @staticmethod
    def create_default(feature_names: List[str]) -> "OnlineLogisticBelief":
        # Initialize to roughly mirror the legacy blend:
        # hail_score = 0.7 * env + 0.3 * xg + floors/adders afterwards.
        # We represent that as a sigmoid fusion with conservative weights.
        w = []
        for name in feature_names:
            if name == "bias":
                w.append(-1.35)
            elif name == "env_score":
                w.append(2.8)
            elif name == "xgboost_score":
                w.append(1.6)
            elif name == "s2s_baseline":
                w.append(0.9)
            elif name == "outlook_0_1":
                w.append(0.8)
            elif name == "satellite_nowcast_0_1":
                w.append(0.9)
            elif name == "lightning_boost":
                w.append(1.1)
            elif name == "updraft_helicity_boost":
                w.append(0.7)
            elif name == "radar_lock_strength":
                w.append(0.8)
            elif name == "supercooled_layer_depth_km":
                w.append(0.45)
            elif name == "updraft_survival_index":
                w.append(0.55)
            elif name == "hail_growth_potential_0_1":
                w.append(0.7)
            elif name == "downdraft_cooling_index":
                w.append(0.35)
            elif name == "storm_organization_proxy":
                w.append(0.4)
            else:
                w.append(0.0)

        # Broad prior uncertainty: lets observations move weights.
        p_diag = [1.25] * len(feature_names)
        return OnlineLogisticBelief(
            feature_names=feature_names,
            w=w,
            p_diag=p_diag,
            started_at=datetime.now(tz=timezone.utc).isoformat(),
        )

    def _feature_vector(self, features: Dict[str, float]) -> List[float]:
        return [float(features.get(name, 0.0)) for name in self.feature_names]

    def predict_proba(self, features: Dict[str, float]) -> float:
        x = self._feature_vector(features)
        s = 0.0
        for wi, xi in zip(self.w, x):
            s += wi * xi
        return float(_sigmoid(s))

    def update(self, features: Dict[str, float], y: int) -> Dict[str, Any]:
        """One online update with Bernoulli observation y∈{0,1}.

        Uses an ADF-style diagonal covariance update:
          r = p*(1-p)
          P_new^{-1} = P^{-1} + r x x^T
          w_new = w + P_new x (y - p)
        with P approximated as diagonal for speed.
        """
        if y not in (0, 1):
            raise ValueError("y must be 0 or 1")

        x = self._feature_vector(features)
        p = self.predict_proba(features)
        r = max(1e-6, p * (1.0 - p))

        # Update diagonal precision then invert back to variance.
        new_p_diag: List[float] = []
        for pi, xi in zip(self.p_diag, x):
            prec = (1.0 / max(1e-9, pi)) + r * (xi * xi)
            new_p_diag.append(1.0 / max(1e-9, prec))

        # Gradient step scaled by posterior variance.
        err = float(y - p)
        new_w: List[float] = []
        for wi, xi, pi in zip(self.w, x, new_p_diag):
            new_w.append(wi + pi * xi * err)

        self.w = new_w
        self.p_diag = new_p_diag
        self.observations_seen += 1
        self.positive_seen += int(y)
        self.updated_at = datetime.now(tz=timezone.utc).isoformat()

        return {
            "y": y,
            "p_before": round(p, 4),
            "err": round(err, 4),
            "observations_seen": self.observations_seen,
            "positive_seen": self.positive_seen,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "w": [round(v, 6) for v in self.w],
            "p_diag": [round(v, 6) for v in self.p_diag],
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "observations_seen": self.observations_seen,
            "positive_seen": self.positive_seen,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "OnlineLogisticBelief":
        data = json.loads(path.read_text(encoding="utf-8"))
        return OnlineLogisticBelief(
            feature_names=list(data["feature_names"]),
            w=[float(v) for v in data["w"]],
            p_diag=[float(v) for v in data["p_diag"]],
            started_at=str(data["started_at"]),
            updated_at=data.get("updated_at"),
            observations_seen=int(data.get("observations_seen", 0)),
            positive_seen=int(data.get("positive_seen", 0)),
        )


def default_hail_feature_names() -> List[str]:
    return [
        "bias",
        "env_score",
        "xgboost_score",
        "s2s_baseline",
        "outlook_0_1",
        "satellite_nowcast_0_1",
        "lightning_boost",
        "updraft_helicity_boost",
        "radar_lock_strength",
        "supercooled_layer_depth_km",
        "updraft_survival_index",
        "hail_growth_potential_0_1",
        "downdraft_cooling_index",
        "storm_organization_proxy",
    ]

