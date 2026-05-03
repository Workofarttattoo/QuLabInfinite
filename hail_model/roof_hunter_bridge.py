"""
Roof Hunter Integration Bridge.

Connects the XGBoost hail model, NEXRAD fetcher, Dual-Pol algorithms,
and the existing hail_lab nowcaster / physics engine into a single
callable pipeline for the Roof Hunter lead-generation system.

Usage in Roof Hunter main.py::

    from hail_model.roof_hunter_bridge import HailIntelligence

    intel = HailIntelligence(model_path="hail_model/models/xgboost_hail.json")

    for property in properties:
        result = intel.assess_property(property["lat"], property["lon"])
        if result["action"] == "QUALIFY":
            # property is a high-confidence hail lead
            ...

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dual_pol import (
    DualPolObservation,
    estimate_hail_size,
)
from .nexrad_fetcher import NEXRADFetcher, RadarObservation
from .predict import HailPredictor

logger = logging.getLogger(__name__)

_QUALIFY_THRESHOLD = 0.65
_HIGH_PRIORITY_THRESHOLD = 0.80


@dataclass
class PropertyAssessment:
    """Complete hail-risk assessment for a single property."""

    latitude: float
    longitude: float
    hail_probability: float
    risk_level: str
    hail_predicted: bool
    estimated_hail_size_inches: float
    hydrometeor_class: str
    mesh_inches: float
    posh_percent: float
    dual_pol_confidence: float
    active_warnings: list[str]
    action: str  # QUALIFY / MONITOR / SKIP
    radar_station: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hail_probability": round(self.hail_probability, 4),
            "risk_level": self.risk_level,
            "hail_predicted": self.hail_predicted,
            "estimated_hail_size_inches": self.estimated_hail_size_inches,
            "hydrometeor_class": self.hydrometeor_class,
            "mesh_inches": self.mesh_inches,
            "posh_percent": self.posh_percent,
            "dual_pol_confidence": round(self.dual_pol_confidence, 3),
            "active_warnings": self.active_warnings,
            "action": self.action,
            "radar_station": self.radar_station,
        }


class HailIntelligence:
    """Unified hail-intelligence engine for Roof Hunter.

    Orchestrates:
      1. NEXRAD radar feature extraction
      2. XGBoost probability prediction
      3. Dual-Pol hail classification + size estimation
      4. NWS active-alert enrichment
      5. Action recommendation (QUALIFY / MONITOR / SKIP)
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        qualify_threshold: float = _QUALIFY_THRESHOLD,
        high_priority_threshold: float = _HIGH_PRIORITY_THRESHOLD,
    ):
        self.qualify_threshold = qualify_threshold
        self.high_priority_threshold = high_priority_threshold
        self.fetcher = NEXRADFetcher()

        if model_path and Path(model_path).exists():
            self.predictor = HailPredictor(model_path=model_path)
        else:
            self.predictor = None
            logger.warning(
                "No XGBoost model loaded (path=%s). "
                "Predictions will use dual-pol only.",
                model_path,
            )

    def assess_property(
        self,
        lat: float,
        lon: float,
        radar_obs: RadarObservation | None = None,
        include_alerts: bool = True,
    ) -> PropertyAssessment:
        """Run the full assessment pipeline for one property."""

        # 1. Get radar features
        if radar_obs is None:
            radar_obs = self.fetcher.fetch_radar_features(lat, lon)

        # 2. XGBoost probability
        if self.predictor is not None:
            xgb_result = self.predictor.predict_full(radar_obs.to_dict())
            hail_prob = xgb_result["hail_probability"]
            risk_level = xgb_result["risk_level"]
            hail_predicted = xgb_result["hail_predicted"]
        else:
            hail_prob = 0.0
            risk_level = "UNKNOWN"
            hail_predicted = False

        # 3. Dual-Pol hail estimation
        dp_obs = DualPolObservation(
            reflectivity_h=radar_obs.reflectivity_max,
            differential_reflectivity=radar_obs.differential_reflectivity,
            correlation_coefficient=radar_obs.correlation_coefficient,
            specific_differential_phase=radar_obs.specific_differential_phase,
        )
        hail_est = estimate_hail_size(dp_obs)

        # 4. Merge XGBoost + dual-pol for final probability
        if self.predictor is not None:
            combined_prob = 0.7 * hail_prob + 0.3 * (hail_est.posh_percent / 100.0)
        else:
            combined_prob = hail_est.posh_percent / 100.0

        # 5. Active NWS warnings
        warnings: list[str] = []
        if include_alerts:
            try:
                alerts = self.fetcher.fetch_active_alerts(lat, lon, event_filter="hail")
                warnings = [a.headline for a in alerts]
            except Exception:
                pass

        # 6. Action recommendation
        if combined_prob >= self.high_priority_threshold or hail_est.hail_detected:
            action = "QUALIFY"
        elif combined_prob >= self.qualify_threshold:
            action = "MONITOR"
        else:
            action = "SKIP"

        return PropertyAssessment(
            latitude=lat,
            longitude=lon,
            hail_probability=combined_prob,
            risk_level=risk_level if self.predictor else _risk_from_prob(combined_prob),
            hail_predicted=hail_predicted or hail_est.hail_detected,
            estimated_hail_size_inches=hail_est.estimated_diameter_inches,
            hydrometeor_class=hail_est.hydrometeor_class.value,
            mesh_inches=hail_est.mesh_inches,
            posh_percent=hail_est.posh_percent,
            dual_pol_confidence=hail_est.confidence,
            active_warnings=warnings,
            action=action,
            radar_station=radar_obs.station_id,
        )

    def assess_batch(
        self, properties: list[dict[str, float]], include_alerts: bool = False
    ) -> list[PropertyAssessment]:
        """Assess multiple properties and return sorted by risk."""
        results = []
        for prop in properties:
            result = self.assess_property(
                prop["lat"], prop["lon"], include_alerts=include_alerts
            )
            results.append(result)
        results.sort(key=lambda r: r.hail_probability, reverse=True)
        return results


def _risk_from_prob(p: float) -> str:
    if p >= 0.8:
        return "EXTREME"
    if p >= 0.6:
        return "HIGH"
    if p >= 0.4:
        return "MODERATE"
    if p >= 0.2:
        return "LOW"
    return "MINIMAL"
