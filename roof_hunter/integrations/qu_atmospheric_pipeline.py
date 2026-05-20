"""QuLab end-to-end forecast row enrichment before ``RoofHunterWeatherTwin`` simulation."""

from __future__ import annotations

from typing import Any, Dict, List

from roof_hunter.integrations.global_outlook_engine import enrich_forecast_outlook
from roof_hunter.integrations.satellite_nowcast_engine import enrich_forecast_satellite


def enrich_forecast_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Apply QuLab global outlook + satellite nowcast enrichment to each forecast row."""
    rows: List[Dict[str, Any]] = list(payload.get("forecast", []))
    rows = enrich_forecast_outlook(rows)
    rows = enrich_forecast_satellite(rows)
    return {"forecast": rows}
