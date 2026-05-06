"""
Roof Hunter climate simulation routes.

These endpoints expose the Azure Digital Twins-ready roof weather simulator
without requiring Azure credentials for local use.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from hail_model.azure_digital_twins import (
    RoofHunterWeatherSimulator,
    RoofProfile,
    TwinModelFactory,
    WeatherSnapshot,
)
from qulab.api.auth import get_api_key

router = APIRouter()


class RoofProfileRequest(BaseModel):
    property_id: str
    latitude: float
    longitude: float
    material: str = "asphalt_shingle"
    area_sqft: float = 1800.0
    slope_degrees: float = 25.0
    age_years: float = 8.0
    albedo: float = Field(0.18, ge=0.0, le=1.0)
    emissivity: float = Field(0.90, ge=0.0, le=1.0)
    drainage_score: float = Field(0.75, ge=0.0, le=1.0)
    tree_cover_percent: float = Field(10.0, ge=0.0, le=100.0)
    elevation_m: float = 300.0


class WeatherSnapshotRequest(BaseModel):
    latitude: float
    longitude: float
    timestamp: str | None = None
    air_temp_c: float = 24.0
    dewpoint_c: float = 16.0
    humidity_percent: float = Field(65.0, ge=0.0, le=100.0)
    wind_speed_mps: float = Field(6.0, ge=0.0)
    wind_direction_degrees: float = Field(220.0, ge=0.0, le=360.0)
    gust_mps: float = Field(12.0, ge=0.0)
    pressure_hpa: float = 1010.0
    precipitation_rate_mm_hr: float = Field(0.0, ge=0.0)
    cloud_cover_percent: float = Field(45.0, ge=0.0, le=100.0)
    solar_radiation_w_m2: float = Field(650.0, ge=0.0)
    reflectivity_dbz: float = Field(35.0, ge=0.0)
    differential_reflectivity: float = 1.5
    correlation_coefficient: float = Field(0.96, ge=0.0, le=1.0)
    specific_differential_phase: float = 0.8
    cape_j_kg: float = Field(1200.0, ge=0.0)
    shear_0_6km_kt: float = Field(30.0, ge=0.0)
    freezing_level_m: float = Field(3500.0, ge=0.0)


class RoofHunterSimulationRequest(BaseModel):
    roof: RoofProfileRequest
    weather: WeatherSnapshotRequest


@router.get("/digital-twin-models")
def digital_twin_models():
    """Return the DTDL models needed for an Azure Digital Twins deployment."""

    return {"models": TwinModelFactory.all_models()}


@router.post("/simulate", dependencies=[Depends(get_api_key)])
def simulate_roof_weather(request: RoofHunterSimulationRequest):
    """Run a roof-level weather simulation from supplied roof and weather state."""

    try:
        roof = RoofProfile(**request.roof.model_dump())
        weather_payload = request.weather.model_dump(exclude_none=True)
        weather = WeatherSnapshot(**weather_payload)
        result = RoofHunterWeatherSimulator().simulate_roof(roof, weather)
        return result.to_dict()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Roof Hunter simulation failed: {exc}",
        ) from exc
