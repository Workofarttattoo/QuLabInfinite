"""
Azure Digital Twins weather simulator for Roof Hunter.

The local simulator and DTDL model generation work without Azure packages.
Publishing to Azure Digital Twins is enabled when ``azure-digitaltwins-core``
and ``azure-identity`` are installed in the deployment environment.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .nexrad_fetcher import RadarObservation, nearest_station
from .roof_hunter_bridge import HailIntelligence, PropertyAssessment

_DTMI_PREFIX = "dtmi:roofhunter"
_MAX_SCORE = 1.0


@dataclass
class RoofProfile:
    """Physical roof attributes modeled as a Roof Hunter digital twin."""

    property_id: str
    latitude: float
    longitude: float
    material: str = "asphalt_shingle"
    area_sqft: float = 1800.0
    slope_degrees: float = 25.0
    age_years: float = 8.0
    albedo: float = 0.18
    emissivity: float = 0.90
    drainage_score: float = 0.75
    tree_cover_percent: float = 10.0
    elevation_m: float = 300.0

    def twin_id(self) -> str:
        return _safe_twin_id(f"roof-{self.property_id}")

    def to_twin(self) -> dict[str, Any]:
        return {
            "$metadata": {"$model": TwinModelFactory.roof_model_id()},
            "propertyId": self.property_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "material": self.material,
            "areaSqft": self.area_sqft,
            "slopeDegrees": self.slope_degrees,
            "ageYears": self.age_years,
            "albedo": self.albedo,
            "emissivity": self.emissivity,
            "drainageScore": self.drainage_score,
            "treeCoverPercent": self.tree_cover_percent,
            "elevationM": self.elevation_m,
        }


@dataclass
class WeatherSnapshot:
    """Weather inputs for a roof-level simulation."""

    latitude: float
    longitude: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    air_temp_c: float = 24.0
    dewpoint_c: float = 16.0
    humidity_percent: float = 65.0
    wind_speed_mps: float = 6.0
    wind_direction_degrees: float = 220.0
    gust_mps: float = 12.0
    pressure_hpa: float = 1010.0
    precipitation_rate_mm_hr: float = 0.0
    cloud_cover_percent: float = 45.0
    solar_radiation_w_m2: float = 650.0
    reflectivity_dbz: float = 35.0
    differential_reflectivity: float = 1.5
    correlation_coefficient: float = 0.96
    specific_differential_phase: float = 0.8
    cape_j_kg: float = 1200.0
    shear_0_6km_kt: float = 30.0
    freezing_level_m: float = 3500.0

    def twin_id(self) -> str:
        stamp = re.sub(r"[^0-9A-Za-z]+", "-", self.timestamp).strip("-")
        return _safe_twin_id(f"weather-{round(self.latitude, 4)}-{round(self.longitude, 4)}-{stamp}")

    def to_radar_observation(self) -> RadarObservation:
        reflectivity = max(0.0, self.reflectivity_dbz)
        return RadarObservation(
            latitude=self.latitude,
            longitude=self.longitude,
            time=self.timestamp,
            station_id=nearest_station(self.latitude, self.longitude),
            reflectivity_max=reflectivity,
            reflectivity_mean=max(0.0, reflectivity - 7.0),
            reflectivity_std=4.0 + min(8.0, reflectivity / 12.0),
            velocity_max=max(self.gust_mps, self.wind_speed_mps),
            velocity_mean=self.wind_speed_mps,
            spectrum_width_mean=max(1.0, self.gust_mps - self.wind_speed_mps),
            differential_reflectivity=self.differential_reflectivity,
            correlation_coefficient=self.correlation_coefficient,
            specific_differential_phase=self.specific_differential_phase,
            vil=max(0.0, (reflectivity - 20.0) * 0.9 + self.cape_j_kg / 250.0),
            echo_top_km=max(3.0, reflectivity / 5.5),
        )

    def to_twin(self) -> dict[str, Any]:
        return {
            "$metadata": {"$model": TwinModelFactory.weather_cell_model_id()},
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timestamp": self.timestamp,
            "airTempC": self.air_temp_c,
            "dewpointC": self.dewpoint_c,
            "humidityPercent": self.humidity_percent,
            "windSpeedMps": self.wind_speed_mps,
            "windDirectionDegrees": self.wind_direction_degrees,
            "gustMps": self.gust_mps,
            "pressureHpa": self.pressure_hpa,
            "precipitationRateMmHr": self.precipitation_rate_mm_hr,
            "cloudCoverPercent": self.cloud_cover_percent,
            "solarRadiationWM2": self.solar_radiation_w_m2,
            "reflectivityDbz": self.reflectivity_dbz,
            "capeJKg": self.cape_j_kg,
            "shear06kmKt": self.shear_0_6km_kt,
            "freezingLevelM": self.freezing_level_m,
        }


@dataclass
class WeatherSimulationResult:
    """Roof Hunter simulation output suitable for storing as a digital twin."""

    simulation_id: str
    roof_twin_id: str
    weather_twin_id: str
    timestamp: str
    roof_surface_temp_c: float
    hail_probability: float
    estimated_hail_size_inches: float
    hail_damage_score: float
    heat_stress_score: float
    runoff_risk_score: float
    combined_climate_risk_score: float
    action: str
    risk_level: str
    drivers: list[str]
    source_quality: str

    def to_twin(self) -> dict[str, Any]:
        return {
            "$metadata": {"$model": TwinModelFactory.simulation_run_model_id()},
            "simulationId": self.simulation_id,
            "timestamp": self.timestamp,
            "roofSurfaceTempC": self.roof_surface_temp_c,
            "hailProbability": self.hail_probability,
            "estimatedHailSizeInches": self.estimated_hail_size_inches,
            "hailDamageScore": self.hail_damage_score,
            "heatStressScore": self.heat_stress_score,
            "runoffRiskScore": self.runoff_risk_score,
            "combinedClimateRiskScore": self.combined_climate_risk_score,
            "action": self.action,
            "riskLevel": self.risk_level,
            "drivers": self.drivers,
            "sourceQuality": self.source_quality,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TwinModelFactory:
    """Creates DTDL models for the Roof Hunter weather graph."""

    @staticmethod
    def roof_model_id() -> str:
        return f"{_DTMI_PREFIX}:Roof;1"

    @staticmethod
    def weather_cell_model_id() -> str:
        return f"{_DTMI_PREFIX}:WeatherCell;1"

    @staticmethod
    def weather_station_model_id() -> str:
        return f"{_DTMI_PREFIX}:WeatherStation;1"

    @staticmethod
    def simulation_run_model_id() -> str:
        return f"{_DTMI_PREFIX}:SimulationRun;1"

    @classmethod
    def all_models(cls) -> list[dict[str, Any]]:
        return [
            cls.roof_model(),
            cls.weather_cell_model(),
            cls.weather_station_model(),
            cls.simulation_run_model(),
        ]

    @classmethod
    def write_models(cls, directory: str | Path) -> list[Path]:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for model in cls.all_models():
            name = model["@id"].replace("dtmi:", "").replace(":", "_").replace(";", "_")
            path = target / f"{name}.json"
            path.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
            paths.append(path)
        return paths

    @classmethod
    def roof_model(cls) -> dict[str, Any]:
        return _interface(
            cls.roof_model_id(),
            "Roof",
            [
                _property("propertyId", "string"),
                _property("latitude", "double"),
                _property("longitude", "double"),
                _property("material", "string"),
                _property("areaSqft", "double"),
                _property("slopeDegrees", "double"),
                _property("ageYears", "double"),
                _property("albedo", "double"),
                _property("emissivity", "double"),
                _property("drainageScore", "double"),
                _property("treeCoverPercent", "double"),
                _property("elevationM", "double"),
                _relationship("affectedBy", cls.weather_cell_model_id()),
                _relationship("hasSimulation", cls.simulation_run_model_id()),
            ],
        )

    @classmethod
    def weather_cell_model(cls) -> dict[str, Any]:
        return _interface(
            cls.weather_cell_model_id(),
            "WeatherCell",
            [
                _property("latitude", "double"),
                _property("longitude", "double"),
                _property("timestamp", "dateTime"),
                _property("airTempC", "double"),
                _property("dewpointC", "double"),
                _property("humidityPercent", "double"),
                _property("windSpeedMps", "double"),
                _property("windDirectionDegrees", "double"),
                _property("gustMps", "double"),
                _property("pressureHpa", "double"),
                _property("precipitationRateMmHr", "double"),
                _property("cloudCoverPercent", "double"),
                _property("solarRadiationWM2", "double"),
                _property("reflectivityDbz", "double"),
                _property("capeJKg", "double"),
                _property("shear06kmKt", "double"),
                _property("freezingLevelM", "double"),
            ],
        )

    @classmethod
    def weather_station_model(cls) -> dict[str, Any]:
        return _interface(
            cls.weather_station_model_id(),
            "WeatherStation",
            [
                _property("stationId", "string"),
                _property("provider", "string"),
                _property("latitude", "double"),
                _property("longitude", "double"),
                _property("lastObservedAt", "dateTime"),
                _relationship("observes", cls.weather_cell_model_id()),
            ],
        )

    @classmethod
    def simulation_run_model(cls) -> dict[str, Any]:
        return _interface(
            cls.simulation_run_model_id(),
            "SimulationRun",
            [
                _property("simulationId", "string"),
                _property("timestamp", "dateTime"),
                _property("roofSurfaceTempC", "double"),
                _property("hailProbability", "double"),
                _property("estimatedHailSizeInches", "double"),
                _property("hailDamageScore", "double"),
                _property("heatStressScore", "double"),
                _property("runoffRiskScore", "double"),
                _property("combinedClimateRiskScore", "double"),
                _property("action", "string"),
                _property("riskLevel", "string"),
                _property("drivers", {"@type": "Array", "elementSchema": "string"}),
                _property("sourceQuality", "string"),
            ],
        )


class AzureDigitalTwinsPublisher:
    """Small adapter around Azure Digital Twins SDK calls."""

    def __init__(self, endpoint: str, credential: Any | None = None, client: Any | None = None):
        self.endpoint = endpoint
        if client is not None:
            self.client = client
            return

        try:
            from azure.digitaltwins.core import DigitalTwinsClient
            from azure.identity import DefaultAzureCredential
        except ImportError as exc:
            raise RuntimeError(
                "Azure publishing requires azure-digitaltwins-core and azure-identity. "
                "Install them before deploying twins to Azure."
            ) from exc

        self.client = DigitalTwinsClient(endpoint, credential or DefaultAzureCredential())

    def upsert_models(self, models: list[dict[str, Any]] | None = None) -> None:
        payload = models or TwinModelFactory.all_models()
        try:
            self.client.create_models(payload)
        except Exception as exc:
            if "ModelAlreadyExists" not in str(exc) and "ModelIdAlreadyExists" not in str(exc):
                raise

    def publish_simulation(
        self,
        roof: RoofProfile,
        weather: WeatherSnapshot,
        result: WeatherSimulationResult,
    ) -> None:
        self.client.upsert_digital_twin(roof.twin_id(), roof.to_twin())
        self.client.upsert_digital_twin(weather.twin_id(), weather.to_twin())
        self.client.upsert_digital_twin(result.simulation_id, result.to_twin())
        self._upsert_relationship(roof.twin_id(), "affectedBy", weather.twin_id())
        self._upsert_relationship(roof.twin_id(), "hasSimulation", result.simulation_id)

    def _upsert_relationship(self, source_id: str, name: str, target_id: str) -> None:
        rel_id = _safe_twin_id(f"{source_id}-{name}-{target_id}")
        relationship = {
            "$relationshipId": rel_id,
            "$sourceId": source_id,
            "$relationshipName": name,
            "$targetId": target_id,
        }
        self.client.upsert_relationship(source_id, rel_id, relationship)


class RoofHunterWeatherSimulator:
    """Combines roof metadata, local weather, hail intelligence, and digital twins."""

    def __init__(
        self,
        hail_intelligence: HailIntelligence | None = None,
        publisher: AzureDigitalTwinsPublisher | None = None,
    ):
        self.hail_intelligence = hail_intelligence or HailIntelligence()
        self.publisher = publisher

    def simulate_roof(
        self,
        roof: RoofProfile,
        weather: WeatherSnapshot,
        publish: bool = False,
    ) -> WeatherSimulationResult:
        radar_obs = weather.to_radar_observation()
        hail = self.hail_intelligence.assess_property(
            roof.latitude,
            roof.longitude,
            radar_obs=radar_obs,
            include_alerts=False,
        )

        surface_temp = _roof_surface_temperature(roof, weather)
        hail_damage = _hail_damage_score(roof, hail)
        heat_stress = _heat_stress_score(surface_temp, roof)
        runoff = _runoff_risk_score(roof, weather)
        combined = _clamp(0.50 * hail_damage + 0.30 * heat_stress + 0.20 * runoff)
        drivers = _risk_drivers(hail_damage, heat_stress, runoff, weather)
        risk_level = _risk_level(combined)
        action = _action(combined, hail)

        result = WeatherSimulationResult(
            simulation_id=_safe_twin_id(f"sim-{roof.property_id}-{weather.timestamp}"),
            roof_twin_id=roof.twin_id(),
            weather_twin_id=weather.twin_id(),
            timestamp=weather.timestamp,
            roof_surface_temp_c=round(surface_temp, 2),
            hail_probability=round(hail.hail_probability, 4),
            estimated_hail_size_inches=hail.estimated_hail_size_inches,
            hail_damage_score=round(hail_damage, 4),
            heat_stress_score=round(heat_stress, 4),
            runoff_risk_score=round(runoff, 4),
            combined_climate_risk_score=round(combined, 4),
            action=action,
            risk_level=risk_level,
            drivers=drivers,
            source_quality="roof_twin+weather_snapshot+hail_model",
        )

        if publish:
            if self.publisher is None:
                raise RuntimeError("publish=True requires an AzureDigitalTwinsPublisher")
            self.publisher.publish_simulation(roof, weather, result)

        return result


def _interface(model_id: str, display_name: str, contents: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "@context": "dtmi:dtdl:context;2",
        "@id": model_id,
        "@type": "Interface",
        "displayName": display_name,
        "contents": contents,
    }


def _property(name: str, schema: str | dict[str, Any]) -> dict[str, Any]:
    return {"@type": "Property", "name": name, "schema": schema}


def _relationship(name: str, target: str) -> dict[str, str]:
    return {"@type": "Relationship", "name": name, "target": target}


def _safe_twin_id(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_-]+", "-", value).strip("-")
    return safe[:128] or "roofhunter-twin"


def _roof_surface_temperature(roof: RoofProfile, weather: WeatherSnapshot) -> float:
    solar_gain = weather.solar_radiation_w_m2 * (1.0 - _clamp(roof.albedo)) * 0.018
    wind_cooling = min(12.0, weather.wind_speed_mps * 0.65)
    rain_cooling = min(10.0, weather.precipitation_rate_mm_hr * 0.45)
    shade_cooling = min(8.0, roof.tree_cover_percent * 0.06)
    emissivity_adjustment = (1.0 - _clamp(roof.emissivity)) * 4.0
    return weather.air_temp_c + solar_gain - wind_cooling - rain_cooling - shade_cooling + emissivity_adjustment


def _hail_damage_score(roof: RoofProfile, hail: PropertyAssessment) -> float:
    vulnerability = {
        "asphalt_shingle": 0.72,
        "wood_shake": 0.82,
        "metal": 0.42,
        "tile": 0.66,
        "slate": 0.55,
        "membrane": 0.78,
    }.get(roof.material.lower(), 0.70)
    age_factor = 0.75 + min(0.35, max(0.0, roof.age_years) / 85.0)
    slope_factor = 0.90 + min(0.20, abs(roof.slope_degrees - 25.0) / 120.0)
    size_factor = 1.0 + min(0.75, hail.estimated_hail_size_inches / 3.0)
    return _clamp(hail.hail_probability * vulnerability * age_factor * slope_factor * size_factor)


def _heat_stress_score(surface_temp_c: float, roof: RoofProfile) -> float:
    temp_component = _clamp((surface_temp_c - 35.0) / 35.0)
    low_albedo_component = 1.0 - _clamp(roof.albedo)
    return _clamp(0.75 * temp_component + 0.25 * low_albedo_component)


def _runoff_risk_score(roof: RoofProfile, weather: WeatherSnapshot) -> float:
    rain_component = _clamp(weather.precipitation_rate_mm_hr / 55.0)
    drainage_component = 1.0 - _clamp(roof.drainage_score)
    slope_component = _clamp(roof.slope_degrees / 45.0)
    return _clamp(0.65 * rain_component + 0.25 * drainage_component + 0.10 * slope_component)


def _risk_drivers(
    hail_damage: float,
    heat_stress: float,
    runoff: float,
    weather: WeatherSnapshot,
) -> list[str]:
    drivers: list[str] = []
    if hail_damage >= 0.35:
        drivers.append("hail")
    if heat_stress >= 0.45:
        drivers.append("heat")
    if runoff >= 0.35:
        drivers.append("runoff")
    if weather.wind_speed_mps >= 18 or weather.gust_mps >= 25:
        drivers.append("wind")
    return drivers or ["baseline"]


def _risk_level(score: float) -> str:
    if score >= 0.75:
        return "EXTREME"
    if score >= 0.55:
        return "HIGH"
    if score >= 0.35:
        return "MODERATE"
    if score >= 0.18:
        return "LOW"
    return "MINIMAL"


def _action(score: float, hail: PropertyAssessment) -> str:
    if score >= 0.55 or hail.action == "QUALIFY":
        return "QUALIFY"
    if score >= 0.30 or hail.action == "MONITOR":
        return "MONITOR"
    return "SKIP"


def _clamp(value: float, lower: float = 0.0, upper: float = _MAX_SCORE) -> float:
    if math.isnan(value):
        return lower
    return max(lower, min(upper, value))
