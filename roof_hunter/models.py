from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

@dataclass
class ForecastState:
    timestamp: datetime
    latitude: float
    longitude: float
    surface_temp_c: float
    relative_humidity: float
    surface_pressure_hpa: float = 1013.25
    surface_dewpoint_c: Optional[float] = None
    precipitable_water_mm: Optional[float] = None
    low_level_moisture_g_m3: Optional[float] = None
    surface_pressure_trend_hpa_per_hour: Optional[float] = None
    wind_speed_m_s: float = 0.0
    wind_direction_deg: float = 0.0
    precip_mm: float = 0.0
    lightning_potential_j_kg: Optional[float] = None
    lightning_flashes_per_hour: Optional[float] = None
    hail_probability: float = 0.0
    tornado_probability: float = 0.0
    risk_score: float = 0.0
    hail_core_confidence: float = 0.0
    hail_core_radius_ft: float = 0.0
    hail_core_note: str = ""
    note: str = ""
    graphcast_cape_j_kg: Optional[float] = None
    graphcast_cin_j_kg: Optional[float] = None
    graphcast_hail_potential_0_1: Optional[float] = None
    qu_global_cape_j_kg: Optional[float] = None
    qu_global_cin_j_kg: Optional[float] = None
    qu_outlook_hail_environment_0_1: Optional[float] = None
    qu_outlook_uncertainty_0_1: Optional[float] = None
    qu_satellite_hail_nowcast_0_1: Optional[float] = None
    lightning_severe_hail_prob_0_1: Optional[float] = None
    mesoscale_refined: bool = False
    corrdiff_refined: bool = False
    property_metadata: Optional[Dict[str, Any]] = None
    vulnerability_score: float = 0.0
    s2s_pattern_risk: float = 0.0
    updraft_helicity: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'latitude': self.latitude,
            'longitude': self.longitude,
            'surface_temp_c': self.surface_temp_c,
            'relative_humidity': self.relative_humidity,
            'surface_pressure_hpa': self.surface_pressure_hpa,
            'surface_dewpoint_c': self.surface_dewpoint_c,
            'precipitable_water_mm': self.precipitable_water_mm,
            'low_level_moisture_g_m3': self.low_level_moisture_g_m3,
            'surface_pressure_trend_hpa_per_hour': self.surface_pressure_trend_hpa_per_hour,
            'wind_speed_m_s': self.wind_speed_m_s,
            'wind_direction_deg': self.wind_direction_deg,
            'precip_mm': self.precip_mm,
            'lightning_potential_j_kg': self.lightning_potential_j_kg,
            'lightning_flashes_per_hour': self.lightning_flashes_per_hour,
            'hail_probability': self.hail_probability,
            'tornado_probability': self.tornado_probability,
            'risk_score': self.risk_score,
            'hail_core_confidence': self.hail_core_confidence,
            'hail_core_radius_ft': self.hail_core_radius_ft,
            'hail_core_note': self.hail_core_note,
            'note': self.note,
            'graphcast_cape_j_kg': self.graphcast_cape_j_kg,
            'graphcast_cin_j_kg': self.graphcast_cin_j_kg,
            'graphcast_hail_potential_0_1': self.graphcast_hail_potential_0_1,
            'qu_global_cape_j_kg': self.qu_global_cape_j_kg,
            'qu_global_cin_j_kg': self.qu_global_cin_j_kg,
            'qu_outlook_hail_environment_0_1': self.qu_outlook_hail_environment_0_1,
            'qu_outlook_uncertainty_0_1': self.qu_outlook_uncertainty_0_1,
            'qu_satellite_hail_nowcast_0_1': self.qu_satellite_hail_nowcast_0_1,
            'lightning_severe_hail_prob_0_1': self.lightning_severe_hail_prob_0_1,
            'mesoscale_refined': self.mesoscale_refined,
            'corrdiff_refined': self.corrdiff_refined,
            'property_metadata': self.property_metadata,
            'vulnerability_score': self.vulnerability_score,
            's2s_pattern_risk': self.s2s_pattern_risk,
            'updraft_helicity': self.updraft_helicity,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ForecastState':
        def _opt_float(key: str, default: Optional[float] = None) -> Optional[float]:
            value = data.get(key, default)
            return None if value is None else float(value)

        return ForecastState(
            timestamp=datetime.fromisoformat(data['timestamp']),
            latitude=float(data.get('latitude', 0.0)),
            longitude=float(data.get('longitude', 0.0)),
            surface_temp_c=float(data.get('surface_temp_c', 20.0)),
            relative_humidity=float(data.get('relative_humidity', 0.5)),
            surface_pressure_hpa=float(data.get('surface_pressure_hpa', 1013.25)),
            surface_dewpoint_c=_opt_float('surface_dewpoint_c'),
            precipitable_water_mm=_opt_float('precipitable_water_mm'),
            low_level_moisture_g_m3=_opt_float('low_level_moisture_g_m3'),
            surface_pressure_trend_hpa_per_hour=_opt_float('surface_pressure_trend_hpa_per_hour'),
            wind_speed_m_s=float(data.get('wind_speed_m_s', 0.0)),
            wind_direction_deg=float(data.get('wind_direction_deg', 0.0)),
            precip_mm=float(data.get('precip_mm', 0.0)),
            lightning_potential_j_kg=_opt_float('lightning_potential_j_kg'),
            lightning_flashes_per_hour=_opt_float('lightning_flashes_per_hour'),
            graphcast_cape_j_kg=_opt_float('graphcast_cape_j_kg'),
            graphcast_cin_j_kg=_opt_float('graphcast_cin_j_kg'),
            graphcast_hail_potential_0_1=_opt_float('graphcast_hail_potential_0_1'),
            qu_global_cape_j_kg=_opt_float('qu_global_cape_j_kg'),
            qu_global_cin_j_kg=_opt_float('qu_global_cin_j_kg'),
            qu_outlook_hail_environment_0_1=_opt_float('qu_outlook_hail_environment_0_1'),
            qu_outlook_uncertainty_0_1=_opt_float('qu_outlook_uncertainty_0_1'),
            qu_satellite_hail_nowcast_0_1=_opt_float('qu_satellite_hail_nowcast_0_1'),
            lightning_severe_hail_prob_0_1=_opt_float('lightning_severe_hail_prob_0_1'),
            mesoscale_refined=bool(data.get('mesoscale_refined', False)),
            corrdiff_refined=bool(data.get('corrdiff_refined', False) or data.get('mesoscale_refined', False)),
            property_metadata=data.get('property_metadata'),
            vulnerability_score=float(data.get('vulnerability_score', 0.0)),
            s2s_pattern_risk=float(data.get('s2s_pattern_risk', 0.0)),
            updraft_helicity=_opt_float('updraft_helicity'),
        )
