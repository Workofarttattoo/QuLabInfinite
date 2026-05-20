from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from atmospheric_science_lab.atmospheric_science_lab import AtmosphericScienceLab

from roof_hunter.integrations.satellite_nowcast_engine import satellite_hail_nowcast_boost
from roof_hunter.integrations.ml_models import predict_hail_xgboost
from roof_hunter.integrations.vulnerability_engine import calculate_roof_vulnerability
from roof_hunter.integrations.radar_engine import NEXRADLevel2
from roof_hunter.integrations.titan_tracker import track_with_irose_titan
from roof_hunter.integrations.s2s_pattern_matcher import calculate_s2s_outlook
from roof_hunter.integrations.environmental_feature_engine import compute_environmental_hail_features


from roof_hunter.models import ForecastState
from roof_hunter.hail_twin_belief import OnlineLogisticBelief, default_hail_feature_names


class RoofHunterWeatherTwin:
    def __init__(self, forecast_states: List[ForecastState], belief_path: Optional[Path] = None):
        self.forecast_states = sorted(forecast_states, key=lambda s: s.timestamp)
        self.atmos_lab = AtmosphericScienceLab()
        self.history: List[Dict[str, Any]] = []
        self.previous_state: Optional[ForecastState] = None
        self.radar_engine = NEXRADLevel2()
        self.active_cells: List[Dict[str, Any]] = []
        self.s2s_baseline: float = 0.0
        self.belief_path = belief_path or Path("artifacts/roof_hunter_hail_belief.json")
        self.hail_belief = self._load_or_init_belief()

    def _load_or_init_belief(self) -> OnlineLogisticBelief:
        try:
            if self.belief_path.exists():
                return OnlineLogisticBelief.load(self.belief_path)
        except Exception:
            pass
        return OnlineLogisticBelief.create_default(default_hail_feature_names())

    def simulate(self) -> List[Dict[str, Any]]:
        """Run the weather twin forward through all forecast states."""
        self.history.clear()
        self.previous_state = None
        
        # 1. Long-Range Baseline (14-30 days out)
        if self.forecast_states:
            s0 = self.forecast_states[0]
            s2s = calculate_s2s_outlook(s0.timestamp, s0.latitude, s0.longitude)
            self.s2s_baseline = s2s['s2s_risk_score']

        # 2. Pre-fetch or simulate storm tracking if we're in 'real-time' mode
        # In a real-world scenario, this would be hooked into a live radar feed
        if not self.active_cells:
            try:
                self.active_cells = self.radar_engine.get_storm_cells(None) # Mocking radar volume for now
            except:
                self.active_cells = []

        for state in self.forecast_states:
            record = self.step(state)
            self.history.append(record)
            self.previous_state = state

        try:
            self.hail_belief.save(self.belief_path)
        except Exception:
            pass
        return self.history

    def step(self, state: ForecastState) -> Dict[str, Any]:
        analysis = self.atmos_lab.run_weather_forecast_analysis(
            altitude_m=0,
            surface_temp_c=state.surface_temp_c,
            relative_humidity=state.relative_humidity,
            surface_pressure_hpa=state.surface_pressure_hpa,
        )

        pressure_trend = state.surface_pressure_trend_hpa_per_hour
        if pressure_trend is None and self.previous_state is not None:
            dt_hours = max(1.0, (state.timestamp - self.previous_state.timestamp).total_seconds() / 3600.0)
            pressure_trend = (state.surface_pressure_hpa - self.previous_state.surface_pressure_hpa) / dt_hours

        moisture_factor = self._estimate_moisture_factor(state, analysis)
        effective_cape = self._resolve_effective_cape_j_kg(state, analysis)
        hail_score, tornado_score, hail_core, lightning_boost, environmental_hail_features = self._assess_severe_risk(
            state, analysis, pressure_trend, moisture_factor, effective_cape
        )

        state.hail_probability = hail_score
        state.tornado_probability = tornado_score
        state.risk_score = hail_core['risk_score']
        state.hail_core_confidence = hail_core['hail_core_confidence']
        state.hail_core_radius_ft = hail_core['hail_core_radius_ft']
        state.hail_core_note = hail_core['hail_core_note']
        state.note = hail_core['note']

        if state.property_metadata:
            # Enrich property metadata with current hail probability for scoring
            state.property_metadata['hail_probability'] = state.hail_probability
            state.vulnerability_score = calculate_roof_vulnerability(state.property_metadata)

        return {
            'timestamp': state.timestamp.isoformat(),
            'latitude': state.latitude,
            'longitude': state.longitude,
            'surface_temp_c': state.surface_temp_c,
            'relative_humidity': state.relative_humidity,
            'surface_pressure_hpa': state.surface_pressure_hpa,
            'surface_dewpoint_c': state.surface_dewpoint_c,
            'precipitable_water_mm': state.precipitable_water_mm,
            'low_level_moisture_g_m3': state.low_level_moisture_g_m3,
            'surface_pressure_trend_hpa_per_hour': state.surface_pressure_trend_hpa_per_hour,
            'wind_speed_m_s': state.wind_speed_m_s,
            'wind_direction_deg': state.wind_direction_deg,
            'precip_mm': state.precip_mm,
            'lightning_potential_j_kg': state.lightning_potential_j_kg,
            'lightning_flashes_per_hour': state.lightning_flashes_per_hour,
            'effective_cape_j_kg': round(effective_cape, 1),
            'lightning_hail_boost': round(lightning_boost, 4),
            'environmental_hail_features': environmental_hail_features,
            'hail_probability': state.hail_probability,
            'tornado_probability': state.tornado_probability,
            'risk_score': state.risk_score,
            'hail_core_confidence': state.hail_core_confidence,
            'hail_core_radius_ft': state.hail_core_radius_ft,
            'hail_core_note': state.hail_core_note,
            'note': state.note,
            'hail_twin_belief': self.hail_belief.snapshot(),
            'analysis': analysis,
            'graphcast_cape_j_kg': state.graphcast_cape_j_kg,
            'graphcast_cin_j_kg': state.graphcast_cin_j_kg,
            'graphcast_hail_potential_0_1': state.graphcast_hail_potential_0_1,
            'qu_global_cape_j_kg': state.qu_global_cape_j_kg,
            'qu_global_cin_j_kg': state.qu_global_cin_j_kg,
            'qu_outlook_hail_environment_0_1': state.qu_outlook_hail_environment_0_1,
            'qu_outlook_uncertainty_0_1': state.qu_outlook_uncertainty_0_1,
            'qu_satellite_hail_nowcast_0_1': state.qu_satellite_hail_nowcast_0_1,
            'lightning_severe_hail_prob_0_1': state.lightning_severe_hail_prob_0_1,
            'mesoscale_refined': state.mesoscale_refined,
            'corrdiff_refined': state.corrdiff_refined,
            'vulnerability_score': state.vulnerability_score,
        }

    def _estimate_moisture_factor(self, state: ForecastState, analysis: Dict[str, Any]) -> float:
        moisture = 0.0
        if state.precipitable_water_mm is not None:
            moisture += min(1.0, state.precipitable_water_mm / 60.0)

        if state.low_level_moisture_g_m3 is not None:
            moisture += min(1.0, state.low_level_moisture_g_m3 / 20.0)

        if state.surface_dewpoint_c is not None:
            moisture += min(1.0, max(0.0, (state.surface_dewpoint_c + 10.0) / 25.0))

        moisture += min(1.0, max(0.0, state.relative_humidity))
        return min(1.0, moisture / 3.0)

    def _resolve_effective_cape_j_kg(self, state: ForecastState, analysis: Dict[str, Any]) -> float:
        """CAPE for severe weather: use observed dewpoint when present and scale toward MLCAPE.

        The lab's parcel CAPE uses a shallow surface excess estimate and is biased low vs.
        MLCAPE / operational indices used for hail forecasting.
        """
        t = state.surface_temp_c
        td = state.surface_dewpoint_c
        if td is None:
            td = float(analysis['moisture_analysis']['dewpoint_c'])
        p = state.surface_pressure_hpa
        raw = float(
            self.atmos_lab.weather.predict_convective_available_potential_energy(t, td, p)['cape_j_kg']
        )
        spread = max(0.0, t - td)
        # Ordinal scaling toward mixed-layer / operational magnitudes (conservative vs. raw parcel).
        ml_scale = 1.22 + 0.42 * min(1.0, spread / 18.0)
        eff = min(5800.0, raw * ml_scale)
        if state.graphcast_cape_j_kg is not None:
            eff = max(eff, float(state.graphcast_cape_j_kg))
        if state.qu_global_cape_j_kg is not None:
            eff = max(eff, float(state.qu_global_cape_j_kg))
        return eff

    @staticmethod
    def _normalize_cape_for_hail(cape: float) -> float:
        """0..1 saturation curve: marginal ~500 J/kg, significant risk ~2000+."""
        if cape <= 0:
            return 0.0
        return float(1.0 - math.exp(-cape / 1450.0))

    @staticmethod
    def _instability_weight_for_hail(t_sfc_c: float, dewpoint_c: float) -> float:
        """Favor mid-range T-Td typical of warm-sector severe (not marine capped, not too dry)."""
        spread = max(0.0, t_sfc_c - dewpoint_c)
        # Gaussian-ish peak near 11 °C spread; width ~5 °C
        return float(math.exp(-((spread - 11.0) ** 2) / (2.0 * 5.5 ** 2)))

    def _hail_composite_score(self,
                              cape: float,
                              t_sfc_c: float,
                              dewpoint_c: float,
                              humidity: float,
                              wind_ms: float,
                              precip_mm: float,
                              moisture_factor: float,
                              pressure_trend: Optional[float]) -> float:
        """CAPE-driven hail potential with optional environment modifiers (no CAPE → no meaningful hail)."""
        cape_n = self._normalize_cape_for_hail(cape)
        if cape < 280:
            # Sub-thunderstorm parcel CAPE: still allow signals from ongoing convection proxies.
            spread = max(0.0, t_sfc_c - dewpoint_c)
            precip_only = min(1.0, precip_mm / 18.0) * min(1.0, max(0.0, humidity - 0.82) / 0.12)
            ongoing = 0.0
            if spread <= 4.5 and humidity >= 0.88 and wind_ms >= 11.0:
                # Warm, nearly saturated air and strong low-level flow — common in MCS / elevated hail.
                ongoing = min(
                    0.26,
                    0.11 + 0.055 * min(1.0, (wind_ms - 11.0) / 12.0) + 0.06 * (1.0 - spread / 4.5),
                )
            return float(max(min(0.12, 0.04 + 0.08 * precip_only), ongoing))

        instab = self._instability_weight_for_hail(t_sfc_c, dewpoint_c)
        moisture_n = min(1.0, 0.55 * moisture_factor + 0.35 * max(0.0, humidity - 0.45) / 0.5)
        td_boost = min(1.0, max(0.0, dewpoint_c + 5.0) / 30.0)
        shear_proxy = min(1.0, max(0.0, wind_ms - 4.0) / 22.0)
        precip_signal = min(1.0, precip_mm / 14.0)
        trend_boost = 0.1 if (pressure_trend is not None and pressure_trend < -0.4) else 0.0

        environment = (
            0.28 * instab
            + 0.32 * moisture_n
            + 0.18 * td_boost
            + 0.22 * shear_proxy
            + 0.12 * precip_signal
        )
        environment = min(1.0, environment)
        base = cape_n * (0.58 + 0.42 * environment) + trend_boost
        return float(max(0.0, min(1.0, base)))

    def _lightning_hail_coupling_boost(self, state: ForecastState, cape: float) -> float:
        """Upweight hail when lightning / LPI is present — same electrified updraft pathway as large hail.

        Operational systems (e.g. lightning fused with models) use this coupling; without a network we
        apply a conservative CAPE–precip–wind proxy. See state fields for optional measured inputs.
        """
        cape_n = self._normalize_cape_for_hail(cape)
        precip = state.precip_mm
        wind = state.wind_speed_m_s
        p = min(1.0, precip / 10.0)
        w = min(1.0, max(0.0, wind - 5.0) / 20.0)

        if state.lightning_potential_j_kg is not None and state.lightning_potential_j_kg > 0:
            lpi = float(state.lightning_potential_j_kg)
            lpi_n = min(1.0, lpi / 3200.0)
            raw = (0.11 + 0.24 * lpi_n) * (0.35 + 0.65 * max(cape_n, 0.12))
            return float(min(0.30, raw))

        if state.lightning_flashes_per_hour is not None and state.lightning_flashes_per_hour > 0:
            flashes = float(state.lightning_flashes_per_hour)
            flash_n = min(1.0, math.log1p(flashes) / math.log1p(1500.0))
            raw = (0.09 + 0.22 * flash_n) * (0.38 + 0.62 * max(cape_n, 0.12))
            return float(min(0.28, raw))

        # Proxy: active cell (precip) + CAPE + shear-like surface wind — no strike list required.
        if cape < 300 and precip < 0.4 and wind < 12.0:
            return 0.0
        raw = cape_n * (0.26 * p + 0.34 * w) + 0.04 * min(1.0, max(0.0, p * w))
        return float(min(0.12, raw))

    def _assess_severe_risk(self,
                            state: ForecastState,
                            analysis: Dict[str, Any],
                            pressure_trend: Optional[float],
                            moisture_factor: float,
                            cape: float) -> tuple[float, float, Dict[str, Any], float, Dict[str, float]]:
        humidity = state.relative_humidity
        temperature = state.surface_temp_c
        wind = state.wind_speed_m_s
        dewpoint = state.surface_dewpoint_c
        if dewpoint is None:
            dewpoint = float(analysis['moisture_analysis']['dewpoint_c'])

        env_score = self._hail_composite_score(
            cape, temperature, dewpoint, humidity, wind,
            state.precip_mm, moisture_factor, pressure_trend
        )
        hail_score = env_score
        tornado_score = 0.0

        has_negative_pressure = pressure_trend is not None and pressure_trend < 0
        shear_factor = min(1.0, wind / 35)

        # Legacy threshold boosts: reward very high CAPE / organized wind without double-counting.
        if cape > 800 and humidity > 0.58 and temperature >= 15:
            hail_score = max(hail_score, min(1.0, 0.22 + (cape - 800) / 3800 + moisture_factor * 0.12))

        if cape > 1800 and temperature >= 16 and wind >= 7:
            hail_score = max(hail_score, min(1.0, 0.35 + (cape - 1800) / 3200 + (wind - 7) / 28))

        if cape > 2600 and temperature >= 17:
            hail_score = max(hail_score, min(1.0, 0.45 + (cape - 2600) / 2800))

        if humidity > 0.82 and temperature >= 16 and wind >= 11 and pressure_trend is not None and pressure_trend < -0.25:
            hail_score = max(hail_score, min(1.0, 0.2 + (humidity - 0.82) * 1.2 + moisture_factor * 0.15))

        if state.precip_mm > 5.0:
            hail_score = max(hail_score, min(1.0, hail_score + 0.06))
            tornado_score = max(tornado_score, 0.05)

        if pressure_trend is not None and pressure_trend < -0.5:
            hail_score = min(1.0, hail_score + 0.1)
            if cape > 400:
                tornado_score = min(1.0, tornado_score + 0.08)

        if moisture_factor > 0.62 and pressure_trend is not None and pressure_trend < -0.25 and cape > 350:
            hail_score = max(hail_score, min(1.0, hail_score + 0.05))

        if cape > 350 and has_negative_pressure:
            tornado_score = min(1.0, tornado_score + shear_factor * 0.25)

        lightning_boost = self._lightning_hail_coupling_boost(state, cape)
        hail_score = min(1.0, hail_score + lightning_boost)

        # 4-Hour Predictive Spike (Updraft Helicity)
        # UH > 50 signifies a severe rotating updraft; UH > 100 is significant.
        if state.updraft_helicity is not None:
            uh_boost = min(0.35, state.updraft_helicity / 300.0)
            if state.updraft_helicity > 50:
                hail_score = max(hail_score, 0.70) # Baseline 70% accuracy trigger 4 hours out
                hail_score = min(1.0, hail_score + uh_boost)
        else:
            uh_boost = 0.0

        # Snippet-derived XGBoost Refinement
        # Features: Radar dBZ (proxy from analysis), CAPE, shear (proxy from wind), temp
        dbz_proxy = analysis.get('storm_analysis', {}).get('reflectivity_dbz_estimate', 45.0)
        xg_boost = predict_hail_xgboost(
            dbz=dbz_proxy,
            cape=cape,
            shear=wind * 1.5,  # Simple proxy for bulk shear
            temp=temperature
        )
        legacy_blend = (hail_score * 0.7) + (xg_boost * 0.3)
        hail_score = legacy_blend

        # S2S Pattern Baseline (Weeks-Ahead Outlook)
        # S2S provides a 'climatological floor' for the risk.
        hail_score = max(hail_score, self.s2s_baseline * 0.25)
        hail_score = min(1.0, hail_score)

        outlook = 0.0
        if state.graphcast_hail_potential_0_1 is not None:
            outlook = max(outlook, float(state.graphcast_hail_potential_0_1))
        if state.qu_outlook_hail_environment_0_1 is not None:
            outlook = max(outlook, float(state.qu_outlook_hail_environment_0_1))
        if outlook > 0.0:
            damp = 0.18
            if state.qu_outlook_uncertainty_0_1 is not None:
                damp *= max(0.35, 1.0 - 0.45 * float(state.qu_outlook_uncertainty_0_1))
            hail_score = min(1.0, hail_score + damp * outlook)

        sat_p: Optional[float] = None
        if state.lightning_severe_hail_prob_0_1 is not None:
            sat_p = float(state.lightning_severe_hail_prob_0_1)
        if state.qu_satellite_hail_nowcast_0_1 is not None:
            q = float(state.qu_satellite_hail_nowcast_0_1)
            sat_p = q if sat_p is None else max(sat_p, q)
        if sat_p is not None:
            hail_score = min(1.0, hail_score + satellite_hail_nowcast_boost(sat_p))

        environmental_hail_features = compute_environmental_hail_features(
            state=state,
            analysis=analysis,
            cape_j_kg=float(cape),
            moisture_factor=float(moisture_factor),
        )

        fused_features = {
            "bias": 1.0,
            "env_score": float(env_score),
            "xgboost_score": float(xg_boost),
            "s2s_baseline": float(self.s2s_baseline),
            "outlook_0_1": float(outlook),
            "satellite_nowcast_0_1": float(sat_p or 0.0),
            "lightning_boost": float(lightning_boost),
            "updraft_helicity_boost": float(uh_boost),
            "radar_lock_strength": 0.0,
            "supercooled_layer_depth_km": float(environmental_hail_features["supercooled_layer_depth_km"]),
            "updraft_survival_index": float(environmental_hail_features["updraft_survival_index"]),
            "hail_growth_potential_0_1": float(environmental_hail_features["hail_growth_potential_0_1"]),
            "downdraft_cooling_index": float(environmental_hail_features["downdraft_cooling_index"]),
            "storm_organization_proxy": float(environmental_hail_features["storm_organization_proxy"]),
        }
        hail_score = float(max(hail_score, self.hail_belief.predict_proba(fused_features)))

        hail_core = self._estimate_hail_core(state, analysis, hail_score, pressure_trend, moisture_factor)
        
        # Last-Mile Radar Localization (Spatial Tightening)
        hail_score, radar_note, radar_strength = self._apply_radar_localization(state, hail_score)
        fused_features["radar_lock_strength"] = float(radar_strength)
        hail_score = float(max(hail_score, self.hail_belief.predict_proba(fused_features)))
        if radar_note:
            hail_core['note'] += f" | {radar_note}"

        hail_core['risk_score'] = round((hail_score * 0.6 + tornado_score * 0.4) * 100, 2)
        hail_core['hail_probability'] = round(hail_score, 3)
        hail_core['tornado_probability'] = round(tornado_score, 3)
        hail_core['lightning_hail_boost'] = round(lightning_boost, 4)

        return round(hail_score, 3), round(tornado_score, 3), hail_core, lightning_boost, environmental_hail_features

    def _apply_radar_localization(self, state: ForecastState, current_score: float) -> tuple[float, str]:
        """Snaps the environment-based probability to real-time radar cell detections."""
        if not self.active_cells:
            return current_score, ""

        best_cell = None
        min_dist = float('inf')
        
        for cell in self.active_cells:
            # Check spatial containment or proximity
            # If we have a polygon, we use it. Otherwise, fallback to center_lat/lon distance.
            dist = self._haversine_km(state.latitude, state.longitude, cell['center_lat'], cell['center_lon'])
            
            # If the property is inside the storm's polygon, we have a 'Locked' hit
            is_contained = False
            if cell.get('polygon_geojson'):
                # Simple bounding box check for speed, would use Shapely in prod
                lats = [p[0] for p in cell['polygon_geojson']]
                lons = [p[1] for p in cell['polygon_geojson']]
                if min(lats) <= state.latitude <= max(lats) and min(lons) <= state.longitude <= max(lons):
                    is_contained = True
            
            if is_contained:
                best_cell = cell
                break
            
            if dist < min_dist:
                min_dist = dist
                best_cell = cell

        if best_cell:
            dist = self._haversine_km(state.latitude, state.longitude, best_cell['center_lat'], best_cell['center_lon'])
            
            if dist < 10:  # Within 10km of a severe core (The "Money Zone")
                # Boost probability and tighten window
                boost = 0.25 * (1.0 - dist / 10.0)
                new_score = min(1.0, current_score + boost)
                
                # Incorporate radar-derived intensity
                if best_cell.get('max_reflectivity_dbz', 0) > 60:
                    new_score = max(new_score, 0.85) # Extreme hail core confirmed
                
                return new_score, f"RADAR_LOCKED: {best_cell['storm_cell_id']} at {dist:.1f}km. dBZ={best_cell.get('max_reflectivity_dbz')}"
            
            elif dist < 30: # Within 30km (The "Proximity Zone")
                return min(1.0, current_score + 0.1), f"RADAR_PROXIMITY: Cell {best_cell['storm_cell_id']} at {dist:.1f}km"

        return current_score, ""

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c

    def _estimate_hail_core(self,
                            state: ForecastState,
                            analysis: Dict[str, Any],
                            hail_score: float,
                            pressure_trend: Optional[float],
                            moisture_factor: float) -> Dict[str, Any]:
        base_radius_m = 150.0
        if hail_score > 0.7:
            radius_m = 25.0 + (1.0 - hail_score) * 200.0
        elif hail_score > 0.4:
            radius_m = 100.0 + (0.7 - hail_score) * 200.0
        else:
            radius_m = 250.0 + (0.4 - hail_score) * 300.0

        if pressure_trend is not None and pressure_trend < -1.0:
            radius_m = max(20.0, radius_m - 50.0)

        if moisture_factor > 0.75:
            radius_m = max(20.0, radius_m - 25.0)

        confidence = min(1.0, hail_score * 0.75 + moisture_factor * 0.15 + 0.1)
        distance_ft = round(radius_m * 3.28084, 1)

        core_note = 'Localized hail core proxy. Not radar-resolved to 10 ft, but useful for short-range lead scoring.'
        if confidence > 0.7:
            core_note = 'Strong localized hail core signature. Use this for high-priority lead targeting.'
        elif confidence > 0.4:
            core_note = 'Moderate hail core likelihood. Validate with radar or observations.'

        return {
            'hail_core_confidence': round(confidence, 3),
            'hail_core_radius_ft': distance_ft,
            'hail_core_note': core_note,
            'note': core_note,
        }

    @classmethod
    def load_forecast_from_payload(cls, payload: Dict[str, Any]) -> 'RoofHunterWeatherTwin':
        states = [ForecastState.from_dict(item) for item in payload['forecast']]
        return cls(states)

    @classmethod
    def load_forecast(cls, path: Path) -> 'RoofHunterWeatherTwin':
        with path.open('r', encoding='utf-8') as f:
            payload = json.load(f)
        return cls.load_forecast_from_payload(payload)

    def export_results(self, path: Path) -> None:
        with path.open('w', encoding='utf-8') as f:
            json.dump({'history': self.history}, f, indent=2)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description='Run Roof Hunter weather twin simulation.')
    parser.add_argument('--forecast', type=Path, default=Path('roof_hunter_forecast.json'))
    parser.add_argument('--output', type=Path, default=Path('roof_hunter_results.json'))
    args = parser.parse_args()

    twin = RoofHunterWeatherTwin.load_forecast(args.forecast)
    history = twin.simulate()
    twin.export_results(args.output)
    print(f'Roof Hunter simulated {len(history)} forecast steps and saved results to {args.output}')


if __name__ == '__main__':
    main()
