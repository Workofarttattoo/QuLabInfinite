"""Validation helper for Roof Hunter using last-week weather data."""

from __future__ import annotations

import csv
import io
import json
import math
import sys
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure the project root is on sys.path when running this script directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roof_hunter.lightning_feeds import (
    apply_lightning_to_forecast_rows,
    build_glm_hourly_flash_lookup,
    load_lightning_sidecar,
)
from roof_hunter.integrations.qu_atmospheric_pipeline import enrich_forecast_payload
from roof_hunter.models import ForecastState
from roof_hunter.roof_hunter_digital_twin import RoofHunterWeatherTwin


def fetch_last_week_weather(latitude: float,
                            longitude: float,
                            end_date: date,
                            start_date: date | None = None) -> Dict[str, Any]:
    if start_date is None:
        start_date = end_date - timedelta(days=6)
    url = (
        f'https://archive-api.open-meteo.com/v1/archive?'
        f'latitude={latitude}&longitude={longitude}&'
        f'start_date={start_date}&end_date={end_date}&'
        f'hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,surface_pressure,precipitation,'
        f'wind_speed_10m,wind_direction_10m,lightning_potential&'
        f'timezone=UTC'
    )
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def build_forecast_payload(weather_data: Dict[str, Any], latitude: float, longitude: float) -> Dict[str, Any]:
    times = weather_data['hourly']['time']
    hourly = weather_data['hourly']
    lpi_series = hourly.get('lightning_potential')
    forecast = []
    prev_pressure = None
    prev_time = None

    for idx, timestamp in enumerate(times):
        temperature = weather_data['hourly']['temperature_2m'][idx]
        relative_humidity = weather_data['hourly']['relativehumidity_2m'][idx] / 100.0
        dewpoint = weather_data['hourly']['dewpoint_2m'][idx]
        pressure = weather_data['hourly']['surface_pressure'][idx]
        precip = weather_data['hourly']['precipitation'][idx]
        wind_speed = weather_data['hourly']['wind_speed_10m'][idx]
        wind_direction = weather_data['hourly']['wind_direction_10m'][idx]

        lpi_val = None
        if lpi_series is not None and idx < len(lpi_series):
            raw_lpi = lpi_series[idx]
            if raw_lpi is not None:
                lpi_val = float(raw_lpi)

        pressure_trend = 0.0
        if prev_pressure is not None and prev_time is not None:
            dt_hours = max(1.0, (datetime.fromisoformat(timestamp) - prev_time).total_seconds() / 3600.0)
            pressure_trend = (pressure - prev_pressure) / dt_hours

        state = {
            'timestamp': timestamp,
            'latitude': latitude,
            'longitude': longitude,
            'surface_temp_c': temperature,
            'relative_humidity': relative_humidity,
            'surface_dewpoint_c': dewpoint,
            'surface_pressure_hpa': pressure,
            'surface_pressure_trend_hpa_per_hour': pressure_trend,
            'precipitable_water_mm': None,
            'low_level_moisture_g_m3': None,
            'wind_speed_m_s': wind_speed,
            'wind_direction_deg': wind_direction,
            'precip_mm': precip,
            'lightning_potential_j_kg': lpi_val,
            'lightning_flashes_per_hour': None,
        }
        forecast.append(state)
        prev_pressure = pressure
        prev_time = datetime.fromisoformat(timestamp)

    return {'forecast': forecast}


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _parse_spc_report_datetime(report_date: date, time_str: str) -> datetime:
    digits = ''.join(ch for ch in time_str if ch.isdigit())
    if len(digits) <= 2:
        digits = digits.zfill(4)
    elif len(digits) == 3:
        digits = '0' + digits

    hour = int(digits[:2])
    minute = int(digits[2:])
    dt = datetime(report_date.year, report_date.month, report_date.day, hour, minute, tzinfo=timezone.utc)
    if hour < 12:
        dt += timedelta(days=1)
    return dt


def fetch_spc_daily_report(report_date: date) -> str:
    url = f'https://www.spc.noaa.gov/climo/reports/{report_date.strftime("%y%m%d")}_rpts.csv'
    req = urllib.request.Request(url, headers={'User-Agent': 'roof-hunter/1.0'})
    with urllib.request.urlopen(req) as response:
        return response.read().decode('utf-8', errors='replace')


def parse_spc_daily_report(csv_text: str, report_date: date) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    reader = csv.reader(io.StringIO(csv_text))
    section: str | None = None

    for row in reader:
        if not row:
            continue

        first_cell = row[0].strip()
        if first_cell == 'Time' and len(row) > 1:
            if 'Size' in row:
                section = 'hail'
                continue
            if 'F_Scale' in row or 'EF_Scale' in row:
                section = 'tornado'
                continue
            if 'Speed' in row:
                section = 'wind'
                continue

        if section not in ('hail', 'tornado'):
            continue

        if not first_cell or first_cell.lower().startswith('time') or 'No reports' in first_cell:
            continue

        try:
            report_dt = _parse_spc_report_datetime(report_date, first_cell)
        except Exception:
            continue

        if len(row) < 7:
            continue

        try:
            latitude = float(row[5])
            longitude = float(row[6])
        except ValueError:
            continue

        comments = row[7].strip() if len(row) > 7 else ''
        report: Dict[str, Any] = {
            'report_datetime': report_dt.isoformat(),
            'type': section,
            'location': row[2].strip() if len(row) > 2 else '',
            'county': row[3].strip() if len(row) > 3 else '',
            'state': row[4].strip() if len(row) > 4 else '',
            'lat': latitude,
            'lon': longitude,
            'comments': comments,
        }

        if section == 'hail':
            size_value = row[1].strip()
            hail_size_in = None
            if size_value and size_value.upper() != 'UNK':
                try:
                    hail_size_in = float(size_value) / 100.0
                except ValueError:
                    hail_size_in = None
            report['size_in'] = hail_size_in
        else:
            report['f_scale'] = row[1].strip()

        reports.append(report)

    return reports


def fetch_spc_reports(latitude: float,
                      longitude: float,
                      start_date: date,
                      end_date: date,
                      radius_km: float = 100.0) -> List[Dict[str, Any]]:
    all_reports: List[Dict[str, Any]] = []
    current_date = start_date
    while current_date <= end_date:
        try:
            csv_text = fetch_spc_daily_report(current_date)
            daily_reports = parse_spc_daily_report(csv_text, current_date)
            all_reports.extend(daily_reports)
        except Exception:
            pass
        current_date += timedelta(days=1)

    filtered = []
    start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)

    for report in all_reports:
        report_dt = datetime.fromisoformat(report['report_datetime'])
        if report_dt.tzinfo is None:
            report_dt = report_dt.replace(tzinfo=timezone.utc)
        if not (start_dt <= report_dt <= end_dt):
            continue
        distance = _haversine_km(latitude, longitude, report['lat'], report['lon'])
        if distance <= radius_km:
            report['distance_km'] = round(distance, 2)
            filtered.append(report)

    return filtered


def match_reports_to_forecast(history: List[Dict[str, Any]],
                              reports: List[Dict[str, Any]],
                              time_window_hours: float = 1.5,
                              match_radius_km: float = 40.0) -> Dict[str, Any]:
    matches: List[Dict[str, Any]] = []
    for report in reports:
        report_dt = datetime.fromisoformat(report['report_datetime'])
        if report_dt.tzinfo is None:
            report_dt = report_dt.replace(tzinfo=timezone.utc)
        best_match = None
        for state in history:
            state_dt = datetime.fromisoformat(state['timestamp'])
            if state_dt.tzinfo is None:
                state_dt = state_dt.replace(tzinfo=timezone.utc)
            dt_hours = abs((state_dt - report_dt).total_seconds()) / 3600.0
            if dt_hours > time_window_hours:
                continue
            distance = _haversine_km(state['latitude'], state['longitude'], report['lat'], report['lon'])
            if distance > match_radius_km:
                continue
            score = dt_hours + distance / 50.0
            if best_match is None or score < best_match['score']:
                best_match = {
                    'report': report,
                    'state': state,
                    'dt_hours': round(dt_hours, 2),
                    'distance_km': round(distance, 2),
                    'score': score,
                }

        if best_match is not None:
            matches.append({
                'report_datetime': best_match['report']['report_datetime'],
                'report_type': best_match['report']['type'],
                'report_location': best_match['report']['location'],
                'report_county': best_match['report']['county'],
                'report_state': best_match['report']['state'],
                'report_comments': best_match['report']['comments'],
                'matched_timestamp': best_match['state']['timestamp'],
                'matched_hail_probability': best_match['state']['hail_probability'],
                'matched_tornado_probability': best_match['state']['tornado_probability'],
                'time_diff_hours': best_match['dt_hours'],
                'distance_km': best_match['distance_km'],
                'forecast_risk_score': best_match['state']['risk_score'],
            })

    total_reports = len(reports)
    total_matches = len(matches)
    hail_pred_count = len([s for s in history if s['hail_probability'] >= 0.2])
    tornado_pred_count = len([s for s in history if s['tornado_probability'] >= 0.1])
    
    # ML Validation Metrics (Snippet Extraction)
    y_true = [1 if any(r['type'] == 'hail' and (r.get('size_in', 0) or 0) >= 1.0 for r in reports) else 0] # Simplified for single location
    # Real evaluation would be across multiple points/times
    y_pred = [1 if any(s['hail_probability'] >= 0.5 for s in history) else 0]
    
    # For now, we'll calculate simple stats if we have enough data
    precision = 1.0 if (y_true[0] == 1 and y_pred[0] == 1) else 0.0
    recall = 1.0 if (y_true[0] == 1 and y_pred[0] == 1) else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'summary': {
            'report_count': total_reports,
            'matched_report_count': total_matches,
            'hail_predictions': hail_pred_count,
            'tornado_predictions': tornado_pred_count,
            'match_rate': round((total_matches / total_reports) if total_reports else 0.0, 3),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'report_time_window_hours': time_window_hours,
            'report_match_radius_km': match_radius_km,
        },
        'matches': matches,
        'reports': reports,
    }


def fetch_noaa_last_week_weather(latitude: float,
                                 longitude: float,
                                 end_date: date,
                                 start_date: date | None = None) -> Dict[str, Any]:
    if start_date is None:
        start_date = end_date - timedelta(days=6)
    points_url = f'https://api.weather.gov/points/{latitude},{longitude}'
    req = urllib.request.Request(points_url, headers={'User-Agent': 'roof-hunter/1.0'})
    with urllib.request.urlopen(req) as response:
        point_data = json.load(response)

    stations_url = point_data['properties']['observationStations']
    req = urllib.request.Request(stations_url, headers={'User-Agent': 'roof-hunter/1.0'})
    with urllib.request.urlopen(req) as response:
        stations_data = json.load(response)

    stations = stations_data.get('features', [])
    if not stations:
        raise RuntimeError('No NOAA observation stations available for location')

    nearest = min(stations, key=lambda feature: feature['properties']['distance']['value'])
    station_id = nearest['properties']['stationIdentifier']

    obs_url = (
        f'https://api.weather.gov/stations/{station_id}/observations?'
        f'start={start_date.isoformat()}T00:00:00Z&end={end_date.isoformat()}T23:59:59Z'
    )
    req = urllib.request.Request(obs_url, headers={'User-Agent': 'roof-hunter/1.0'})
    with urllib.request.urlopen(req) as response:
        return json.load(response)


def build_forecast_payload_from_noaa(observation_data: Dict[str, Any], latitude: float, longitude: float) -> Dict[str, Any]:
    hourly = {}
    for feature in observation_data.get('features', []):
        props = feature['properties']
        timestamp = props['timestamp']
        try:
            dt = datetime.fromisoformat(timestamp)
        except ValueError:
            continue

        key = dt.replace(minute=0, second=0, microsecond=0)
        current = hourly.get(key)
        if current is None or dt > current['dt']:
            hourly[key] = {'dt': dt, 'props': props}

    forecast = []
    for key in sorted(hourly):
        props = hourly[key]['props']
        temp = props.get('temperature', {}).get('value')
        humidity = props.get('relativeHumidity', {}).get('value')
        dewpoint = props.get('dewpoint', {}).get('value')
        pressure_pa = props.get('barometricPressure', {}).get('value')
        if pressure_pa is None:
            pressure_pa = props.get('seaLevelPressure', {}).get('value')
        pressure_hpa = pressure_pa / 100.0 if pressure_pa is not None else 1013.25
        wind_speed_kmh = props.get('windSpeed', {}).get('value')
        wind_speed = wind_speed_kmh / 3.6 if wind_speed_kmh is not None else 0.0
        wind_dir = props.get('windDirection', {}).get('value') or 0.0
        precip = props.get('precipitationLastHour', {}).get('value') or 0.0

        if temp is None or humidity is None or dewpoint is None:
            continue

        forecast.append({
            'timestamp': key.isoformat(),
            'latitude': latitude,
            'longitude': longitude,
            'surface_temp_c': float(temp),
            'relative_humidity': float(humidity) / 100.0,
            'surface_dewpoint_c': float(dewpoint),
            'surface_pressure_hpa': float(pressure_hpa),
            'surface_pressure_trend_hpa_per_hour': None,
            'precipitable_water_mm': None,
            'low_level_moisture_g_m3': None,
            'wind_speed_m_s': float(wind_speed),
            'wind_direction_deg': float(wind_dir),
            'precip_mm': float(precip),
            'lightning_potential_j_kg': None,
            'lightning_flashes_per_hour': None,
        })

    # Compute pressure trend when possible
    for i in range(1, len(forecast)):
        prev = forecast[i - 1]
        curr = forecast[i]
        dt = (datetime.fromisoformat(curr['timestamp']) - datetime.fromisoformat(prev['timestamp'])).total_seconds() / 3600.0
        if dt > 0:
            curr['surface_pressure_trend_hpa_per_hour'] = (curr['surface_pressure_hpa'] - prev['surface_pressure_hpa']) / dt

    return {'forecast': forecast}


def run_validation(latitude: float = 35.0,
                   longitude: float = -97.0,
                   end_date: date | None = None,
                   window_days: int = 7,
                   source: str = 'open-meteo',
                   compare_reports: bool = False,
                   report_radius_km: float = 100.0,
                   report_match_radius_km: float = 40.0,
                   report_match_hours: float = 1.5,
                   lightning_glm: bool = True,
                   lightning_sidecar: Optional[str] = None,
                   glm_radius_km: float = 25.0,
                   glm_satellite: str = 'goes16',
                   glm_cache_dir: Optional[str] = None,
                   enrich_qu: bool = False) -> None:
    if end_date is None:
        end_date = date.today() - timedelta(days=1)
    window_days = max(1, window_days)
    start_date = end_date - timedelta(days=window_days - 1)

    if source == 'noaa':
        weather_data = fetch_noaa_last_week_weather(latitude, longitude, end_date, start_date)
        payload = build_forecast_payload_from_noaa(weather_data, latitude, longitude)
    else:
        weather_data = fetch_last_week_weather(latitude, longitude, end_date, start_date)
        payload = build_forecast_payload(weather_data, latitude, longitude)

    glm_lookup = None
    if lightning_glm:
        cache = Path(glm_cache_dir) if glm_cache_dir else Path('roof_hunter/.lightning_cache')
        print(
            f'Fetching GOES GLM flash counts (satellite={glm_satellite}, radius={glm_radius_km:g} km, '
            f'cache={cache}) — this downloads many small NetCDF files; first run can be slow.',
            flush=True,
        )
        glm_lookup = build_glm_hourly_flash_lookup(
            payload['forecast'],
            latitude,
            longitude,
            radius_km=glm_radius_km,
            satellite=glm_satellite,
            cache_dir=cache,
            log=lambda m: print(m, flush=True),
        )
    sidecar = None
    if lightning_sidecar:
        sidecar = load_lightning_sidecar(Path(lightning_sidecar))
    if glm_lookup or sidecar:
        apply_lightning_to_forecast_rows(payload['forecast'], glm_lookup, sidecar)

    if enrich_qu:
        payload = enrich_forecast_payload(payload)
        print('Applied QuLab enrich_forecast_payload (global outlook + satellite hooks)', flush=True)

    forecast_path = Path('roof_hunter/roof_hunter_last_week_forecast.json')
    results_path = Path('roof_hunter/roof_hunter_last_week_results.json')

    save_json(forecast_path, payload)
    print(f'Saved last-week forecast to {forecast_path}')

    twin = RoofHunterWeatherTwin([ForecastState.from_dict(item) for item in payload['forecast']])
    history = twin.simulate()
    twin.export_results(results_path)
    print(f'Validation simulated {len(history)} steps and wrote results to {results_path}')

    if compare_reports:
        spc_reports = fetch_spc_reports(latitude, longitude, start_date, end_date, radius_km=report_radius_km)
        matches = match_reports_to_forecast(history, spc_reports,
                                           time_window_hours=report_match_hours,
                                           match_radius_km=report_match_radius_km)
        report_match_path = Path('roof_hunter/roof_hunter_last_week_report_matches.json')
        save_json(report_match_path, matches)
        print(f'Compared forecast to {len(spc_reports)} SPC hail/tornado reports and saved matches to {report_match_path}')
        print('Summary:', matches['summary'])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Validate Roof Hunter with last-week weather data.')
    parser.add_argument('--lat', type=float, default=35.0)
    parser.add_argument('--lon', type=float, default=-97.0)
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date in YYYY-MM-DD format')
    parser.add_argument('--source', type=str, choices=['open-meteo', 'noaa'], default='noaa')
    parser.add_argument('--window-days', type=int, default=7,
                        help='Length of the validation window in days')
    parser.add_argument('--compare-reports', action='store_true', help='Fetch SPC hail/tornado reports and compare them to the forecast')
    parser.add_argument('--report-radius-km', type=float, default=100.0,
                        help='Search radius around the forecast location for SPC reports')
    parser.add_argument('--report-match-radius-km', type=float, default=40.0,
                        help='Maximum distance in kilometers to match a report to a forecast state')
    parser.add_argument('--report-match-hours', type=float, default=1.5,
                        help='Maximum time window in hours to match a report to a forecast state')
    parser.set_defaults(lightning_glm=True)
    parser.add_argument('--lightning-glm', dest='lightning_glm', action='store_true',
                        help='Merge hourly GOES GLM L2 flash counts (default: on; needs: pip install netCDF4)')
    parser.add_argument('--no-lightning-glm', dest='lightning_glm', action='store_false',
                        help='Skip GOES GLM merge (faster; no flash counts from satellite)')
    parser.add_argument('--lightning-sidecar', type=str, default=None,
                        help='JSON file with per-hour lightning_flashes_per_hour / lightning_potential_j_kg (see LIGHTNING_DATA_FEEDS.md)')
    parser.add_argument('--glm-radius-km', type=float, default=25.0,
                        help='Radius around the grid point for counting GLM flashes')
    parser.add_argument('--glm-satellite', type=str, choices=['goes16', 'goes18'], default='goes16',
                        help='S3 bucket for GLM (goes16=East, goes18=West)')
    parser.add_argument('--glm-cache-dir', type=str, default=None,
                        help='Cache directory for GLM hourly aggregates (default: roof_hunter/.lightning_cache)')
    parser.add_argument('--enrich-qu', action='store_true',
                        help='Run enrich_forecast_payload (QuLab outlook + satellite nowcast fields) before the twin')
    args = parser.parse_args()

    end_date = date.fromisoformat(args.end_date) if args.end_date else None
    run_validation(latitude=args.lat,
                   longitude=args.lon,
                   end_date=end_date,
                   window_days=args.window_days,
                   source=args.source,
                   compare_reports=args.compare_reports,
                   report_radius_km=args.report_radius_km,
                   report_match_radius_km=args.report_match_radius_km,
                   report_match_hours=args.report_match_hours,
                   lightning_glm=args.lightning_glm,
                   lightning_sidecar=args.lightning_sidecar,
                   glm_radius_km=args.glm_radius_km,
                   glm_satellite=args.glm_satellite,
                   glm_cache_dir=args.glm_cache_dir,
                   enrich_qu=args.enrich_qu)
