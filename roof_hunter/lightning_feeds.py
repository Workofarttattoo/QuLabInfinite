"""Lightning and LPI data feeds for the Roof Hunter digital twin.

See ``LIGHTNING_DATA_FEEDS.md`` in this directory for human-oriented explanations
of each source, licensing, and links.

This module wires:

* **Open-Meteo** ``lightning_potential`` (handled in ``validate_last_week.build_forecast_payload``;
  values are often null over CONUS in archive — still request it for regions that populate it).
* **GOES-R GLM** Level 2 LCFA (flashes near a point, aggregated per UTC hour).
* **Sidecar JSON** for third-party strike/LPI columns you supply offline.

Optional dependency for GLM: ``pip install netCDF4`` (or ``-r roof_hunter/requirements-lightning.txt``).
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

S3_NS = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
GLM_BUCKETS = {
    'goes16': 'https://noaa-goes16.s3.amazonaws.com',
    'goes18': 'https://noaa-goes18.s3.amazonaws.com',
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _norm_hour_key(ts: str) -> str:
    """Match Open-Meteo archive timestamps like ``2026-04-18T00:00`` (no trailing Z)."""
    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime('%Y-%m-%dT%H:%M')


def _utc_floor_hour(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.replace(minute=0, second=0, microsecond=0)


def _s3_list_keys(bucket_url: str, prefix: str, max_keys: int = 1000) -> List[str]:
    keys: List[str] = []
    token: Optional[str] = None
    while True:
        q = f'?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys={max_keys}'
        if token:
            q += f'&continuation-token={urllib.parse.quote(token)}'
        url = bucket_url.rstrip('/') + '/' + q
        with urllib.request.urlopen(url, timeout=120) as resp:
            root = ET.fromstring(resp.read().decode())
        for contents in root.findall('.//s:Contents', S3_NS):
            key_el = contents.find('s:Key', S3_NS)
            if key_el is not None and key_el.text:
                keys.append(key_el.text)
        is_trunc = root.find('s:IsTruncated', S3_NS)
        if is_trunc is not None and is_trunc.text == 'true':
            nxt = root.find('s:NextContinuationToken', S3_NS)
            token = nxt.text if nxt is not None else None
            if not token:
                break
        else:
            break
    return keys


def _glm_prefix_for_utc_hour(utc_hour: datetime) -> str:
    utc_hour = _utc_floor_hour(utc_hour)
    y = utc_hour.year
    doy = int(utc_hour.strftime('%j'))
    h = utc_hour.hour
    return f'GLM-L2-LCFA/{y}/{doy:03d}/{h:02d}/'


def _count_flashes_near_point_nc(path: str, lat: float, lon: float, radius_km: float) -> int:
    from netCDF4 import Dataset

    ds = Dataset(path, 'r')
    try:
        fla = ds.variables['flash_lat'][:]
        flo = ds.variables['flash_lon'][:]
        if getattr(fla, 'mask', False):
            fla = fla.data
            flo = flo.data
        if fla.size == 0:
            return 0
        if hasattr(fla, 'astype'):
            import numpy as np

            flat = np.asarray(fla, dtype=float)
            flon = np.asarray(flo, dtype=float)
            # vectorized haversine (small arrays; ~few flashes per 20s file)
            dlat = np.radians(flat - lat)
            dlon = np.radians(flon - lon)
            lat1 = np.radians(lat)
            lat2 = np.radians(flat)
            a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
            dist_km = 6371.0 * c
            return int(np.sum(dist_km <= radius_km))
        n = 0
        for a, o in zip(fla, flo):
            try:
                fa, fo = float(a), float(o)
            except (TypeError, ValueError):
                continue
            if not (-90.0 <= fa <= 90.0 and -180.0 <= fo <= 180.0):
                continue
            if haversine_km(lat, lon, fa, fo) <= radius_km:
                n += 1
        return n
    finally:
        ds.close()


def _download_to_temp(url: str) -> str:
    fd, path = tempfile.mkstemp(suffix='.nc')
    os.close(fd)
    urllib.request.urlretrieve(url, path)
    return path


def glm_flashes_one_utc_hour(
    utc_hour: datetime,
    latitude: float,
    longitude: float,
    radius_km: float = 25.0,
    satellite: str = 'goes16',
    cache_dir: Optional[Path] = None,
) -> Tuple[int, int]:
    """Sum GLM L2-LCFA flash detections within *radius_km* of a point for one UTC hour.

    Returns ``(flash_count, nc_files_read)``.

    Files are processed **sequentially**; the netCDF/HDF5 stack is not reliably safe
    under parallel ``Dataset`` access in one interpreter.
    """
    try:
        from netCDF4 import Dataset  # noqa: F401
    except ImportError as e:
        raise ImportError(
            'GLM ingestion requires netCDF4. Install with: '
            'pip install netCDF4   or   pip install -r roof_hunter/requirements-lightning.txt'
        ) from e

    utc_hour = _utc_floor_hour(utc_hour)
    bucket = GLM_BUCKETS.get(satellite)
    if not bucket:
        raise ValueError(f'satellite must be one of {list(GLM_BUCKETS)}, got {satellite!r}')

    y, doy, h = utc_hour.year, int(utc_hour.strftime('%j')), utc_hour.hour
    cache_key = f'{satellite}_{latitude:.4f}_{longitude:.4f}_r{radius_km:.0f}_{y}_{doy:03d}_{h:02d}.json'
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / cache_key
        if cache_path.is_file():
            meta = json.loads(cache_path.read_text(encoding='utf-8'))
            return int(meta['count']), int(meta['files'])

    prefix = _glm_prefix_for_utc_hour(utc_hour)
    keys = [k for k in _s3_list_keys(bucket, prefix) if k.endswith('.nc')]
    if not keys:
        if cache_dir:
            cache_path.write_text(json.dumps({'count': 0, 'files': 0}), encoding='utf-8')
        return 0, 0

    total = 0

    def job(key: str) -> int:
        url = f'{bucket}/{urllib.parse.quote(key)}'
        path = _download_to_temp(url)
        try:
            return _count_flashes_near_point_nc(path, latitude, longitude, radius_km)
        finally:
            os.unlink(path)

    for key in keys:
        total += job(key)

    if cache_dir:
        cache_path.write_text(json.dumps({'count': total, 'files': len(keys)}), encoding='utf-8')

    return total, len(keys)


def build_glm_hourly_flash_lookup(
    forecast_entries: List[Dict[str, Any]],
    latitude: float,
    longitude: float,
    radius_km: float = 25.0,
    satellite: str = 'goes16',
    cache_dir: Optional[Path] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, int]:
    """Build ``timestamp_hour_key -> flash_count`` for every unique UTC hour in *forecast_entries*."""
    unique_hours: List[datetime] = []
    seen = set()
    for row in forecast_entries:
        dt = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
        fh = _utc_floor_hour(dt)
        t = fh.isoformat().replace('+00:00', '')
        if t not in seen:
            seen.add(t)
            unique_hours.append(fh)

    out: Dict[str, int] = {}
    for i, fh in enumerate(sorted(unique_hours)):
        if log:
            log(f'GLM {i + 1}/{len(unique_hours)} UTC hour {fh.isoformat()} …')
        cnt, nfiles = glm_flashes_one_utc_hour(
            fh, latitude, longitude, radius_km, satellite=satellite, cache_dir=cache_dir
        )
        key = _norm_hour_key(fh.isoformat())
        out[key] = cnt
        if log:
            log(f'  -> {cnt} flashes (<= {radius_km:g} km), {nfiles} NetCDF granules')
    return out


def load_lightning_sidecar(path: Path) -> Dict[str, Dict[str, float]]:
    """Load sidecar JSON. Supported shapes:

    **Object with hours:**

        {"version": 1, "hours": {"2026-04-18T00:00": {"lightning_flashes_per_hour": 10, "lightning_potential_j_kg": 500}}}

    **List of rows:**

        [{"timestamp": "2026-04-18T00:00", "lightning_flashes_per_hour": 10}, ...]
    """
    raw = json.loads(path.read_text(encoding='utf-8'))
    by_hour: Dict[str, Dict[str, float]] = {}
    if isinstance(raw, dict) and 'hours' in raw:
        for k, v in raw['hours'].items():
            nk = _norm_hour_key(str(k))
            by_hour[nk] = {kk: float(vv) for kk, vv in v.items() if vv is not None}
    elif isinstance(raw, list):
        for row in raw:
            ts = row.get('timestamp')
            if not ts:
                continue
            nk = _norm_hour_key(str(ts))
            bucket: Dict[str, float] = {}
            if row.get('lightning_flashes_per_hour') is not None:
                bucket['lightning_flashes_per_hour'] = float(row['lightning_flashes_per_hour'])
            if row.get('lightning_potential_j_kg') is not None:
                bucket['lightning_potential_j_kg'] = float(row['lightning_potential_j_kg'])
            if bucket:
                by_hour[nk] = bucket
    else:
        raise ValueError('Sidecar must be a dict with "hours" or a list of timestamp rows')
    return by_hour


def apply_lightning_to_forecast_rows(
    forecast_entries: List[Dict[str, Any]],
    glm_by_hour: Optional[Dict[str, int]] = None,
    sidecar_by_hour: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """Mutates *forecast_entries* in place: sets ``lightning_flashes_per_hour`` / ``lightning_potential_j_kg``."""
    for row in forecast_entries:
        key = _norm_hour_key(row['timestamp'])
        if glm_by_hour and key in glm_by_hour:
            row['lightning_flashes_per_hour'] = float(glm_by_hour[key])
        if sidecar_by_hour and key in sidecar_by_hour:
            for k, v in sidecar_by_hour[key].items():
                row[k] = v
