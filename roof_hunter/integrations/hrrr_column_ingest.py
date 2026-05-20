"""HRRR pressure-column ingest for hail-physics features (0-48h lead workflows).

Builds a compact feature vector from HRRR pressure-level fields at a point:
- lapse rate proxy (700-500 mb)
- deep-layer shear proxy (1000-500 mb)
- vertical velocity -> updraft speed proxy (700/500 mb omega)
- CAPE proxy for hail physics initialization
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from roof_hunter.integrations.nomads_hrrr_ingest import (
    _download_stitched_grib,
    _kelvin_to_c,
    _nearest_scalar,
    _open_field,
)

RD = 287.05
G = 9.80665


def _pressure_file_urls(run_date: str, cycle_hh: str, fxx: int) -> Tuple[str, str]:
    base = f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{run_date}/conus"
    fname = f"hrrr.t{cycle_hh}z.wrfprsf{fxx:02d}.grib2"
    return f"{base}/{fname}", f"{base}/{fname}.idx"


def _url_exists(url: str) -> bool:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "QuLab-RoofHunter/1.1"})
    try:
        with urllib.request.urlopen(req, timeout=25):
            return True
    except urllib.error.HTTPError:
        return False
    except urllib.error.URLError:
        return False


def _resolve_available_run(run_time_utc: datetime) -> datetime:
    """Find latest available HRRR run by probing f00 presence, walking back up to 36h."""
    base = run_time_utc.replace(minute=0, second=0, microsecond=0)
    for back in range(0, 37):
        cand = base - timedelta(hours=back)
        date_s = cand.strftime("%Y%m%d")
        cyc = cand.strftime("%H")
        grib_url, _ = _pressure_file_urls(date_s, cyc, 0)
        if _url_exists(grib_url):
            return cand
    return base


def _rh_to_dewpoint_c(temp_c: float, rh_0_1: float) -> float:
    rh = min(0.999, max(0.05, rh_0_1))
    es = 6.112 * math.exp((17.67 * temp_c) / (temp_c + 243.5))
    e = rh * es
    b = math.log(max(1e-6, e / 6.112))
    return float((243.5 * b) / (17.67 - b))


def _omega_to_w_ms(omega_pa_s: float, pressure_hpa: float, temp_k: float) -> float:
    """Convert pressure vertical velocity (Pa/s) to approximate geometric w (m/s)."""
    p_pa = max(1.0, pressure_hpa * 100.0)
    return float((-omega_pa_s) * RD * temp_k / (p_pa * G))


def _read_isobaric(path: Path, short_name: str, level_hpa: int, lat: float, lon: float) -> Optional[float]:
    keys = {"shortName": short_name, "typeOfLevel": "isobaricInhPa", "level": level_hpa}
    try:
        ds = _open_field(path, keys)
    except Exception:
        return None
    try:
        return _nearest_scalar(ds, lat, lon)
    finally:
        ds.close()


def fetch_hrrr_column_features(
    latitude: float,
    longitude: float,
    run_time_utc: Union[str, datetime],
    fxx_hours: Sequence[int],
) -> List[Dict[str, Any]]:
    """Return one feature dict per forecast hour for physics + lead scoring."""
    if isinstance(run_time_utc, datetime):
        dt = run_time_utc.astimezone(timezone.utc)
    else:
        dt = datetime.strptime(run_time_utc.strip(), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    dt = _resolve_available_run(dt)
    date_s = dt.strftime("%Y%m%d")
    cycle = dt.strftime("%H")

    needles = (
        ":TMP:500 mb:",
        ":TMP:700 mb:",
        ":TMP:850 mb:",
        ":TMP:1000 mb:",
        ":RH:700 mb:",
        ":RH:850 mb:",
        ":RH:1000 mb:",
        ":UGRD:500 mb:",
        ":VGRD:500 mb:",
        ":UGRD:1000 mb:",
        ":VGRD:1000 mb:",
        ":VVEL:500 mb:",
        ":VVEL:700 mb:",
        ":VVEL:850 mb:",
    )

    cache_dir = Path.cwd() / ".roof_hunter_grib_cache"
    cache_dir.mkdir(exist_ok=True)

    out: List[Dict[str, Any]] = []
    for fxx in fxx_hours:
        grib_url, idx_url = _pressure_file_urls(date_s, cycle, int(fxx))
        dest = cache_dir / f"hrrr_prs_{date_s}_{cycle}_{int(fxx):02d}_subset.grib2"
        try:
            if not dest.exists() or dest.stat().st_size < 100:
                _download_stitched_grib(grib_url, idx_url, needles, dest)
        except Exception:
            # Missing horizon for this cycle (or transient fetch failure): skip this lead hour.
            continue

        t500 = _read_isobaric(dest, "t", 500, latitude, longitude)
        t700 = _read_isobaric(dest, "t", 700, latitude, longitude)
        t850 = _read_isobaric(dest, "t", 850, latitude, longitude)
        t1000 = _read_isobaric(dest, "t", 1000, latitude, longitude)
        rh850 = _read_isobaric(dest, "r", 850, latitude, longitude)
        rh1000 = _read_isobaric(dest, "r", 1000, latitude, longitude)
        u500 = _read_isobaric(dest, "u", 500, latitude, longitude)
        v500 = _read_isobaric(dest, "v", 500, latitude, longitude)
        u1000 = _read_isobaric(dest, "u", 1000, latitude, longitude)
        v1000 = _read_isobaric(dest, "v", 1000, latitude, longitude)
        om500 = _read_isobaric(dest, "w", 500, latitude, longitude)
        om700 = _read_isobaric(dest, "w", 700, latitude, longitude)
        om850 = _read_isobaric(dest, "w", 850, latitude, longitude)

        if t500 is None or t700 is None or t850 is None:
            continue

        t500c = _kelvin_to_c(t500)
        t700c = _kelvin_to_c(t700)
        t850c = _kelvin_to_c(t850)
        t1000c = _kelvin_to_c(t1000) if t1000 is not None else (t850c + 2.0)

        # Approx 700-500 thickness around 2.5 km.
        lapse_700_500 = (t700c - t500c) / 2.5

        rh850_0_1 = (rh850 / 100.0) if rh850 is not None else 0.65
        rh1000_0_1 = (rh1000 / 100.0) if rh1000 is not None else 0.65
        td1000c = _rh_to_dewpoint_c(t1000c, rh1000_0_1)

        shear_1000_500 = 0.0
        if None not in (u500, v500, u1000, v1000):
            du = float(u500) - float(u1000)
            dv = float(v500) - float(v1000)
            shear_1000_500 = math.hypot(du, dv)

        w_list: List[float] = []
        if om500 is not None:
            w_list.append(_omega_to_w_ms(float(om500), 500.0, float(t500)))
        if om700 is not None:
            w_list.append(_omega_to_w_ms(float(om700), 700.0, float(t700)))
        if om850 is not None:
            w_list.append(_omega_to_w_ms(float(om850), 850.0, float(t850)))
        updraft_ms = max([0.0] + [w for w in w_list if w > 0.0])

        # CAPE proxy tuned for lead ranking (not sounding-derived CAPE).
        instability = max(0.0, lapse_700_500 - 6.0) * 280.0
        moisture = max(0.0, (rh850_0_1 - 0.45) * 2200.0)
        shear_bonus = min(450.0, max(0.0, shear_1000_500 - 12.0) * 16.0)
        updraft_bonus = min(700.0, updraft_ms * 45.0)
        cape_proxy = max(150.0, min(6000.0, 200.0 + instability + moisture + shear_bonus + updraft_bonus))

        ts = dt + timedelta(hours=int(fxx))

        out.append(
            {
                "timestamp": ts.isoformat(),
                "latitude": latitude,
                "longitude": longitude,
                "surface_temp_c": round(t1000c, 2),
                "surface_dewpoint_c": round(td1000c, 2),
                "relative_humidity": round(rh1000_0_1, 4),
                "t850_c": round(t850c, 2),
                "rh850": round(rh850_0_1, 4),
                "lapse_700_500_k_per_km": round(lapse_700_500, 3),
                "bulk_shear_1000_500_ms": round(shear_1000_500, 2),
                "updraft_speed_ms": round(updraft_ms, 3),
                "cape_proxy_j_kg": round(cape_proxy, 1),
            }
        )
    return out


def write_column_features_json(features: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"column_features": list(features)}, indent=2), encoding="utf-8")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Fetch HRRR pressure-column features for hail-physics input.")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--run", type=str, required=True, help="UTC run, e.g. '2026-05-06 12:00'")
    p.add_argument("--fxx", type=str, default="0,3,6,9,12,18,24,30,36,42,48")
    p.add_argument("--output", type=Path, default=Path("roof_hunter_column_features.json"))
    args = p.parse_args()

    hours = [int(x.strip()) for x in args.fxx.split(",") if x.strip()]
    feats = fetch_hrrr_column_features(args.lat, args.lon, args.run, hours)
    write_column_features_json(feats, args.output)
    print(f"Wrote {len(feats)} HRRR column feature rows to {args.output}")


if __name__ == "__main__":
    main()

