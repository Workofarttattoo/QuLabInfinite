"""QuLab proprietary HRRR CONUS ingest: NOAA AWS open-data GRIB2 + .idx byte-range subsetting.

Downloads only the GRIB messages needed for 2 m T/Td, 10 m wind, and surface pressure by
parsing the companion ``.idx`` file and issuing HTTP ``Range`` requests — typically a few MB
instead of a full 80–120 MB cycle file.

Public NOAA bucket: ``noaa-hrrr-bdp-pds`` on AWS. Requires ``numpy``, ``xarray``, ``cfgrib``
(eccodes) to decode the stitched GRIB bytes.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

from roof_hunter.models import ForecastState

UA = "QuLab-RoofHunter/1.1 (+https://github.com/QuLab)"


def _require_numpy() -> None:
    if np is None:
        raise ImportError("nomads_hrrr_ingest requires numpy")


def _http_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 180) -> bytes:
    h = {"User-Agent": UA}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_head_clen(url: str, timeout: int = 90) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        cl = resp.headers.get("Content-Length")
        if not cl:
            raise RuntimeError(f"No Content-Length for {url}")
        return int(cl)


def _parse_idx_for_variables(idx_text: str, needles: Sequence[str], clen: int) -> Tuple[int, int]:
    """Return one inclusive byte range [start, end] covering all matching GRIB messages in order.

    Non-contiguous GRIB messages cannot be concatenated into a valid file; we download the full
    contiguous span from the first to the last matching message (preserving original byte order).
    """
    rows: List[Tuple[int, str]] = []
    for line in idx_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(":")
        if len(parts) < 3:
            continue
        try:
            off = int(parts[1])
        except ValueError:
            continue
        desc = ":".join(parts[2:])
        rows.append((off, desc))
    rows.sort(key=lambda x: x[0])
    hits: List[Tuple[int, int]] = []
    for i, (off, desc) in enumerate(rows):
        if not any(n in desc for n in needles):
            continue
        end = (rows[i + 1][0] - 1) if i + 1 < len(rows) else (clen - 1)
        hits.append((off, end))
    if not hits:
        raise ValueError("No matching GRIB records in idx for requested variables")
    return min(s for s, _ in hits), max(e for _, e in hits)


def _download_stitched_grib(grib_url: str, idx_url: str, needles: Sequence[str], dest: Path) -> None:
    clen = _http_head_clen(grib_url)
    idx_text = _http_get(idx_url).decode("utf-8", errors="replace")
    start, end = _parse_idx_for_variables(idx_text, needles, clen)
    hdr = {"User-Agent": UA, "Range": f"bytes={start}-{end}"}
    dest.write_bytes(_http_get(grib_url, headers=hdr))


def _kelvin_to_c(tk: float) -> float:
    return float(tk) - 273.15


def _rename_latlon(ds: Any) -> Any:
    rename: Dict[str, str] = {}
    for cand_lat in ("latitude", "lat"):
        if cand_lat in ds.coords:
            rename[cand_lat] = "lat"
            break
    for cand_lon in ("longitude", "lon"):
        if cand_lon in ds.coords:
            rename[cand_lon] = "lon"
            break
    return ds.rename(rename) if rename else ds


def _nearest_scalar(ds: Any, lat: float, lon: float) -> float:
    """Nearest grid value; supports 1D lat/lon or 2D curvilinear (e.g. HRRR Lambert)."""
    _require_numpy()
    ds = _rename_latlon(ds)
    lonv = float(lon)
    vname = list(ds.data_vars)[0]
    arr = np.asarray(ds[vname].values)
    lats = np.asarray(ds["lat"].values)
    lons = np.asarray(ds["lon"].values)
    if lonv < 0 and float(lons.max()) > 180:
        lonv += 360.0
    if lats.ndim == 2 and lons.ndim == 2:
        d = (lats - lat) ** 2 + (lons - lonv) ** 2
        idx = np.unravel_index(int(np.argmin(d)), d.shape)
        return float(arr[idx])
    if lats.ndim == 1 and lons.ndim == 1:
        i = int(np.argmin(np.abs(lats - lat)))
        j = int(np.argmin(np.abs(lons - lonv)))
        if arr.ndim == 2:
            if arr.shape[0] == lats.shape[0] and arr.shape[1] == lons.shape[0]:
                return float(arr[i, j])
            return float(arr[j, i])
    return float(np.asarray(ds.sel(lat=lat, lon=lonv, method="nearest")[vname].values).flat[0])


def _hrrr_s3_urls(run_date: str, cycle_hh: str, fxx: int) -> Tuple[str, str]:
    base = f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{run_date}/conus"
    fname = f"hrrr.t{cycle_hh}z.wrfsfcf{fxx:02d}.grib2"
    return f"{base}/{fname}", f"{base}/{fname}.idx"


def _open_field(path: Path, filter_by_keys: Dict[str, Any]) -> Any:
    import xarray as xr  # type: ignore

    return xr.open_dataset(
        path,
        engine="cfgrib",
        filter_by_keys=filter_by_keys,
        errors="ignore",
        backend_kwargs={"indexpath": ""},
    )


def _extract_surface_point(path: Path, lat: float, lon: float) -> Tuple[float, Optional[float], float, float, float]:
    """Read surface fields via explicit cfgrib filters (robust on AWS byte-range slices)."""
    t_k: Optional[float] = None
    td_k: Optional[float] = None
    u10, v10, sp = 0.0, 0.0, 101325.0

    filters = (
        ("t2", {"shortName": "2t", "typeOfLevel": "heightAboveGround", "level": 2}),
        ("d2", {"shortName": "2d", "typeOfLevel": "heightAboveGround", "level": 2}),
        ("u10", {"shortName": "10u", "typeOfLevel": "heightAboveGround", "level": 10}),
        ("v10", {"shortName": "10v", "typeOfLevel": "heightAboveGround", "level": 10}),
        ("sp", {"shortName": "sp", "typeOfLevel": "surface"}),
    )
    for kind, keys in filters:
        try:
            ds = _open_field(path, keys)
        except Exception:
            continue
        try:
            val = _nearest_scalar(ds, lat, lon)
            if kind == "t2":
                t_k = val
            elif kind == "d2":
                td_k = val
            elif kind == "u10":
                u10 = val
            elif kind == "v10":
                v10 = val
            elif kind == "sp":
                sp = val
        finally:
            ds.close()

    if t_k is None:
        raise ValueError("Subset GRIB did not contain 2 m temperature (2t); check idx needles / download range")
    t_c = _kelvin_to_c(t_k)
    td_c = _kelvin_to_c(td_k) if td_k is not None else None
    return t_c, td_c, u10, v10, sp


def fetch_hrrr_point_series(
    latitude: float,
    longitude: float,
    run_time_utc: Union[str, datetime],
    fxx_hours: Sequence[int],
) -> List[ForecastState]:
    if isinstance(run_time_utc, datetime):
        dt = run_time_utc.astimezone(timezone.utc)
    else:
        dt = datetime.strptime(run_time_utc.strip(), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    date_s = dt.strftime("%Y%m%d")
    cycle = dt.strftime("%H")

    needles = (
        ":TMP:2 m above ground:",
        ":DPT:2 m above ground:",
        ":UGRD:10 m above ground:",
        ":VGRD:10 m above ground:",
        ":PRES:surface:",
    )

    tmpdir = Path.cwd() / ".roof_hunter_grib_cache"
    tmpdir.mkdir(exist_ok=True)

    states: List[ForecastState] = []
    for fxx in fxx_hours:
        grib_url, idx_url = _hrrr_s3_urls(date_s, cycle, int(fxx))
        dest = tmpdir / f"hrrr_{date_s}_{cycle}_{int(fxx):02d}_subset.grib2"
        try:
            if not dest.exists() or dest.stat().st_size < 100:
                _download_stitched_grib(grib_url, idx_url, needles, dest)
        except (urllib.error.HTTPError, urllib.error.URLError, ValueError, RuntimeError) as err:
            raise RuntimeError(f"HRRR subset failed for {grib_url}: {err}") from err

        t_c, td_c, u, v, sp = _extract_surface_point(dest, latitude, longitude)
        wind_ms = float(np.hypot(u, v))
        wind_deg = float((math.degrees(math.atan2(u, v)) + 360.0) % 360.0)
        p_hpa = sp / 100.0 if sp > 2000 else float(sp)

        rh: Optional[float] = None
        if td_c is not None:
            es = 6.112 * math.exp(17.67 * t_c / (t_c + 243.5))
            e = 6.112 * math.exp(17.67 * td_c / (td_c + 243.5))
            rh = float(max(0.0, min(1.0, e / es)))

        ts = dt + timedelta(hours=int(fxx))
        states.append(
            ForecastState(
                timestamp=ts,
                latitude=latitude,
                longitude=longitude,
                surface_temp_c=t_c,
                relative_humidity=float(rh) if rh is not None else 0.65,
                surface_pressure_hpa=p_hpa,
                surface_dewpoint_c=td_c,
                wind_speed_m_s=wind_ms,
                wind_direction_deg=wind_deg,
            )
        )
    return states


def states_to_forecast_payload(states: Sequence[ForecastState]) -> Dict[str, Any]:
    return {"forecast": [s.to_dict() for s in states]}


def write_forecast_json(states: Sequence[ForecastState], path: Path) -> None:
    path.write_text(json.dumps(states_to_forecast_payload(states), indent=2), encoding="utf-8")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Build roof_hunter_forecast.json from HRRR (QuLab subset ingest).")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--run", type=str, required=True, help="UTC run, e.g. '2025-05-01 12:00'")
    p.add_argument("--fxx", type=str, default="0,1,2,3,6,12,18,24", help="Comma-separated forecast hours")
    p.add_argument("--output", type=Path, default=Path("roof_hunter_forecast.json"))
    args = p.parse_args()
    hours = [int(x.strip()) for x in args.fxx.split(",") if x.strip()]
    states = fetch_hrrr_point_series(args.lat, args.lon, args.run, hours)
    write_forecast_json(states, args.output)
    print(f"Wrote {len(states)} steps to {args.output}")


if __name__ == "__main__":
    main()
