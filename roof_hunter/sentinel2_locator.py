"""Locate post-event Sentinel-2 scenes for lead rows via public STAC API.

This script does not download full imagery or run CV. It appends scene metadata
so downstream workers can fetch chips and verify roof damage.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EARTH_SEARCH_STAC = "https://earth-search.aws.element84.com/v1/search"
USER_AGENT = "QuLabInfinite-RoofHunter/1.0 sentinel2-locator"

S2_COLUMNS = [
    "sentinel2_status",
    "sentinel2_scene_id",
    "sentinel2_acquired_utc",
    "sentinel2_days_after_event",
    "sentinel2_cloud_cover_pct",
    "sentinel2_preview_url",
    "sentinel2_data_url",
    "sentinel2_stac_item_url",
]


def _parse_dt(value: str) -> Optional[datetime]:
    v = (value or "").strip()
    if not v:
        return None
    try:
        # Accept both Z and explicit offsets.
        return datetime.fromisoformat(v.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _post_json(url: str, payload: Dict[str, Any], timeout_sec: float = 30.0) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _select_asset(item: Dict[str, Any], names: Tuple[str, ...]) -> str:
    assets = item.get("assets") or {}
    for name in names:
        href = ((assets.get(name) or {}).get("href") or "").strip()
        if href:
            return href
    return ""


def _choose_scene(
    lat: float,
    lon: float,
    event_dt: datetime,
    max_days_after_event: int,
    cloud_cover_max: float,
    limit: int,
) -> Dict[str, Any]:
    end_dt = event_dt + timedelta(days=max_days_after_event)
    payload = {
        "collections": ["sentinel-2-l2a"],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "datetime": f"{event_dt.isoformat()}/{end_dt.isoformat()}",
        "limit": limit,
        "query": {"eo:cloud_cover": {"lte": float(cloud_cover_max)}},
        "sortby": [{"field": "properties.datetime", "direction": "asc"}],
    }
    try:
        resp = _post_json(EARTH_SEARCH_STAC, payload)
    except urllib.error.HTTPError as e:
        return {"status": f"api_http_{e.code}"}
    except urllib.error.URLError:
        return {"status": "api_network_error"}
    except Exception:
        return {"status": "api_error"}

    features = resp.get("features") or []
    if not features:
        return {"status": "no_scene_found"}

    # Prefer earliest post-event scene with lowest cloud in close tie.
    scored: List[Tuple[float, float, Dict[str, Any]]] = []
    for f in features:
        p = f.get("properties") or {}
        acq = _parse_dt(str(p.get("datetime") or ""))
        if acq is None:
            continue
        days_after = max(0.0, (acq - event_dt).total_seconds() / 86400.0)
        cloud = float(p.get("eo:cloud_cover") or 100.0)
        scored.append((days_after, cloud, f))
    if not scored:
        return {"status": "scene_missing_datetime"}

    scored.sort(key=lambda t: (t[0], t[1]))
    _, _, item = scored[0]
    p = item.get("properties") or {}
    acq_dt = _parse_dt(str(p.get("datetime") or "")) or event_dt
    days_after = max(0.0, (acq_dt - event_dt).total_seconds() / 86400.0)
    cloud = float(p.get("eo:cloud_cover") or 0.0)

    preview = _select_asset(item, ("thumbnail", "overview", "visual"))
    data_url = _select_asset(item, ("visual", "B04", "B08", "B02"))
    links = item.get("links") or []
    item_url = ""
    for link in links:
        if (link.get("rel") or "") == "self":
            item_url = (link.get("href") or "").strip()
            break

    return {
        "status": "ok",
        "scene_id": str(item.get("id") or ""),
        "acquired_utc": acq_dt.isoformat(),
        "days_after_event": round(days_after, 3),
        "cloud_cover_pct": round(cloud, 2),
        "preview_url": preview,
        "data_url": data_url,
        "stac_item_url": item_url,
    }


def append_sentinel2(
    in_csv: Path,
    out_csv: Path,
    *,
    max_days_after_event: int = 14,
    cloud_cover_max: float = 45.0,
    limit: int = 20,
    max_rows: Optional[int] = None,
) -> int:
    with in_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])

    if max_rows is not None:
        rows = rows[: max_rows]

    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        lat_raw = row.get("lat", "")
        lon_raw = row.get("lon", "")
        dt_raw = row.get("report_datetime", "")
        try:
            lat = float(lat_raw)
            lon = float(lon_raw)
        except (TypeError, ValueError):
            for col in S2_COLUMNS:
                out[col] = ""
            out["sentinel2_status"] = "bad_lat_lon"
            out_rows.append(out)
            continue
        dt = _parse_dt(str(dt_raw))
        if dt is None:
            for col in S2_COLUMNS:
                out[col] = ""
            out["sentinel2_status"] = "bad_report_datetime"
            out_rows.append(out)
            continue

        scene = _choose_scene(
            lat=lat,
            lon=lon,
            event_dt=dt,
            max_days_after_event=max_days_after_event,
            cloud_cover_max=cloud_cover_max,
            limit=limit,
        )
        out["sentinel2_status"] = scene.get("status", "")
        out["sentinel2_scene_id"] = scene.get("scene_id", "")
        out["sentinel2_acquired_utc"] = scene.get("acquired_utc", "")
        out["sentinel2_days_after_event"] = scene.get("days_after_event", "")
        out["sentinel2_cloud_cover_pct"] = scene.get("cloud_cover_pct", "")
        out["sentinel2_preview_url"] = scene.get("preview_url", "")
        out["sentinel2_data_url"] = scene.get("data_url", "")
        out["sentinel2_stac_item_url"] = scene.get("stac_item_url", "")
        out_rows.append(out)

    out_fields = list(fields)
    for c in S2_COLUMNS:
        if c not in out_fields:
            out_fields.append(c)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in out_fields})
    return len(out_rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Append Sentinel-2 post-event scene metadata to lead CSV")
    p.add_argument("--in", dest="in_csv", type=Path, required=True)
    p.add_argument("--out", dest="out_csv", type=Path, default=None)
    p.add_argument("--days-after", type=int, default=14, help="Search window after event datetime")
    p.add_argument("--cloud-max", type=float, default=45.0, help="Max eo:cloud_cover")
    p.add_argument("--limit", type=int, default=20, help="STAC scene limit")
    p.add_argument("--max-rows", type=int, default=None, help="Optional row cap for quick runs")
    args = p.parse_args()

    out = args.out_csv or args.in_csv.with_name(args.in_csv.stem + "_sentinel2.csv")
    n = append_sentinel2(
        args.in_csv,
        out,
        max_days_after_event=args.days_after,
        cloud_cover_max=args.cloud_max,
        limit=args.limit,
        max_rows=args.max_rows,
    )
    print(f"Wrote {n} rows with Sentinel-2 metadata to {out}")


if __name__ == "__main__":
    main()
