"""Build per-lead satellite chip job specs for worker execution.

This scaffolding prepares property-level chip jobs from Sentinel-enriched lead CSVs.
It does not fetch parcel polygons itself; instead it exposes footprint placeholders
that downstream assessor/parcel enrichers can fill.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _bbox_from_point(lat: float, lon: float, half_size_m: float = 30.0) -> Dict[str, float]:
    """Approximate bbox around point in degrees (sufficient for chip API requests)."""
    lat_deg = half_size_m / 111_320.0
    lon_deg = half_size_m / (111_320.0 * max(0.1, math.cos(math.radians(lat))))
    return {
        "min_lat": round(lat - lat_deg, 7),
        "max_lat": round(lat + lat_deg, 7),
        "min_lon": round(lon - lon_deg, 7),
        "max_lon": round(lon + lon_deg, 7),
    }


def build_jobs(in_csv: Path, out_jsonl: Path, out_csv: Path, *, max_rows: int | None = None) -> int:
    with in_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if max_rows is not None:
        rows = rows[: max_rows]

    jobs: List[Dict[str, Any]] = []
    flat_rows: List[Dict[str, Any]] = []
    for r in rows:
        if (r.get("sentinel2_status") or "").strip().lower() != "ok":
            continue
        lead_id = (r.get("lead_id") or "").strip()
        lat = _to_float(r.get("lat"), default=float("nan"))
        lon = _to_float(r.get("lon"), default=float("nan"))
        if math.isnan(lat) or math.isnan(lon):
            continue

        bbox = _bbox_from_point(lat, lon, half_size_m=30.0)
        job = {
            "lead_id": lead_id,
            "event_datetime_utc": r.get("report_datetime", ""),
            "property_address": r.get("property_address", ""),
            "property_address_unit": r.get("property_address_unit", ""),
            "property_segment": r.get("property_segment", ""),
            "coords": {"lat": lat, "lon": lon},
            "chip_bbox_wgs84": bbox,
            "after_scene": {
                "scene_id": r.get("sentinel2_scene_id", ""),
                "acquired_utc": r.get("sentinel2_acquired_utc", ""),
                "cloud_cover_pct": _to_float(r.get("sentinel2_cloud_cover_pct"), 999.0),
                "stac_item_url": r.get("sentinel2_stac_item_url", ""),
                "preview_url": r.get("sentinel2_preview_url", ""),
                "data_url": r.get("sentinel2_data_url", ""),
            },
            "before_scene": {
                "status": "pending_lookup",
                "search_window_days_before_event": 45,
                "preferred_max_cloud_pct": 30.0,
            },
            "property_footprint": {
                "status": "pending_parcel_or_building_fetch",
                "source": "",
                "wkt_or_geojson": "",
            },
            "verification_requirements": {
                "source_type": "satellite",
                "single_property_damage_visible_required": True,
                "minimum_confidence": 0.74,
            },
        }
        jobs.append(job)
        flat_rows.append(
            {
                "lead_id": lead_id,
                "event_datetime_utc": r.get("report_datetime", ""),
                "property_address": r.get("property_address", ""),
                "property_address_unit": r.get("property_address_unit", ""),
                "property_segment": r.get("property_segment", ""),
                "lat": lat,
                "lon": lon,
                "chip_bbox_min_lon": bbox["min_lon"],
                "chip_bbox_min_lat": bbox["min_lat"],
                "chip_bbox_max_lon": bbox["max_lon"],
                "chip_bbox_max_lat": bbox["max_lat"],
                "after_scene_id": r.get("sentinel2_scene_id", ""),
                "after_scene_utc": r.get("sentinel2_acquired_utc", ""),
                "after_scene_cloud_cover_pct": r.get("sentinel2_cloud_cover_pct", ""),
                "after_scene_stac_item_url": r.get("sentinel2_stac_item_url", ""),
                "before_scene_status": "pending_lookup",
                "property_footprint_status": "pending_parcel_or_building_fetch",
                "worker_job_status": "queued",
            }
        )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(job) + "\n")

    fieldnames = list(flat_rows[0].keys()) if flat_rows else [
        "lead_id",
        "event_datetime_utc",
        "property_address",
        "property_address_unit",
        "property_segment",
        "lat",
        "lon",
        "chip_bbox_min_lon",
        "chip_bbox_min_lat",
        "chip_bbox_max_lon",
        "chip_bbox_max_lat",
        "after_scene_id",
        "after_scene_utc",
        "after_scene_cloud_cover_pct",
        "after_scene_stac_item_url",
        "before_scene_status",
        "property_footprint_status",
        "worker_job_status",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in flat_rows:
            w.writerow(row)

    return len(jobs)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate per-lead satellite chip jobs from Sentinel-enriched lead CSV")
    p.add_argument("--in", dest="in_csv", type=Path, required=True)
    p.add_argument("--out-jsonl", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--max-rows", type=int, default=None)
    args = p.parse_args()

    n = build_jobs(args.in_csv, args.out_jsonl, args.out_csv, max_rows=args.max_rows)
    print(f"Wrote {n} chip jobs to {args.out_jsonl} and {args.out_csv}")


if __name__ == "__main__":
    main()
