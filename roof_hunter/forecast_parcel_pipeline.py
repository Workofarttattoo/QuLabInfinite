"""Forecast-to-parcel spatial pipeline.

Expands the HRRR hail forecast from two hub points to a proper spatial grid
(or a supplied set of parcel centroids), scores each cell with HailPhysicsEngine,
and enriches surviving hits with reverse geocoding + INCOG parcel data.

Outputs a lead DataFrame / CSV ready for unified_lead_sender.py.

Usage:
    python -m roof_hunter.forecast_parcel_pipeline \
        --state OK \
        --grid-spacing-deg 0.05 \
        --min-prob 0.08 \
        --out roof_hunter/output/forecast_parcel_hits.csv
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roof_hunter.integrations.hrrr_column_ingest import fetch_hrrr_column_features
from roof_hunter.simulator import HailPhysicsEngine
from roof_hunter.integrations.ok_incog_parcels import (
    COUNTY_TO_SERVICE,
    flatten_for_batchdata,
    normalize_county,
    query_parcel_point,
)

# ── region definitions ────────────────────────────────────────────────────────
# Each region: (label, center_lat, center_lon, lat_span_deg, lon_span_deg)
FORECAST_REGIONS: List[Tuple[str, float, float, float, float]] = [
    ("Oklahoma City Metro", 35.47, -97.52, 1.8, 2.8),
    ("Tulsa Metro",         36.15, -95.99, 1.4, 2.2),
    ("Fort Worth / DFW",    32.76, -97.33, 1.6, 2.4),
]

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
_NOMINATIM_UA = "QuLabInfinite-RoofHunter/1.2-forecast-parcel"
_GEOCODE_CACHE_PATH = Path(__file__).parent / ".cache" / "nominatim_grid_cache.json"

_HRRR_FXX = list(range(0, 49, 3))   # 0-48h, every 3h


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class GridCell:
    lat: float
    lon: float
    region: str
    cape_proxy: float = 0.0
    updraft_ms: float = 0.0
    lapse_700_500: float = 0.0
    shear_1000_500: float = 0.0
    prob_gt_1in: float = 0.0
    prob_gt_2in: float = 0.0
    p95_hail_mm: float = 0.0
    max_hail_mm: float = 0.0
    peak_timestamp_utc: str = ""
    # enrichment
    nominatim_address: str = ""
    nominatim_county: str = ""
    nominatim_state: str = ""
    nominatim_zip: str = ""
    parcel_status: str = ""
    parcel_detail: str = ""
    parcel_attributes: Dict[str, Any] = field(default_factory=dict)


# ── grid generation ───────────────────────────────────────────────────────────

def generate_grid(
    region: Tuple[str, float, float, float, float],
    spacing_deg: float,
) -> List[Tuple[float, float, str]]:
    """Return list of (lat, lon, region_label) points on a uniform grid."""
    label, clat, clon, lat_span, lon_span = region
    lat_min = clat - lat_span / 2
    lat_max = clat + lat_span / 2
    lon_min = clon - lon_span / 2
    lon_max = clon + lon_span / 2

    points: List[Tuple[float, float, str]] = []
    lat = lat_min
    while lat <= lat_max + 1e-9:
        lon = lon_min
        while lon <= lon_max + 1e-9:
            points.append((round(lat, 5), round(lon, 5), label))
            lon += spacing_deg
        lat += spacing_deg
    return points


def grid_points_all_regions(
    spacing_deg: float = 0.05,
    regions: Optional[Sequence[Tuple[str, float, float, float, float]]] = None,
) -> List[Tuple[float, float, str]]:
    pts: List[Tuple[float, float, str]] = []
    for region in (regions or FORECAST_REGIONS):
        pts.extend(generate_grid(region, spacing_deg))
    return pts


# ── HRRR scoring ──────────────────────────────────────────────────────────────

def score_grid_cells(
    points: List[Tuple[float, float, str]],
    run_utc: datetime,
    engine: HailPhysicsEngine,
    fxx: Sequence[int] = _HRRR_FXX,
) -> List[GridCell]:
    """Fetch HRRR columns for each grid point and score with HailPhysicsEngine.

    Returns one GridCell per point with the peak probability across forecast hours.
    """
    cells: List[GridCell] = []
    for lat, lon, region in points:
        try:
            features = fetch_hrrr_column_features(lat, lon, run_utc, list(fxx))
        except Exception:
            # HRRR fetch failures are acceptable for individual grid cells
            cells.append(GridCell(lat=lat, lon=lon, region=region))
            continue

        best: Optional[Dict[str, Any]] = None
        best_p = -1.0
        for feat in features:
            sim = engine.run_from_hrrr_column_features(feat)
            p = float(sim.get("damage_probability_gt_1in", 0.0))
            if p > best_p:
                best_p = p
                best = {"feat": feat, "sim": sim}

        if best is None:
            cells.append(GridCell(lat=lat, lon=lon, region=region))
            continue

        feat = best["feat"]
        sim = best["sim"]
        cells.append(
            GridCell(
                lat=lat,
                lon=lon,
                region=region,
                cape_proxy=float(feat.get("cape_proxy_j_kg", 0)),
                updraft_ms=float(feat.get("updraft_speed_ms", 0)),
                lapse_700_500=float(feat.get("lapse_700_500_k_per_km", 0)),
                shear_1000_500=float(feat.get("bulk_shear_1000_500_ms", 0)),
                prob_gt_1in=round(float(sim.get("damage_probability_gt_1in", 0)), 4),
                prob_gt_2in=round(float(sim.get("damage_probability_gt_2in", 0)), 4),
                p95_hail_mm=round(float(sim.get("p95_diameter_mm", 0)), 2),
                max_hail_mm=round(float(sim.get("max_diameter_mm", 0)), 2),
                peak_timestamp_utc=str(feat.get("timestamp", "")),
            )
        )
    return cells


def filter_hits(cells: List[GridCell], min_prob: float = 0.08) -> List[GridCell]:
    return [c for c in cells if c.prob_gt_1in >= min_prob]


# ── reverse geocode ───────────────────────────────────────────────────────────

def _load_geocode_cache() -> Dict[str, Dict[str, str]]:
    if not _GEOCODE_CACHE_PATH.is_file():
        return {}
    try:
        return json.loads(_GEOCODE_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_geocode_cache(cache: Dict[str, Dict[str, str]]) -> None:
    _GEOCODE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _GEOCODE_CACHE_PATH.write_text(json.dumps(cache, indent=0), encoding="utf-8")


def _nominatim_reverse(lat: float, lon: float, sleep_s: float) -> Dict[str, str]:
    time.sleep(max(0.0, sleep_s))
    params = f"lat={lat}&lon={lon}&format=json&addressdetails=1&zoom=18"
    url = f"{_NOMINATIM_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": _NOMINATIM_UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {"address": "", "county": "", "state": "", "zip": ""}

    addr_parts = data.get("address") or {}
    house_number = addr_parts.get("house_number", "")
    road = addr_parts.get("road", "")
    display_address = f"{house_number} {road}".strip() if house_number or road else data.get("display_name", "")
    county_raw = (
        addr_parts.get("county")
        or addr_parts.get("municipality")
        or ""
    )
    county = county_raw.replace(" County", "").strip()
    state = addr_parts.get("state_code") or addr_parts.get("state") or ""
    if len(state) > 2:
        # Normalize to abbreviation for OK, TX, etc.
        _STATE_MAP = {
            "Oklahoma": "OK", "Texas": "TX", "Kansas": "KS",
            "Missouri": "MO", "Arkansas": "AR", "Louisiana": "LA",
            "Colorado": "CO", "New Mexico": "NM",
        }
        state = _STATE_MAP.get(state, state)
    postcode = str(addr_parts.get("postcode") or "")[:5]
    return {
        "address": display_address,
        "county": county,
        "state": state,
        "zip": postcode,
    }


def reverse_geocode_cells(
    cells: List[GridCell],
    sleep_s: float = 1.05,
) -> None:
    """Mutate cells in-place with nominatim address fields (cached)."""
    cache = _load_geocode_cache()
    dirty = False
    for cell in cells:
        key = f"{round(cell.lat, 4)},{round(cell.lon, 4)}"
        if key in cache:
            geo = cache[key]
        else:
            geo = _nominatim_reverse(cell.lat, cell.lon, sleep_s)
            cache[key] = geo
            dirty = True
        cell.nominatim_address = geo.get("address", "")
        cell.nominatim_county = geo.get("county", "")
        cell.nominatim_state = geo.get("state", "")
        cell.nominatim_zip = geo.get("zip", "")
    if dirty:
        _save_geocode_cache(cache)


# ── parcel enrichment ─────────────────────────────────────────────────────────

def enrich_cells_with_parcels(
    cells: List[GridCell],
    sleep_s: float = 0.12,
    state_filter: str = "OK",
) -> None:
    """Mutate cells in-place with INCOG parcel attributes where available."""
    for cell in cells:
        if state_filter and cell.nominatim_state.upper() != state_filter.upper():
            cell.parcel_status = "state_not_covered"
            continue
        cnorm = normalize_county(cell.nominatim_county)
        if cnorm not in COUNTY_TO_SERVICE:
            cell.parcel_status = "no_parcel_layer"
            continue
        res = query_parcel_point(cell.lat, cell.lon, cell.nominatim_county, sleep_s=sleep_s)
        if res.matched:
            cell.parcel_status = "ok"
            cell.parcel_attributes = res.parcel_attributes
        else:
            cell.parcel_status = f"lookup_failed:{res.arcgis_error}"


# ── lead row conversion ───────────────────────────────────────────────────────

def cells_to_lead_rows(cells: List[GridCell]) -> List[Dict[str, Any]]:
    """Convert scored+enriched GridCells to lead dicts compatible with the rest of the pipeline."""
    rows = []
    for cell in cells:
        bd = flatten_for_batchdata(cell.parcel_attributes)
        lead_id = _cell_id(cell)
        # Derive a rough severity / rank score from hail physics outputs
        lead_rank_score = round(
            cell.prob_gt_1in * 1.0
            + cell.prob_gt_2in * 0.5
            + min(1.0, cell.max_hail_mm / 100.0) * 0.4,
            4,
        )
        rows.append(
            {
                "lead_id": lead_id,
                "data_source": "hrrr_forecast_grid",
                "region": cell.region,
                "lat": cell.lat,
                "lon": cell.lon,
                "peak_timestamp_utc": cell.peak_timestamp_utc,
                "report_datetime": cell.peak_timestamp_utc,   # alias for downstream compat
                "lead_rank_score": lead_rank_score,
                "severity_score_0_1": round(cell.prob_gt_1in, 4),
                "property_segment": "unclassified",   # refined by parcel later
                "county": cell.nominatim_county,
                "state": cell.nominatim_state,
                "inferred_zip": cell.nominatim_zip,
                "property_address_candidate": cell.nominatim_address,
                "cape_proxy_j_kg": cell.cape_proxy,
                "updraft_speed_ms": cell.updraft_ms,
                "lapse_700_500_k_per_km": cell.lapse_700_500,
                "bulk_shear_1000_500_ms": cell.shear_1000_500,
                "projected_damage_prob_gt_1in": cell.prob_gt_1in,
                "projected_damage_prob_gt_2in": cell.prob_gt_2in,
                "projected_p95_hail_mm": cell.p95_hail_mm,
                "projected_max_hail_mm": cell.max_hail_mm,
                "parcel_enrichment_status": cell.parcel_status,
                **bd,
            }
        )
    return rows


def _cell_id(cell: GridCell) -> str:
    blob = f"{cell.lat}:{cell.lon}:{cell.peak_timestamp_utc}"
    return hashlib.md5(blob.encode()).hexdigest()[:16]


# ── CSV I/O ───────────────────────────────────────────────────────────────────

_FIELDNAMES = [
    "lead_id", "data_source", "region", "lat", "lon",
    "peak_timestamp_utc", "report_datetime",
    "lead_rank_score", "severity_score_0_1",
    "property_segment", "county", "state", "inferred_zip",
    "property_address_candidate",
    "cape_proxy_j_kg", "updraft_speed_ms", "lapse_700_500_k_per_km", "bulk_shear_1000_500_ms",
    "projected_damage_prob_gt_1in", "projected_damage_prob_gt_2in",
    "projected_p95_hail_mm", "projected_max_hail_mm",
    "parcel_enrichment_status",
    "batch_apn", "batch_owner_name", "batch_property_address",
    "batch_property_city", "batch_property_zip",
    "batch_mailing_line1", "batch_mailing_city", "batch_mailing_state", "batch_mailing_zip",
    "parcel_year_built", "parcel_total_acct_value", "parcel_legal",
]


def write_lead_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_keys = list(dict.fromkeys(list(_FIELDNAMES) + [k for r in rows for k in r]))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in all_keys})


# ── main entry point ──────────────────────────────────────────────────────────

def run(
    *,
    out_path: Path,
    spacing_deg: float = 0.05,
    min_prob: float = 0.08,
    geocode_sleep: float = 1.05,
    parcel_sleep: float = 0.12,
    state_filter: str = "OK",
    run_utc: Optional[datetime] = None,
    regions: Optional[Sequence[Tuple[str, float, float, float, float]]] = None,
) -> List[Dict[str, Any]]:
    run_utc = run_utc or datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    engine = HailPhysicsEngine(iterations=2500, seed=42)

    print(f"Generating grid (spacing={spacing_deg}°)…", flush=True)
    points = grid_points_all_regions(spacing_deg, regions)
    print(f"  {len(points)} grid points across {len(regions or FORECAST_REGIONS)} region(s)", flush=True)

    print("Scoring grid with HRRR + HailPhysicsEngine…", flush=True)
    cells = score_grid_cells(points, run_utc, engine)

    hits = filter_hits(cells, min_prob)
    print(f"  {len(hits)}/{len(cells)} cells above prob_gt_1in ≥ {min_prob}", flush=True)

    if not hits:
        print("No cells exceed probability threshold — writing empty CSV.", flush=True)
        write_lead_csv([], out_path)
        return []

    print(f"Reverse geocoding {len(hits)} hit cells (Nominatim)…", flush=True)
    reverse_geocode_cells(hits, sleep_s=geocode_sleep)

    print("Enriching with INCOG parcel data…", flush=True)
    enrich_cells_with_parcels(hits, sleep_s=parcel_sleep, state_filter=state_filter)

    rows = cells_to_lead_rows(hits)
    rows.sort(key=lambda r: -float(r.get("lead_rank_score") or 0))
    write_lead_csv(rows, out_path)
    print(f"Wrote {len(rows)} forecast-parcel leads to {out_path}", flush=True)
    return rows


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("roof_hunter/output/forecast_parcel_hits.csv"))
    ap.add_argument("--spacing", dest="spacing_deg", type=float, default=0.05,
                    help="Grid point spacing in degrees (~5.5 km at 0.05)")
    ap.add_argument("--min-prob", type=float, default=0.08,
                    help="Minimum projected_damage_prob_gt_1in to keep a cell")
    ap.add_argument("--geocode-sleep", type=float, default=1.05,
                    help="Seconds between Nominatim calls (respect 1 req/s limit)")
    ap.add_argument("--parcel-sleep", type=float, default=0.12)
    ap.add_argument("--state", default="OK", help="State filter for parcel enrichment")
    args = ap.parse_args()

    run(
        out_path=args.out.expanduser().resolve(),
        spacing_deg=args.spacing_deg,
        min_prob=args.min_prob,
        geocode_sleep=args.geocode_sleep,
        parcel_sleep=args.parcel_sleep,
        state_filter=args.state,
    )


if __name__ == "__main__":
    main()
