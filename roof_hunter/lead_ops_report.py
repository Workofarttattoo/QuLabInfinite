"""Lead-ops report generator: recent severe reports + next-48h projected hail risk CSVs.

Outputs:
- roof_hunter/output/actual_events_5_to_14_days.csv (residential → unclassified → commercial)
- roof_hunter/output/actual_events_5_to_14_days_residential.csv / _commercial.csv / _unclassified.csv
- roof_hunter/output/projected_hail_next_48h.csv

Optional (``--enrich-assessor``): also writes ``*_assessor.csv`` with county CAD/assessor URLs
plus OpenStreetMap reverse-geocode hints (~1 req/s to Nominatim). Or run
``python roof_hunter/enrich_assessor_leads.py`` standalone.
"""

from __future__ import annotations

import csv
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roof_hunter.integrations.hrrr_column_ingest import fetch_hrrr_column_features
from roof_hunter.simulator import HailPhysicsEngine
from roof_hunter.validate_last_week import fetch_spc_reports

REGIONS: Sequence[Tuple[str, float, float]] = (
    ("Fort Worth, TX", 32.7555, -97.3308),
    ("Oklahoma City, OK", 35.4676, -97.5164),
)

# Heuristic only — SPC text often lacks structure type; parcel/CAD enrichment overrides later.
_RES_HINT_COMMERCIAL = (
    "warehouse",
    "industrial",
    "strip mall",
    "plaza",
    "shopping center",
    "gas station",
    "convenience store",
    "storage unit",
    "self storage",
    "dollar general",
    "walmart",
    "wal-mart",
    "lowe's",
    "lowes ",
    "farm supply",
    "dealership",
    "motel",
    "hotel",
    "restaurant",
    "brewery",
    "distillery",
    "church",  # insured as institution; not single-family
    "school",
    "university",
    "hospital",
    "clinic ",
    "fedex",
    "ups facility",
    "post office",
)
_RES_HINT_RESIDENTIAL = (
    "residence",
    "residential",
    "home ",
    " homeowner",
    "house ",
    "homes ",
    "mobile home",
    "trailer park",
    "rv park",
    "apartment",
    "condo",
    "townhome",
    "townhouse",
    "duplex",
    "subdivision",
    "neighbor's house",
    "neighbors house",
)


def _infer_property_segment_hint(comments: str, spc_location: str) -> str:
    blob = f"{comments} {spc_location}".lower()
    if any(k in blob for k in _RES_HINT_COMMERCIAL):
        return "commercial"
    if any(k in blob for k in _RES_HINT_RESIDENTIAL):
        return "residential"
    return "unclassified"


def _segment_sort_key(segment: str) -> int:
    return {"residential": 0, "unclassified": 1, "commercial": 2}.get(segment, 1)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _severity_score(report: Dict[str, Any]) -> float:
    """Simple severe score from report type/comments; higher = operationally urgent."""
    rtype = str(report.get("type", "")).lower()
    comments = str(report.get("comments", "")).lower()
    score = 0.2
    if rtype == "hail":
        score += 0.55
    elif rtype == "tornado":
        score += 0.75
    if "baseball" in comments or "softball" in comments:
        score += 0.25
    if "2.00" in comments or "2 inch" in comments or "2.0" in comments:
        score += 0.18
    if "roof" in comments or "damage" in comments or "destroyed" in comments:
        score += 0.12
    return min(1.0, round(score, 3))


def build_actual_events_report() -> List[Dict[str, Any]]:
    today = date.today()
    end = today - timedelta(days=5)
    start = today - timedelta(days=14)

    rows: List[Dict[str, Any]] = []
    for region, lat, lon in REGIONS:
        reports = fetch_spc_reports(lat, lon, start, end, radius_km=120.0)
        for rep in reports:
            sev = _severity_score(rep)
            if sev < 0.65:
                continue
            loc = str(rep.get("location") or "")
            com = str(rep.get("comments") or "")
            segment = _infer_property_segment_hint(com, loc)
            rows.append(
                {
                    "region": region,
                    "property_segment_hint": segment,
                    "report_datetime": rep.get("report_datetime"),
                    "report_type": rep.get("type"),
                    "severity_score_0_1": sev,
                    "spc_location": loc,
                    "county": rep.get("county"),
                    "state": rep.get("state"),
                    "lat": rep.get("lat"),
                    "lon": rep.get("lon"),
                    "distance_km_from_region_center": rep.get("distance_km"),
                    "comments": com,
                }
            )
    rows.sort(
        key=lambda r: (
            _segment_sort_key(str(r["property_segment_hint"])),
            r["region"],
            -float(r["severity_score_0_1"]),
        )
    )
    return rows


def build_projected_report() -> List[Dict[str, Any]]:
    run_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    fxx = list(range(0, 49, 3))
    engine = HailPhysicsEngine(iterations=2500, seed=42)
    rows: List[Dict[str, Any]] = []
    for region, lat, lon in REGIONS:
        features = fetch_hrrr_column_features(lat, lon, run_utc, fxx)
        for feat in features:
            sim = engine.run_from_hrrr_column_features(feat)
            rows.append(
                {
                    "region": region,
                    "timestamp_utc": feat["timestamp"],
                    "lat": lat,
                    "lon": lon,
                    "cape_proxy_j_kg": feat["cape_proxy_j_kg"],
                    "updraft_speed_ms": feat["updraft_speed_ms"],
                    "lapse_700_500_k_per_km": feat["lapse_700_500_k_per_km"],
                    "bulk_shear_1000_500_ms": feat["bulk_shear_1000_500_ms"],
                    "projected_damage_prob_gt_1in": round(sim["damage_probability_gt_1in"], 4),
                    "projected_damage_prob_gt_2in": round(sim["damage_probability_gt_2in"], 4),
                    "projected_p95_hail_mm": round(sim["p95_diameter_mm"], 2),
                    "projected_max_hail_mm": round(sim["max_diameter_mm"], 2),
                }
            )
    rows.sort(key=lambda r: (r["region"], -float(r["projected_damage_prob_gt_1in"]), r["timestamp_utc"]))
    return rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enrich-assessor",
        action="store_true",
        help="After CSV write, emit actual_events*_assessor.csv with portal URLs + OSM reverse geocode hints (~1 Hz Nominatim).",
    )
    parser.add_argument(
        "--enrich-assessor-skip-geocode",
        action="store_true",
        help="Portal columns only when used with --enrich-assessor (no network).",
    )
    args = parser.parse_args()

    out_dir = Path("roof_hunter/output")
    actual_rows = build_actual_events_report()
    projected_rows = build_projected_report()

    actual_path = out_dir / "actual_events_5_to_14_days.csv"
    projected_path = out_dir / "projected_hail_next_48h.csv"

    actual_fieldnames = (
        "region",
        "property_segment_hint",
        "report_datetime",
        "report_type",
        "severity_score_0_1",
        "spc_location",
        "county",
        "state",
        "lat",
        "lon",
        "distance_km_from_region_center",
        "comments",
    )

    _write_csv(actual_path, actual_rows, actual_fieldnames)

    by_seg = {"residential": [], "commercial": [], "unclassified": []}
    for r in actual_rows:
        seg = str(r.get("property_segment_hint", "unclassified"))
        if seg in by_seg:
            by_seg[seg].append(r)
        else:
            by_seg["unclassified"].append(r)

    _write_csv(
        actual_path.with_name(f"{actual_path.stem}_residential.csv"),
        by_seg["residential"],
        actual_fieldnames,
    )
    _write_csv(
        actual_path.with_name(f"{actual_path.stem}_commercial.csv"),
        by_seg["commercial"],
        actual_fieldnames,
    )
    _write_csv(
        actual_path.with_name(f"{actual_path.stem}_unclassified.csv"),
        by_seg["unclassified"],
        actual_fieldnames,
    )
    _write_csv(
        projected_path,
        projected_rows,
        (
            "region",
            "timestamp_utc",
            "lat",
            "lon",
            "cape_proxy_j_kg",
            "updraft_speed_ms",
            "lapse_700_500_k_per_km",
            "bulk_shear_1000_500_ms",
            "projected_damage_prob_gt_1in",
            "projected_damage_prob_gt_2in",
            "projected_p95_hail_mm",
            "projected_max_hail_mm",
        ),
    )

    print(
        f"Wrote {len(actual_rows)} rows to {actual_path} "
        f"(segment split: res={len(by_seg['residential'])} comm={len(by_seg['commercial'])} uncl={len(by_seg['unclassified'])})"
    )
    print(f"Wrote {len(projected_rows)} rows to {projected_path}")

    if args.enrich_assessor:
        from roof_hunter.enrich_assessor_leads import enrich_csv as _assessor_enrich_csv

        assessor_path = actual_path.with_name(f"{actual_path.stem}_assessor.csv")
        n = _assessor_enrich_csv(
            actual_path,
            assessor_path,
            skip_geocode=args.enrich_assessor_skip_geocode,
        )
        print(f"Wrote {n} assessor-enriched rows to {assessor_path}")


if __name__ == "__main__":
    main()

