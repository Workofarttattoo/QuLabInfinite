"""Attach official county appraisal/assessor portal URLs plus OSM reverse-geocode hints to lead CSVs.

This does **not** scrape CAD sites (fragile / often prohibited). Produce search hints so humans or
later licensed parcel APIs can complete parcel match.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roof_hunter.integrations.assessor_portals import get_county_property_portal  # noqa: E402
from roof_hunter.integrations.osm_nominatim import reverse_geocode_cached  # noqa: E402


_CACHE_DEFAULT = Path(__file__).resolve().parent / ".cache" / "nominatim_reverse_cache.json"

_EXTRA_FIELDS: Sequence[str] = (
    "assessor_portal_matched",
    "assessor_organization",
    "assessor_home_url",
    "assessor_property_search_url",
    "assessor_notes",
    "assessor_workflow_hint",
    "reverse_geocode_display_name",
    "reverse_geocode_suggested_search_address",
    "reverse_geocode_road",
    "reverse_geocode_city",
    "reverse_geocode_state",
    "reverse_geocode_postcode",
    "reverse_geocode_osm_county",
    "reverse_geocode_near_poi",
    "reverse_geocode_precision",
    "reverse_geocode_confidence_note",
    "reverse_geocode_error",
)


def enrich_row(row: Dict[str, str], *, skip_geocode: bool, mem: Dict[str, Dict[str, Any]], cache_path: Path) -> Dict[str, Any]:
    p = get_county_property_portal(row["county"], row["state"])
    out: Dict[str, Any] = dict(row)

    if p:
        out["assessor_portal_matched"] = "yes"
        out["assessor_organization"] = p.organization
        out["assessor_home_url"] = p.home_url
        out["assessor_property_search_url"] = p.property_search_url
        out["assessor_notes"] = p.notes or ""
        out["assessor_workflow_hint"] = p.primary_search_hint()
    else:
        out["assessor_portal_matched"] = "no"
        out["assessor_organization"] = ""
        out["assessor_home_url"] = ""
        out["assessor_property_search_url"] = ""
        out["assessor_notes"] = ""
        out["assessor_workflow_hint"] = (
            "No portal row in roof_hunter.integrations.assessor_portals for this county; "
            "add it or locate the CAD/assessor manually via state comptroller / county clerk."
        )

    if skip_geocode:
        out.update(
            {
                "reverse_geocode_display_name": "",
                "reverse_geocode_suggested_search_address": "",
                "reverse_geocode_road": "",
                "reverse_geocode_city": "",
                "reverse_geocode_state": "",
                "reverse_geocode_postcode": "",
                "reverse_geocode_osm_county": "",
                "reverse_geocode_near_poi": "",
                "reverse_geocode_precision": "",
                "reverse_geocode_confidence_note": "",
                "reverse_geocode_error": "",
            }
        )
        return out

    try:
        lat = float(row["lat"])
        lon = float(row["lon"])
    except (KeyError, TypeError, ValueError):
        out.update(
            {
                "reverse_geocode_display_name": "",
                "reverse_geocode_suggested_search_address": "",
                "reverse_geocode_road": "",
                "reverse_geocode_city": "",
                "reverse_geocode_state": "",
                "reverse_geocode_postcode": "",
                "reverse_geocode_osm_county": "",
                "reverse_geocode_near_poi": "",
                "reverse_geocode_precision": "",
                "reverse_geocode_confidence_note": "",
                "reverse_geocode_error": "bad_lat_lon",
            }
        )
        return out

    geo = reverse_geocode_cached(lat, lon, memory=mem, cache_path=cache_path)
    err = geo.get("error")
    out["reverse_geocode_display_name"] = geo.get("display_name") or ""
    out["reverse_geocode_suggested_search_address"] = geo.get("suggested_search_address") or ""
    out["reverse_geocode_road"] = geo.get("road") or ""
    out["reverse_geocode_city"] = geo.get("city") or ""
    out["reverse_geocode_state"] = geo.get("state") or ""
    out["reverse_geocode_postcode"] = geo.get("postcode") or ""
    out["reverse_geocode_osm_county"] = geo.get("county_osm") or ""
    out["reverse_geocode_near_poi"] = geo.get("amenity_near") or ""
    precision = geo.get("reverse_geocode_precision") or ""
    out["reverse_geocode_precision"] = precision
    conf_full = geo.get("reverse_geocode_confidence_note") or ""
    if precision == "house":
        out["reverse_geocode_confidence_note"] = ""
    else:
        out["reverse_geocode_confidence_note"] = conf_full
    out["reverse_geocode_error"] = "" if err in (None, "") else str(err)
    return out


def enrich_csv(
    in_path: Path,
    out_path: Path,
    *,
    skip_geocode: bool = False,
    max_rows: int | None = None,
    cache_path: Path | None = None,
) -> int:
    in_path = in_path.expanduser().resolve()
    out_path = out_path.expanduser().resolve()
    cpath = (cache_path or _CACHE_DEFAULT).resolve()

    with in_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []

    if max_rows is not None:
        rows = rows[: max(0, max_rows)]

    mem: Dict[str, Dict[str, Any]] = {}
    enriched: List[Dict[str, Any]] = [enrich_row(r, skip_geocode=skip_geocode, mem=mem, cache_path=cpath) for r in rows]

    fields = list(header) + [c for c in _EXTRA_FIELDS if c not in header]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in enriched:
            w.writerow({k: r.get(k, "") for k in fields})

    return len(enriched)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add assessor portals + OSM reverse geocode hints")
    parser.add_argument("--in", dest="in_path", type=Path, default=ROOT / "roof_hunter/output/actual_events_5_to_14_days.csv")
    parser.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=None,
        help="Default: sibling actual_events_5_to_14_days_assessor.csv next to input",
    )
    parser.add_argument("--skip-geocode", action="store_true", help="Portals only (no Nominatim network calls)")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--cache", type=Path, default=_CACHE_DEFAULT, help="JSON cache for reverse geocode")
    args = parser.parse_args()
    out = args.out_path or args.in_path.with_name(args.in_path.stem + "_assessor.csv")
    n = enrich_csv(args.in_path, out, skip_geocode=args.skip_geocode, max_rows=args.max_rows, cache_path=args.cache)
    print(f"Wrote {n} rows to {out}")


if __name__ == "__main__":
    main()
