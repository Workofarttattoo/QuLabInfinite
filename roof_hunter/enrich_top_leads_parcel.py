"""Select top roof_hunter leads and attach parcel assessor fields (INCOG ArcGIS where available).

Writes a BatchData-friendly CSV: property + mailing hints plus full JSON attributes per row.

Example:

    python -m roof_hunter.enrich_top_leads_parcel \\
      --in roof_hunter/output/oklahoma_leads_14d_all_sentinel2.csv \\
      --out roof_hunter/output/oklahoma_leads_14d_elite_parcel_batchdata.csv \\
      --slim-out roof_hunter/output/oklahoma_leads_14d_elite_batchdata_upload.csv

Defaults target a **wider elite** slice (about 20 percent of rows, floor 35, cap 200) after residential-first sort.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roof_hunter.integrations.ok_incog_parcels import (  # noqa: E402
    COUNTY_TO_SERVICE,
    flatten_for_batchdata,
    normalize_county,
    query_parcel_point,
)


def _segment_rank(seg: str) -> int:
    s = (seg or "").strip().lower()
    if s == "residential":
        return 0
    if s == "unclassified":
        return 1
    if s == "commercial":
        return 2
    return 3


def _score(row: Dict[str, str]) -> float:
    try:
        return float(row.get("lead_rank_score") or 0.0)
    except ValueError:
        return 0.0


def select_top_rows(rows: List[Dict[str, str]], top_percent: float, min_top: int, max_top: int) -> List[Dict[str, str]]:
    """Same ordering as lead pack policy: residential first, then rank score descending."""
    pct = max(0.0, min(100.0, top_percent))
    n = len(rows)
    k = int(round(n * (pct / 100.0)))
    k = max(min_top, k)
    k = min(max_top, k, n)

    sorted_rows = sorted(
        rows,
        key=lambda r: (_segment_rank(r.get("property_segment", "")), -_score(r), r.get("county", ""), r.get("lead_id", "")),
    )
    return sorted_rows[:k]


_SLIM_COLS = (
    "lead_id",
    "lead_rank_score",
    "property_segment",
    "report_type",
    "county",
    "state",
    "lat",
    "lon",
    "parcel_enrichment_status",
    "parcel_enrichment_detail",
    "batch_apn",
    "batch_owner_name",
    "batch_property_address",
    "batch_property_city",
    "batch_property_zip",
    "batch_mailing_line1",
    "batch_mailing_city",
    "batch_mailing_state",
    "batch_mailing_zip",
    "parcel_total_acct_value",
    "parcel_legal",
    "property_address_candidate",
    "full_geocode_display",
)


def run(
    in_path: Path,
    out_path: Path,
    *,
    top_percent: float,
    min_top: int,
    max_top: int,
    sleep_s: float,
    state_filter: str,
    slim_out: Path | None,
) -> int:
    with in_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header_in = reader.fieldnames or []
        rows = list(reader)

    st = state_filter.strip().upper()
    if st:
        rows = [r for r in rows if (r.get("state") or "").strip().upper() == st]

    top = select_top_rows(rows, top_percent, min_top, max_top)

    extra_cols: Sequence[str] = (
        "parcel_enrichment_status",
        "parcel_enrichment_detail",
        "parcel_service_url",
        "parcel_attributes_json",
        "batch_apn",
        "batch_owner_name",
        "batch_property_address",
        "batch_property_city",
        "batch_property_zip",
        "batch_mailing_line1",
        "batch_mailing_city",
        "batch_mailing_state",
        "batch_mailing_zip",
        "parcel_year_built",
        "parcel_total_acct_value",
        "parcel_legal",
    )

    out_fields = list(dict.fromkeys(list(header_in) + list(extra_cols)))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        for row in top:
            lat_s = row.get("lat", "")
            lon_s = row.get("lon", "")
            county = row.get("county", "")
            out_row: Dict[str, Any] = dict(row)
            try:
                lat = float(lat_s)
                lon = float(lon_s)
            except (TypeError, ValueError):
                out_row["parcel_enrichment_status"] = "bad_coordinates"
                out_row["parcel_enrichment_detail"] = ""
                out_row["parcel_service_url"] = ""
                out_row["parcel_attributes_json"] = ""
                for c in extra_cols:
                    if c not in out_row:
                        out_row[c] = ""
                w.writerow(out_row)
                written += 1
                continue

            cnorm = normalize_county(county)
            if cnorm not in COUNTY_TO_SERVICE:
                out_row["parcel_enrichment_status"] = "no_free_parcel_layer"
                out_row["parcel_enrichment_detail"] = (
                    f"County {county!r} has no INCOG parcel endpoint in this tool; use BatchData or add a layer."
                )
                out_row["parcel_service_url"] = ""
                out_row["parcel_attributes_json"] = ""
                fd = {k: "" for k in flatten_for_batchdata({}).keys()}
                out_row.update(fd)
                w.writerow(out_row)
                written += 1
                continue

            res = query_parcel_point(lat, lon, county, sleep_s=sleep_s)
            out_row["parcel_service_url"] = res.service_url
            if res.matched:
                out_row["parcel_enrichment_status"] = "ok"
                out_row["parcel_enrichment_detail"] = ""
                out_row["parcel_attributes_json"] = json.dumps(res.parcel_attributes, ensure_ascii=False)
                out_row.update(flatten_for_batchdata(res.parcel_attributes))
            else:
                out_row["parcel_enrichment_status"] = "parcel_lookup_failed"
                out_row["parcel_enrichment_detail"] = res.arcgis_error
                out_row["parcel_attributes_json"] = ""
                out_row.update({k: "" for k in flatten_for_batchdata({}).keys()})

            w.writerow(out_row)
            written += 1

    if slim_out and written:
        slim_out = slim_out.expanduser().resolve()
        slim_out.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open(encoding="utf-8") as rf:
            reader = csv.DictReader(rf)
            slim_rows = list(reader)
        with slim_out.open("w", newline="", encoding="utf-8") as sf:
            sw = csv.DictWriter(sf, fieldnames=list(_SLIM_COLS), extrasaction="ignore")
            sw.writeheader()
            for row in slim_rows:
                sw.writerow({c: row.get(c, "") for c in _SLIM_COLS})

    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Top leads + INCOG parcel + BatchData-oriented CSV")
    here = Path(__file__).resolve().parent
    p.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=here / "output" / "oklahoma_leads_14d_all_sentinel2.csv",
        help="Lead CSV (e.g. oklahoma_leads_*_all_sentinel2.csv)",
    )
    p.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=None,
        help="Default: output/<input_stem>_top{N}pct_parcel_batchdata.csv",
    )
    p.add_argument(
        "--top-percent",
        type=float,
        default=20.0,
        help="Share of rows to keep after ordering (default ~20 = wider elite band)",
    )
    p.add_argument("--min-top", type=int, default=35, help="Minimum rows to output when the list is small")
    p.add_argument("--max-top", type=int, default=200, help="Cap rows (BatchData spend control)")
    p.add_argument("--sleep", type=float, default=0.12, help="Seconds between ArcGIS calls")
    p.add_argument("--state", default="OK", help="Filter rows by state (empty = all)")
    p.add_argument(
        "--slim-out",
        type=Path,
        default=None,
        help="Optional second CSV with BatchData-oriented columns only",
    )
    args = p.parse_args()

    in_path = args.in_path.expanduser().resolve()
    if args.out_path:
        out_path = args.out_path.expanduser().resolve()
    else:
        stem = in_path.stem + f"_top{args.top_percent:g}pct_parcel_batchdata"
        out_path = in_path.with_name(stem + ".csv")

    n = run(
        in_path,
        out_path,
        top_percent=args.top_percent,
        min_top=args.min_top,
        max_top=args.max_top,
        sleep_s=args.sleep,
        state_filter=args.state,
        slim_out=args.slim_out,
    )
    print(f"Wrote {n} rows to {out_path}")
    if args.slim_out:
        print(f"Slim upload CSV: {args.slim_out.resolve()}")


if __name__ == "__main__":
    main()
