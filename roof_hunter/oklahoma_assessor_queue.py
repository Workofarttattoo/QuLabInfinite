"""Create high-damage Oklahoma assessor lookup queue from generated lead packs.

This prepares top-scored rows for fast owner/address validation using county assessor portals.
It does not scrape assessor sites automatically.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roof_hunter.integrations.assessor_portals import get_county_property_portal

OUT = ROOT / "roof_hunter" / "output"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_queue(
    src: Path,
    out: Path,
    *,
    min_score: float = 1.39,
) -> int:
    with src.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    picked = [r for r in rows if _to_float(r.get("lead_rank_score")) >= min_score]
    picked.sort(key=lambda r: -_to_float(r.get("lead_rank_score")))

    out_rows: List[Dict[str, Any]] = []
    for r in picked:
        portal = get_county_property_portal(str(r.get("county", "")), "OK")
        search_url = portal.property_search_url if portal else ""
        org = portal.organization if portal else ""
        query = (r.get("property_address") or r.get("property_address_candidate") or "").strip()
        out_rows.append(
            {
                "lead_id": r.get("lead_id", ""),
                "lead_rank_score": r.get("lead_rank_score", ""),
                "severity_score_0_1": r.get("severity_score_0_1", ""),
                "report_datetime": r.get("report_datetime", ""),
                "report_type": r.get("report_type", ""),
                "property_segment": r.get("property_segment", ""),
                "county": r.get("county", ""),
                "state": r.get("state", ""),
                "property_address": r.get("property_address", ""),
                "property_address_unit": r.get("property_address_unit", ""),
                "assessor_organization": org,
                "assessor_property_search_url": search_url,
                "assessor_search_query": query,
                "assessor_lookup_status": "ready_manual_lookup",
                "owner_name": "",
                "assessor_confirmed_address": "",
                "parcel_id": "",
                "assessor_match_confidence": "",
                "assessor_notes": "",
            }
        )

    fields = list(out_rows[0].keys()) if out_rows else [
        "lead_id",
        "lead_rank_score",
        "severity_score_0_1",
        "report_datetime",
        "report_type",
        "property_segment",
        "county",
        "state",
        "property_address",
        "property_address_unit",
        "assessor_organization",
        "assessor_property_search_url",
        "assessor_search_query",
        "assessor_lookup_status",
        "owner_name",
        "assessor_confirmed_address",
        "parcel_id",
        "assessor_match_confidence",
        "assessor_notes",
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)
    return len(out_rows)


def main() -> None:
    src = OUT / "oklahoma_leads_14d_all.csv"
    out = OUT / "oklahoma_top_damage_assessor_queue_14d.csv"
    n = build_queue(src, out, min_score=1.39)
    print(f"Wrote {n} top-damage assessor queue rows to {out}")


if __name__ == "__main__":
    main()
