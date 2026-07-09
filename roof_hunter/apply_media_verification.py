"""Merge agent media verification results into Oklahoma lead packs.

Input 1: base lead CSV from oklahoma_lead_pack.py (must contain `lead_id`)
Input 2: verification CSV with at least:
  - lead_id
  - image_verified (yes/no)
  - verification_image_source_type (must be `satellite` for accepted verification)
  - one_property_damage_visible (yes/no)
Optional:
  - visual_damage_confidence_0_1
  - visual_damage_type
  - image_capture_timestamp_utc
  - image_source_url
  - verification_notes

Output:
  - <base>_verified.csv
  - <base>_ready_to_call.csv (satellite + one-property verified prioritized)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List


VERIFICATION_COLUMNS = [
    "image_verified",
    "verification_image_source_type",
    "one_property_damage_visible",
    "visual_damage_confidence_0_1",
    "visual_damage_type",
    "image_capture_timestamp_utc",
    "image_source_url",
    "verification_notes",
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _satellite_verified(row: Dict[str, Any]) -> bool:
    return (
        _truthy(row.get("image_verified"))
        and str(row.get("verification_image_source_type", "")).strip().lower() == "satellite"
        and _truthy(row.get("one_property_damage_visible"))
    )


def merge(base_csv: Path, verification_csv: Path, out_csv: Path, ready_csv: Path) -> None:
    base_rows = _read_csv(base_csv)
    ver_rows = _read_csv(verification_csv)
    ver_map = {str(r.get("lead_id", "")).strip(): r for r in ver_rows if str(r.get("lead_id", "")).strip()}

    merged: List[Dict[str, Any]] = []
    for row in base_rows:
        lead_id = str(row.get("lead_id", "")).strip()
        v = ver_map.get(lead_id, {})
        out = dict(row)
        for k in VERIFICATION_COLUMNS:
            out[k] = str(v.get(k, "")).strip()
        merged.append(out)

    fields = list(merged[0].keys()) if merged else []
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in merged:
            w.writerow(r)

    # "Ready to call": only satellite + one-property verified leads float to top.
    ready = sorted(
        merged,
        key=lambda r: (
            0 if _satellite_verified(r) else 1,
            -_to_float(r.get("visual_damage_confidence_0_1"), 0.0),
            -_to_float(r.get("lead_rank_score"), 0.0),
        ),
    )
    with ready_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in ready:
            w.writerow(r)


def main() -> None:
    p = argparse.ArgumentParser(description="Merge media verification into Oklahoma lead CSV")
    p.add_argument("--base", type=Path, required=True, help="e.g. roof_hunter/output/oklahoma_leads_14d_all.csv")
    p.add_argument("--verification", type=Path, required=True, help="CSV from scanner agent with lead_id + verification cols")
    p.add_argument("--out", type=Path, default=None, help="Default: <base>_verified.csv")
    p.add_argument("--ready", type=Path, default=None, help="Default: <base>_ready_to_call.csv")
    args = p.parse_args()

    out_csv = args.out or args.base.with_name(args.base.stem + "_verified.csv")
    ready_csv = args.ready or args.base.with_name(args.base.stem + "_ready_to_call.csv")
    merge(args.base, args.verification, out_csv, ready_csv)
    print(f"Wrote merged verification file: {out_csv}")
    print(f"Wrote prioritized ready-to-call file: {ready_csv}")


if __name__ == "__main__":
    main()
