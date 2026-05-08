"""Generate Oklahoma severe-weather lead packs for 7/14/30 day windows.

Outputs per window (default):
- oklahoma_leads_<N>d_all.csv  (segments after Oklahoma residential bias)
- oklahoma_leads_<N>d_residential.csv
- oklahoma_leads_<N>d_commercial.csv
- oklahoma_leads_<N>d_unclassified.csv

With ``--residential-only``, writes only:
- oklahoma_leads_<N>d_residential.csv

Segmentation uses a **consumer-roofing bias**: unknown or near-POI-commercial without
institutional keywords is classified residential; explicit school/store/church/etc.
stays commercial.

Notes:
- "image evidence" is inferred from SPC/NWS report comments (photo/video/social text),
  not binary-validated media ingestion.
- Addresses come from OSM reverse geocode nearest match and are high-quality hints;
  parcel-grade verification should still be done before outreach.
"""

from __future__ import annotations

import csv
import hashlib
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roof_hunter.integrations.osm_nominatim import reverse_geocode_cached
from roof_hunter.lead_ops_report import (
    _infer_property_segment_hint,
    _RES_HINT_COMMERCIAL,
    _severity_score,
)
from roof_hunter.validate_last_week import fetch_spc_daily_report, parse_spc_daily_report

WINDOW_DAYS: Sequence[int] = (7, 14, 30)
OUT_DIR = ROOT / "roof_hunter" / "output"
GEOCODE_CACHE = ROOT / "roof_hunter" / ".cache" / "nominatim_reverse_cache.json"

_IMG_HINTS = (
    "photo",
    "photos",
    "picture",
    "video",
    "social media",
    "facebook",
    "twitter",
    "x.com",
    "imgur",
    "youtube",
    "snapchat",
    "instagram",
    "captured on camera",
)
_HEAVY_DAMAGE_HINTS = (
    "destroyed",
    "collapsed",
    "major damage",
    "significant",
    "severe damage",
    "roof torn",
    "roof gone",
    "windows blown",
    "fatal",
    "inj",
    "ef-2",
    "ef-3",
    "ef-4",
    "ef-5",
    "baseball",
    "softball",
)
_COMMERCIAL_HINTS = (
    "business",
    "store",
    "shop",
    "warehouse",
    "plant",
    "factory",
    "school",
    "church",
    "hospital",
    "clinic",
    "airport",
    "air force base",
    "dealership",
    "restaurant",
    "hotel",
    "motel",
)
_RESIDENTIAL_HINTS = (
    "home",
    "homes",
    "house",
    "houses",
    "residence",
    "residential",
    "neighborhood",
    "subdivision",
    "apartment",
    "mobile home",
    "duplex",
    "condo",
    "townhome",
    "trailer",
    "housing",
    "outbuilding",
    "outbuildings",
    "farmstead",
    "dwellings",
    "residents",
)

# Institutional / obvious commercial — only rows matching this stay "commercial".
_OK_STRICT_COMMERCIAL: Tuple[str, ...] = tuple(sorted(set(_RES_HINT_COMMERCIAL + _COMMERCIAL_HINTS)))

def _strict_commercial_blob(blob: str) -> bool:
    return any(k in blob for k in _OK_STRICT_COMMERCIAL)


def _residential_signal_blob(blob: str) -> bool:
    return any(k in blob for k in _RESIDENTIAL_HINTS)


def _finalize_oklahoma_residential_bias(segment: str, comments: str, location: str) -> str:
    """Consumer-roofing bias: default to residential unless text is clearly institutional."""
    blob = f"{comments} {location}".lower()
    if _strict_commercial_blob(blob):
        return "commercial"
    if _residential_signal_blob(blob):
        return "residential"
    if segment == "commercial":
        # Often from reverse-geocode POI alone — treat as residential if not explicit above.
        return "residential"
    if segment == "unclassified":
        return "residential"
    return segment


def _split_unit(address_text: str) -> tuple[str, str]:
    text = (address_text or "").strip()
    if not text:
        return "", ""
    m = re.search(
        r"(?:\b(?:apt|apartment|unit|ste|suite)\.?\s+|#\s*)([A-Za-z0-9\-]+)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return text, ""
    unit_token = m.group(0).strip()
    return text, unit_token


def _iter_reports(start_d: date, end_d: date) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    d = start_d
    while d <= end_d:
        try:
            txt = fetch_spc_daily_report(d)
            out.extend(parse_spc_daily_report(txt, d))
        except Exception:
            pass
        d += timedelta(days=1)
    return out


def _image_evidence(comments: str) -> tuple[int, str]:
    c = comments.lower()
    hits = [h for h in _IMG_HINTS if h in c]
    if not hits:
        return 0, ""
    return min(3, len(hits)), ",".join(sorted(set(hits)))


def _damage_intensity_bonus(comments: str) -> float:
    c = comments.lower()
    bonus = 0.0
    for h in _HEAVY_DAMAGE_HINTS:
        if h in c:
            bonus += 0.08
    m = re.search(r"\b([2-5](?:\.\d+)?)\s*inch\b", c)
    if m:
        bonus += min(0.3, (float(m.group(1)) - 1.5) * 0.1)
    return min(0.9, bonus)


def _rank_score(rep: Dict[str, Any], sev: float, img_score: int) -> float:
    base = sev * 1.2
    type_bonus = 0.25 if str(rep.get("type", "")).lower() == "tornado" else 0.05
    img_bonus = img_score * 0.15
    dmg_bonus = _damage_intensity_bonus(str(rep.get("comments", "")))
    return round(base + type_bonus + img_bonus + dmg_bonus, 4)


def _lead_id(rep: Dict[str, Any]) -> str:
    raw = "|".join(
        [
            str(rep.get("report_datetime", "")),
            str(rep.get("type", "")),
            str(rep.get("lat", "")),
            str(rep.get("lon", "")),
            str(rep.get("county", "")),
            str(rep.get("location", "")),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _resolve_segment(spc_segment: str, geo_precision: str, near_poi: str) -> str:
    if spc_segment != "unclassified":
        return spc_segment
    if geo_precision == "house":
        return "residential"
    if near_poi:
        return "commercial"
    return "unclassified"


def _promote_unclassified(segment: str, comments: str, location: str, near_poi: str) -> str:
    if segment != "unclassified":
        return segment
    blob = f"{comments} {location}".lower()
    if any(k in blob for k in _RESIDENTIAL_HINTS):
        return "residential"
    if any(k in blob for k in _COMMERCIAL_HINTS):
        return "commercial"
    if near_poi:
        return "commercial"
    return "unclassified"


def build_window(days: int) -> List[Dict[str, Any]]:
    today = date.today()
    end_d = today - timedelta(days=1)
    start_d = today - timedelta(days=days)
    reports = _iter_reports(start_d, end_d)
    mem: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    geocode_limited = False
    for rep in reports:
        if str(rep.get("state", "")).upper() != "OK":
            continue
        rtype = str(rep.get("type", "")).lower()
        if rtype not in ("hail", "tornado"):
            continue
        sev = _severity_score(rep)
        if sev < 0.65:
            continue
        comments = str(rep.get("comments", ""))
        loc = str(rep.get("location", ""))
        img_score, img_terms = _image_evidence(comments)
        geo: Dict[str, Any] = {}
        if not geocode_limited:
            geo = reverse_geocode_cached(
                float(rep["lat"]),
                float(rep["lon"]),
                memory=mem,
                cache_path=GEOCODE_CACHE,
            )
            if str(geo.get("error", "")).strip() == "429":
                geocode_limited = True
        spc_segment = _infer_property_segment_hint(comments, loc)
        near_poi = str(geo.get("amenity_near") or "")
        resolved_segment = _resolve_segment(
            spc_segment=spc_segment,
            geo_precision=str(geo.get("reverse_geocode_precision") or ""),
            near_poi=near_poi,
        )
        resolved_segment = _promote_unclassified(
            resolved_segment,
            comments=comments,
            location=loc,
            near_poi=near_poi,
        )
        resolved_segment = _finalize_oklahoma_residential_bias(resolved_segment, comments, loc)
        score = _rank_score(rep, sev, img_score)
        fallback_addr = (
            f"{loc}, {rep.get('county', '')} County, Oklahoma".strip(", ")
            if loc
            else f"{rep.get('county', '')} County, Oklahoma".strip(", ")
        )
        addr_candidate = (geo.get("suggested_search_address") or "").strip() or fallback_addr
        display = (geo.get("display_name") or "").strip() or fallback_addr
        property_address = display if display else addr_candidate
        if not property_address:
            property_address = fallback_addr
        property_address, property_address_unit = _split_unit(property_address)
        image_verification_method = "satellite_property_damage_required"
        image_verification_status = "pending_satellite_review"
        image_verification_note = (
            "Only single-property satellite roof-damage proof qualifies as verified. "
            "Social posts/photos are excluded for final verification."
        )
        if img_score > 0:
            image_verification_method = "satellite_property_damage_required"
            image_verification_status = "pending_satellite_review"
            image_verification_note = (
                "SPC/NWS text indicates possible media, but accepted verification must be one-property satellite roof-damage imagery."
            )
        rows.append(
            {
                "lead_id": _lead_id(rep),
                "window_days": days,
                "lead_rank_score": score,
                "severity_score_0_1": sev,
                "image_evidence_score_0_3": img_score,
                "image_evidence_terms": img_terms,
                "image_verification_method": image_verification_method,
                "image_verification_status": image_verification_status,
                "image_verification_note": image_verification_note,
                "report_datetime": rep.get("report_datetime"),
                "report_type": rep.get("type"),
                "property_segment": resolved_segment,
                "spc_location": loc,
                "county": rep.get("county"),
                "state": rep.get("state"),
                "lat": rep.get("lat"),
                "lon": rep.get("lon"),
                "spc_comments": comments,
                "property_address": property_address,
                "property_address_unit": property_address_unit,
                "property_address_candidate": addr_candidate,
                "full_geocode_display": display,
                "geocode_precision": geo.get("reverse_geocode_precision") or "",
                "geocode_confidence_note": (
                    geo.get("reverse_geocode_confidence_note")
                    or ("Fallback from SPC location/county due geocoder throttle." if geocode_limited else "")
                ),
                "near_poi": near_poi,
                "geocode_error": geo.get("error") or "",
            }
        )

    rows.sort(
        key=lambda r: (
            -float(r["image_evidence_score_0_3"]),  # image-backed leads first
            -float(r["lead_rank_score"]),          # then worst estimated damage
            -float(r["severity_score_0_1"]),
            str(r["report_datetime"]),
        )
    )
    return rows


def _write(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "window_days",
        "lead_id",
        "lead_rank_score",
        "severity_score_0_1",
        "image_evidence_score_0_3",
        "image_evidence_terms",
        "image_verification_method",
        "image_verification_status",
        "image_verification_note",
        "report_datetime",
        "report_type",
        "property_segment",
        "spc_location",
        "county",
        "state",
        "lat",
        "lon",
        "spc_comments",
        "property_address",
        "property_address_unit",
        "property_address_candidate",
        "full_geocode_display",
        "geocode_precision",
        "geocode_confidence_note",
        "near_poi",
        "geocode_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--residential-only",
        action="store_true",
        help="Write only oklahoma_leads_<N>d_residential.csv per window (no all/commercial/unclassified/image_priority).",
    )
    args = parser.parse_args()

    summary: List[str] = []
    for days in WINDOW_DAYS:
        rows = build_window(days)
        res_path = OUT_DIR / f"oklahoma_leads_{days}d_residential.csv"
        n_res = sum(1 for r in rows if r["property_segment"] == "residential")
        if args.residential_only:
            _write(res_path, [r for r in rows if r["property_segment"] == "residential"])
            summary.append(f"{days}d: residential={n_res}")
            continue

        all_path = OUT_DIR / f"oklahoma_leads_{days}d_all.csv"
        com_path = OUT_DIR / f"oklahoma_leads_{days}d_commercial.csv"
        unc_path = OUT_DIR / f"oklahoma_leads_{days}d_unclassified.csv"
        _write(all_path, rows)
        _write(res_path, [r for r in rows if r["property_segment"] == "residential"])
        _write(com_path, [r for r in rows if r["property_segment"] == "commercial"])
        _write(unc_path, [r for r in rows if r["property_segment"] == "unclassified"])
        _write(
            OUT_DIR / f"oklahoma_leads_{days}d_image_priority.csv",
            [r for r in rows if int(r.get("image_evidence_score_0_3", 0)) > 0],
        )
        summary.append(f"{days}d: total={len(rows)} residential={n_res}")
    print("Wrote Oklahoma lead packs -> " + "; ".join(summary))


if __name__ == "__main__":
    main()
