"""High-value ZIP (ZCTA) tiers + last-60d severe reports + next-4h hail outlook.

Outputs under ``roof_hunter/output/``:
- ``rich_zip_tiers_ok_tx.csv`` — tier1/tier2 ZCTAs ranked by ACS5 median household income
- ``rich_zip_events_last60d_tier1.csv`` / ``_tier2.csv`` — SPC reports (OKC & DFW hubs) with ZIP,
  wealth tier, and **full reverse-geocoded addresses** (Nominatim; cached)
- ``rich_zip_large_hail_geocoded_ok_tx.csv`` — hail LSR points **≥ min inches** (default 2.0) in OK/TX
  within ``--large-hail-lookback-days`` (default 30); includes structured address fields.
  (SPC files give **report lat/lon points**, not polygon damage-path geometry.)

Requires ``geopy`` for reverse geocoding (cached). Census ACS calls are unauthenticated GETs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roof_hunter.integrations.affluent_zcta_seeds import seed_zips  # noqa: E402
from roof_hunter.lead_ops_report import REGIONS, _severity_score  # noqa: E402
from roof_hunter.integrations.hrrr_column_ingest import fetch_hrrr_column_features  # noqa: E402
from roof_hunter.simulator import HailPhysicsEngine  # noqa: E402
from roof_hunter.validate_last_week import fetch_spc_reports  # noqa: E402

_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
_ZIP_CACHE_PATH = _CACHE_DIR / "geopy_zip_cache.json"
_GEOCODE_CACHE_PATH = _CACHE_DIR / "nominatim_reverse_cache.json"

_CHUNK = 40
_TIER1_N = 30
_TIER2_N = 30
_HUB_RADIUS_KM = 200.0
_FORECAST_HOURS = 4.0
_RICH_ZIP_TRIGGER = 0.12  # projected_damage_prob_gt_1in


def _http_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "QuLabInfinite-RoofHunter/rich-zip-report"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_acs_median_income(zips: Sequence[str]) -> Dict[str, int]:
    """ZCTA -> median household income (negative / NoneACS codes skipped)."""
    out: Dict[str, int] = {}
    zlist = sorted({z.strip() for z in zips if z and len(z.strip()) == 5})
    for i in range(0, len(zlist), _CHUNK):
        chunk = zlist[i : i + _CHUNK]
        zparam = ",".join(chunk)
        url = (
            "https://api.census.gov/data/2022/acs/acs5"
            f"?get=NAME,B19013_001E&for=zip%20code%20tabulation%20area:{zparam}"
        )
        try:
            data = _http_json(url)
        except urllib.error.HTTPError:
            continue
        if not isinstance(data, list) or len(data) < 2:
            continue
        for row in data[1:]:
            if len(row) < 3:
                continue
            income_s = row[1]
            z = row[-1]
            try:
                income = int(income_s)
            except ValueError:
                continue
            if income <= 0:  # -666666666 missing
                continue
            out[str(z)] = income
        time.sleep(0.15)
    return out


def rank_tiers(income: Dict[str, int], state: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    z = seed_zips(state)
    rows = [{"zip": k, "median_household_income_acs2022": income[k], "state": state} for k in z if k in income]
    rows.sort(key=lambda r: -r["median_household_income_acs2022"])
    tier1 = rows[:_TIER1_N]
    tier2 = rows[_TIER1_N : _TIER1_N + _TIER2_N]
    return tier1, tier2


def _geocode_cache_key(lat: float, lon: float) -> str:
    return f"{round(lat, 4)},{round(lon, 4)}"


def _load_geocode_detail_cache() -> Dict[str, Dict[str, str]]:
    if _GEOCODE_CACHE_PATH.is_file():
        try:
            raw = json.loads(_GEOCODE_CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw = {}
        if isinstance(raw, dict):
            out: Dict[str, Dict[str, str]] = {}
            for k, v in raw.items():
                if isinstance(v, dict):
                    out[str(k)] = {str(kk): str(vv) if vv is not None else "" for kk, vv in v.items()}
                elif isinstance(v, str):
                    # legacy: value was ZIP only
                    out[str(k)] = {**_empty_geocode_row(), "zip": v, "postcode": v}
            return out
    # migrate legacy ZIP-only cache
    if _ZIP_CACHE_PATH.is_file():
        try:
            old = json.loads(_ZIP_CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old = {}
        if isinstance(old, dict):
            return {
                str(k): {**_empty_geocode_row(), "zip": str(v), "postcode": str(v)}
                for k, v in old.items()
                if v
            }
    return {}


def _empty_geocode_row() -> Dict[str, str]:
    return {
        "zip": "",
        "display_name": "",
        "house_number": "",
        "road": "",
        "city": "",
        "county": "",
        "state": "",
        "postcode": "",
    }


def _save_geocode_detail_cache(mem: Dict[str, Dict[str, str]]) -> None:
    _GEOCODE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _GEOCODE_CACHE_PATH.write_text(json.dumps(mem, indent=0), encoding="utf-8")


def _pick_city_from_nominatim_address(addr: Dict[str, Any]) -> str:
    for k in ("city", "town", "village", "hamlet", "municipality", "suburb"):
        v = addr.get(k)
        if v:
            return str(v)
    return ""


def _pick_county_from_nominatim_address(addr: Dict[str, Any]) -> str:
    v = addr.get("county") or addr.get("ISO3166-2-lvl6")
    return str(v) if v else ""


def reverse_geocode_detail(lat: float, lon: float, cache: Dict[str, Dict[str, str]], sleep_s: float) -> Dict[str, str]:
    """Reverse geocode with Nominatim; returns ZIP, display line, and common OSM address parts."""
    key = _geocode_cache_key(lat, lon)
    if key in cache:
        return {**_empty_geocode_row(), **cache[key]}

    from geopy.geocoders import Nominatim  # lazy

    row = _empty_geocode_row()
    geolocator = Nominatim(user_agent="QuLabInfinite-RoofHunter/1.3-rich-zip-report", timeout=25)
    time.sleep(max(0.0, sleep_s))
    loc = geolocator.reverse((lat, lon), language="en")
    if loc and loc.raw:
        addr = loc.raw.get("address") or {}
        pc = str(addr.get("postcode") or "")
        m = re.match(r"(\d{5})", pc)
        row["zip"] = m.group(1) if m else ""
        row["postcode"] = pc[:12] if pc else ""
        row["display_name"] = str(loc.raw.get("display_name") or loc.address or "")
        row["house_number"] = str(addr.get("house_number") or "")
        row["road"] = str(addr.get("road") or addr.get("pedestrian") or addr.get("residential") or "")
        row["city"] = _pick_city_from_nominatim_address(addr)
        row["county"] = _pick_county_from_nominatim_address(addr)
        row["state"] = str(addr.get("state") or "")

    cache[key] = row.copy()
    return row


def zip_tier_label(z: str, tier1_set: set, tier2_set: set) -> str:
    if not z:
        return "unknown_zip"
    if z in tier1_set:
        return "tier1"
    if z in tier2_set:
        return "tier2"
    return "other_zip"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=60, help="SPC lookback window")
    ap.add_argument("--geocode-sleep", type=float, default=1.05, help="Seconds between Nominatim calls")
    ap.add_argument(
        "--max-geocode",
        type=int,
        default=200,
        help="Cap reverse-geocode calls for tier1/tier2 event CSVs (remaining rows get blank address fields)",
    )
    ap.add_argument(
        "--large-hail-min-inches",
        type=float,
        default=2.0,
        help="Minimum SPC hail size (inches) for the large-hail geocoded export",
    )
    ap.add_argument(
        "--large-hail-lookback-days",
        type=int,
        default=30,
        help="Lookback days for large-hail geocoded export (OK/TX hub radius)",
    )
    ap.add_argument(
        "--max-large-hail-geocode",
        type=int,
        default=400,
        help="Cap Nominatim calls for large-hail export (uses same cache as other rows)",
    )
    ap.add_argument("--skip-forecast", action="store_true", help="Skip HRRR downloads; only tiers + events")
    ap.add_argument(
        "--skip-large-hail-export",
        action="store_true",
        help="Skip rich_zip_large_hail_geocoded_ok_tx.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Output directory",
    )
    args = ap.parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching ACS median income for seeded ZCTAs…", flush=True)
    ok_inc = fetch_acs_median_income(seed_zips("OK"))
    tx_inc = fetch_acs_median_income(seed_zips("TX"))
    ok_t1, ok_t2 = rank_tiers(ok_inc, "OK")
    tx_t1, tx_t2 = rank_tiers(tx_inc, "TX")

    tier1_zips = {r["zip"] for r in ok_t1 + tx_t1}
    tier2_zips = {r["zip"] for r in ok_t2 + tx_t2}

    tier_path = out_dir / "rich_zip_tiers_ok_tx.csv"
    with tier_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["tier", "state", "zip", "median_household_income_acs2022", "notes"],
        )
        w.writeheader()
        for r in ok_t1 + tx_t1:
            w.writerow({**r, "tier": "tier1", "notes": "top median income among seeds"})
        for r in ok_t2 + tx_t2:
            w.writerow({**r, "tier": "tier2", "notes": "next tier among seeds"})
    print(f"Wrote {tier_path}", flush=True)

    end_d = date.today()
    start_d = end_d - timedelta(days=args.days)
    print(f"SPC fetch {start_d} .. {end_d} (hubs {REGIONS})…", flush=True)

    event_rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str, str]] = set()
    for region, hlat, hlon in REGIONS:
        reports = fetch_spc_reports(hlat, hlon, start_d, end_d, radius_km=_HUB_RADIUS_KM)
        for rep in reports:
            sev = _SeverityScore(rep)
            if sev < 0.65:
                continue
            st = str(rep.get("state") or "").upper()
            if st not in {"OK", "TX"}:
                continue
            rdt = str(rep.get("report_datetime") or "")
            latk = str(rep.get("lat"))
            lonk = str(rep.get("lon"))
            typ = str(rep.get("type") or "")
            key = (rdt, latk, lonk, typ)
            if key in seen:
                continue
            seen.add(key)
            hs = ""
            if typ == "hail" and rep.get("size_in") is not None:
                try:
                    hs = round(float(rep["size_in"]), 2)
                except (TypeError, ValueError):
                    hs = ""
            event_rows.append(
                {
                    "hub_region": region,
                    "report_datetime": rep.get("report_datetime"),
                    "report_type": rep.get("type"),
                    "severity_score_0_1": sev,
                    "hail_size_in": hs,
                    "spc_location": rep.get("location"),
                    "county": rep.get("county"),
                    "state": st,
                    "lat": rep.get("lat"),
                    "lon": rep.get("lon"),
                    "distance_km_from_hub": rep.get("distance_km"),
                    "comments": rep.get("comments"),
                }
            )

    geocode_cache = _load_geocode_detail_cache()
    geo_n = 0
    for er in event_rows:
        er["inferred_zip"] = ""
        er["full_address"] = ""
        er["geocode_house_number"] = ""
        er["geocode_road"] = ""
        er["geocode_city"] = ""
        er["geocode_county"] = ""
        er["geocode_state"] = ""
        er["geocode_postcode"] = ""
        try:
            lat = float(er["lat"])
            lon = float(er["lon"])
        except (TypeError, ValueError):
            pass
        else:
            if geo_n < args.max_geocode:
                det = reverse_geocode_detail(lat, lon, geocode_cache, args.geocode_sleep)
                er["inferred_zip"] = det["zip"]
                er["full_address"] = det["display_name"]
                er["geocode_house_number"] = det["house_number"]
                er["geocode_road"] = det["road"]
                er["geocode_city"] = det["city"]
                er["geocode_county"] = det["county"]
                er["geocode_state"] = det["state"]
                er["geocode_postcode"] = det["postcode"]
                geo_n += 1
        er["wealth_zip_tier"] = zip_tier_label(er.get("inferred_zip") or "", tier1_zips, tier2_zips)

    def write_events(path: Path, tier: str) -> None:
        subset = [r for r in event_rows if r["wealth_zip_tier"] == tier]
        fields = [
            "hub_region",
            "report_datetime",
            "report_type",
            "severity_score_0_1",
            "hail_size_in",
            "spc_location",
            "county",
            "state",
            "lat",
            "lon",
            "full_address",
            "geocode_house_number",
            "geocode_road",
            "geocode_city",
            "geocode_county",
            "geocode_state",
            "geocode_postcode",
            "inferred_zip",
            "wealth_zip_tier",
            "distance_km_from_hub",
            "comments",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in sorted(subset, key=lambda x: (-float(x["severity_score_0_1"]), str(x["report_datetime"]))):
                w.writerow(r)
        print(f"Wrote {len(subset)} rows {path}", flush=True)

    write_events(out_dir / "rich_zip_events_last60d_tier1.csv", "tier1")
    write_events(out_dir / "rich_zip_events_last60d_tier2.csv", "tier2")

    if not args.skip_large_hail_export:
        end_h = date.today()
        start_h = end_h - timedelta(days=args.large_hail_lookback_days)
        print(
            f"Large hail export (≥{args.large_hail_min_inches} in) {start_h} .. {end_h}, hubs {REGIONS}…",
            flush=True,
        )
        large_rows: List[Dict[str, Any]] = []
        seen_h: set[Tuple[str, str, str]] = set()
        for region, hlat, hlon in REGIONS:
            reports = fetch_spc_reports(hlat, hlon, start_h, end_h, radius_km=_HUB_RADIUS_KM)
            for rep in reports:
                if rep.get("type") != "hail":
                    continue
                sz = rep.get("size_in")
                if sz is None:
                    continue
                try:
                    szf = float(sz)
                except (TypeError, ValueError):
                    continue
                if szf < args.large_hail_min_inches:
                    continue
                st = str(rep.get("state") or "").upper()
                if st not in {"OK", "TX"}:
                    continue
                rdt = str(rep.get("report_datetime") or "")
                latk = str(rep.get("lat"))
                lonk = str(rep.get("lon"))
                key3 = (rdt, latk, lonk)
                if key3 in seen_h:
                    continue
                seen_h.add(key3)
                large_rows.append(
                    {
                        "hub_region": region,
                        "report_datetime": rep.get("report_datetime"),
                        "spc_location": rep.get("location"),
                        "county_spc": rep.get("county"),
                        "state": st,
                        "lat": rep.get("lat"),
                        "lon": rep.get("lon"),
                        "hail_size_inches": round(szf, 2),
                        "distance_km_from_hub": rep.get("distance_km"),
                        "comments": rep.get("comments"),
                        "location_note": "SPC LSR point (lat/lon); not a vector damage swath from this product.",
                        "inferred_zip": "",
                        "full_address": "",
                        "geocode_house_number": "",
                        "geocode_road": "",
                        "geocode_city": "",
                        "geocode_county": "",
                        "geocode_state": "",
                        "geocode_postcode": "",
                        "wealth_zip_tier": "",
                    }
                )

        geocoded_hail = 0
        for lr in large_rows:
            try:
                lat = float(lr["lat"])
                lon = float(lr["lon"])
            except (TypeError, ValueError):
                lr["wealth_zip_tier"] = zip_tier_label("", tier1_zips, tier2_zips)
                continue
            if geocoded_hail < args.max_large_hail_geocode:
                det = reverse_geocode_detail(lat, lon, geocode_cache, args.geocode_sleep)
                lr["inferred_zip"] = det["zip"]
                lr["full_address"] = det["display_name"]
                lr["geocode_house_number"] = det["house_number"]
                lr["geocode_road"] = det["road"]
                lr["geocode_city"] = det["city"]
                lr["geocode_county"] = det["county"]
                lr["geocode_state"] = det["state"]
                lr["geocode_postcode"] = det["postcode"]
                geocoded_hail += 1
            lr["wealth_zip_tier"] = zip_tier_label(lr.get("inferred_zip") or "", tier1_zips, tier2_zips)

        lh_path = out_dir / "rich_zip_large_hail_geocoded_ok_tx.csv"
        lh_fields = [
            "hub_region",
            "report_datetime",
            "state",
            "county_spc",
            "spc_location",
            "hail_size_inches",
            "lat",
            "lon",
            "full_address",
            "geocode_house_number",
            "geocode_road",
            "geocode_city",
            "geocode_county",
            "geocode_state",
            "geocode_postcode",
            "inferred_zip",
            "wealth_zip_tier",
            "distance_km_from_hub",
            "comments",
            "location_note",
        ]
        with lh_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=lh_fields, extrasaction="ignore")
            w.writeheader()
            for r in sorted(
                large_rows,
                key=lambda x: (-float(x["hail_size_inches"]), str(x["report_datetime"])),
            ):
                w.writerow(r)
        print(f"Wrote {len(large_rows)} rows {lh_path} (geocoded {geocoded_hail})", flush=True)

    _save_geocode_detail_cache(geocode_cache)

    fc_fields = [
        "hub_region",
        "forecast_window_hours",
        "best_step_timestamp_utc",
        "max_projected_damage_prob_gt_1in_in_window",
        "batchdata_tier_used",
        "trigger_threshold_prob_gt_1in",
        "state",
        "target_zip",
        "zcta_median_household_income",
        "suggested_batchdata_address",
        "notes",
    ]
    forecast_targets: List[Dict[str, Any]] = []

    if not args.skip_forecast:
        # Next ~4h hail from HRRR column (hub points only; hourly leads f00–f04).
        run_ts = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        window_end = datetime.now(timezone.utc) + timedelta(hours=_FORECAST_HOURS)
        now_utc = datetime.now(timezone.utc)
        fxx = [0, 1, 2, 3, 4]
        engine = HailPhysicsEngine(iterations=2500, seed=42)

        for region, lat, lon in REGIONS:
            feats = fetch_hrrr_column_features(lat, lon, run_ts, fxx)
            best_p = 0.0
            best_ts = ""
            for feat in feats:
                ts = feat["timestamp"]
                if not isinstance(ts, str):
                    continue
                tdt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if tdt.tzinfo is None:
                    tdt = tdt.replace(tzinfo=timezone.utc)
                if tdt < now_utc or tdt > window_end:
                    continue
                sim = engine.run_from_hrrr_column_features(feat)
                p = float(sim["damage_probability_gt_1in"])
                if p > best_p:
                    best_p = p
                    best_ts = ts
            st = "TX" if "TX" in region else "OK"
            use_tier1 = best_p >= _RICH_ZIP_TRIGGER
            zrows = (ok_t1 if use_tier1 else ok_t2) if st == "OK" else (tx_t1 if use_tier1 else tx_t2)
            for zr in zrows:
                forecast_targets.append(
                    {
                        "hub_region": region,
                        "forecast_window_hours": _FORECAST_HOURS,
                        "best_step_timestamp_utc": best_ts,
                        "max_projected_damage_prob_gt_1in_in_window": round(best_p, 4),
                        "batchdata_tier_used": "tier1" if use_tier1 else "tier2",
                        "trigger_threshold_prob_gt_1in": _RICH_ZIP_TRIGGER,
                        "state": st,
                        "target_zip": zr["zip"],
                        "zcta_median_household_income": zr["median_household_income_acs2022"],
                        "suggested_batchdata_address": "",
                        "notes": "HRRR is sampled at metro hub coords only; use ZIP lists as BatchData search anchors.",
                    }
                )
    else:
        for region, _, _ in REGIONS:
            st = "TX" if "TX" in region else "OK"
            z_primary = tx_t1 if st == "TX" else ok_t1
            z_backup = tx_t2 if st == "TX" else ok_t2
            for pool_name, zrows in (("tier1_rich_zip_watchlist", z_primary), ("tier2_next_best", z_backup)):
                for zr in zrows:
                    forecast_targets.append(
                        {
                            "hub_region": region,
                            "forecast_window_hours": _FORECAST_HOURS,
                            "best_step_timestamp_utc": "",
                            "max_projected_damage_prob_gt_1in_in_window": "",
                            "batchdata_tier_used": pool_name,
                            "trigger_threshold_prob_gt_1in": _RICH_ZIP_TRIGGER,
                            "state": st,
                            "target_zip": zr["zip"],
                            "zcta_median_household_income": zr["median_household_income_acs2022"],
                            "suggested_batchdata_address": "",
                            "notes": "Forecast skipped; both rich (tier1) and next-best (tier2) ZIP pools listed for BatchData.",
                        }
                    )

    fc_path = out_dir / "rich_zip_forecast_next4h_batchdata.csv"
    with fc_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fc_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(forecast_targets)
    print(f"Wrote {len(forecast_targets)} forecast target rows {fc_path}", flush=True)


def _SeverityScore(rep: Dict[str, Any]) -> float:
    """Delegate to lead_ops severity with dict shape SPC uses."""
    return _severity_score(
        {
            "type": rep.get("type"),
            "location": rep.get("location"),
            "comments": rep.get("comments"),
        }
    )


if __name__ == "__main__":
    main()
