"""Unified end-to-end lead pipeline: forecast → parcel → Sentinel-2 → priority sort → BatchData.

Pipeline stages:
  1. Ingest actual SPC events (last N days)            [lead_ops_report]
  2. Run HRRR forecast→parcel grid                     [forecast_parcel_pipeline]
  3. Append Sentinel-2 scene metadata                  [sentinel2_locator]
  4. Score satellite spectral damage confidence        [integrations/spectral_damage_analyzer]
  5. Priority sort (damage × recency × ZIP wealth)     [batchdata_client.sort_leads_for_batchdata]
  6. Select top-N pre-cleared leads
  7. Send to BatchData API (skip-trace)                [batchdata_client.BatchDataClient]

Usage:
    # Dry run (no API calls, writes sorted CSV only):
    python -m roof_hunter.unified_lead_sender --dry-run

    # Live send top 50 leads:
    python -m roof_hunter.unified_lead_sender --top 50 --out roof_hunter/output/sent_leads.csv
"""

from __future__ import annotations

import csv
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roof_hunter.batchdata_client import (
    BatchDataClient,
    BatchDataError,
    sort_leads_for_batchdata,
)
from roof_hunter.lead_ops_report import (
    build_actual_events_report,
    build_projected_report,
)
from roof_hunter.sentinel2_locator import append_sentinel2
from roof_hunter.rich_zip_opportunity_report import fetch_acs_median_income
from roof_hunter.integrations.affluent_zcta_seeds import seed_zips

_OUT_DIR = Path(__file__).resolve().parent / "output"


# ── income lookup builder ─────────────────────────────────────────────────────

def _build_income_lookup(states: Sequence[str] = ("OK", "TX")) -> Dict[str, float]:
    """Return zip → median household income from Census ACS (unauthenticated GET)."""
    print("Fetching ACS median income for ZIP wealth ranking…", flush=True)
    lookup: Dict[str, float] = {}
    for st in states:
        try:
            inc = fetch_acs_median_income(seed_zips(st))
            lookup.update({z: float(v) for z, v in inc.items()})
        except Exception as exc:
            print(f"  ACS fetch failed for {st}: {exc} (will use default income)", flush=True)
    print(f"  Income lookup: {len(lookup)} ZCTAs", flush=True)
    return lookup


# ── merge helpers ─────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def _assign_lead_ids(rows: List[Dict[str, Any]]) -> None:
    import hashlib
    for r in rows:
        if not r.get("lead_id"):
            blob = f"{r.get('lat','')}{r.get('lon','')}{r.get('report_datetime','')}"
            r["lead_id"] = hashlib.md5(blob.encode()).hexdigest()[:16]


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ── stage 1+2: collect leads ──────────────────────────────────────────────────

def collect_leads(
    lookback_days: int = 14,
    include_forecast: bool = True,
) -> List[Dict[str, Any]]:
    """Gather actual SPC events + optional HRRR forecast rows into one list."""
    leads: List[Dict[str, Any]] = []

    # Actual hail/tornado events
    print(f"Fetching SPC actual events (last {lookback_days}d)…", flush=True)
    actual = build_actual_events_report()
    for r in actual:
        r.setdefault("data_source", "spc_report")
    leads.extend(actual)
    print(f"  {len(actual)} actual event rows", flush=True)

    # Forward forecast
    if include_forecast:
        print("Fetching HRRR 48h projected hail…", flush=True)
        try:
            projected = build_projected_report()
            for r in projected:
                r.setdefault("data_source", "hrrr_forecast")
                # Map projected fields to common schema
                r.setdefault("report_datetime", r.get("timestamp_utc", ""))
                r.setdefault("lead_rank_score", r.get("projected_damage_prob_gt_1in", 0))
                r.setdefault("severity_score_0_1", r.get("projected_damage_prob_gt_1in", 0))
                r.setdefault("report_type", "projected_hail")
                r.setdefault("property_segment", "unclassified")
            leads.extend(projected)
            print(f"  {len(projected)} projected forecast rows", flush=True)
        except Exception as exc:
            print(f"  HRRR forecast failed (continuing without): {exc}", flush=True)

    _assign_lead_ids(leads)
    return leads


# ── stage 3: Sentinel-2 metadata ─────────────────────────────────────────────

def append_s2_metadata(rows: List[Dict[str, Any]], tmp_dir: Path) -> List[Dict[str, Any]]:
    """Write rows to temp CSV, run sentinel2_locator, reload with S2 columns."""
    tmp_in = tmp_dir / "_s2_input.csv"
    tmp_out = tmp_dir / "_s2_output.csv"
    _write_csv(tmp_in, rows)
    try:
        append_sentinel2(tmp_in, tmp_out, max_days_after_event=21, cloud_cover_max=50.0)
        result = _load_csv(tmp_out)
        print(f"  Sentinel-2 metadata: {sum(1 for r in result if r.get('sentinel2_status')=='ok')} scenes found", flush=True)
        return [dict(r) for r in result]
    except Exception as exc:
        print(f"  Sentinel-2 lookup failed (continuing without): {exc}", flush=True)
        return rows


# ── stage 4: satellite damage model (CLIP + spectral change) ─────────────────

def apply_satellite_damage_scores(rows: List[Dict[str, Any]], run_clip: bool = True) -> None:
    """Run CLIP + spectral change model on rows that have a Sentinel-2 scene.

    Writes the following columns back into each row:
      sentinel2_damage_confidence  — 0-1 score (None/empty if model could not run)
      sentinel2_damage_model_used  — 'clip+spectral' | 'spectral_only' | 'none'
      sentinel2_ndvi_change        — pre→post NDVI delta (positive = vegetation loss)
      sentinel2_nbr_change         — pre→post NBR delta (positive = structural exposure)
      sentinel2_swir_change        — pre→post SWIR ratio delta
      sentinel2_clip_damage_score  — raw CLIP damage probability
      sentinel2_evidence           — pipe-delimited evidence log
    """
    from roof_hunter.satellite_chip_damage_model import assess_lead_rows

    eligible = [r for r in rows if (r.get("sentinel2_status") or "").lower() == "ok"]
    ineligible = [r for r in rows if (r.get("sentinel2_status") or "").lower() != "ok"]

    for r in ineligible:
        _blank_satellite_cols(r)

    if not eligible:
        return

    print(f"  Running satellite damage model on {len(eligible)} scenes (CLIP={'on' if run_clip else 'off'})…", flush=True)
    assessments = assess_lead_rows(eligible, run_clip=run_clip)

    conf_map = {a.lead_id: a for a in assessments}
    for r in eligible:
        a = conf_map.get(str(r.get("lead_id") or ""))
        if a is None:
            _blank_satellite_cols(r)
            continue
        r["sentinel2_damage_confidence"] = a.damage_confidence if a.damage_confidence is not None else ""
        r["sentinel2_damage_model_used"] = a.model_used
        r["sentinel2_ndvi_change"] = a.ndvi_change if a.ndvi_change is not None else ""
        r["sentinel2_nbr_change"] = a.nbr_change if a.nbr_change is not None else ""
        r["sentinel2_swir_change"] = a.swir_change if a.swir_change is not None else ""
        r["sentinel2_clip_damage_score"] = a.clip_damage_score if a.clip_damage_score is not None else ""
        r["sentinel2_evidence"] = "; ".join(a.evidence)
        r["sentinel2_pre_scene_id"] = a.pre_scene_id
        r["sentinel2_pre_scene_utc"] = a.pre_scene_utc

    confirmed = sum(
        1 for r in eligible
        if _to_float(r.get("sentinel2_damage_confidence"), 0) >= 0.60
    )
    print(f"  Satellite model: {confirmed}/{len(eligible)} scenes ≥ 0.60 confidence", flush=True)


def _blank_satellite_cols(r: Dict[str, Any]) -> None:
    for col in (
        "sentinel2_damage_confidence", "sentinel2_damage_model_used",
        "sentinel2_ndvi_change", "sentinel2_nbr_change", "sentinel2_swir_change",
        "sentinel2_clip_damage_score", "sentinel2_evidence",
        "sentinel2_pre_scene_id", "sentinel2_pre_scene_utc",
    ):
        r.setdefault(col, "")


# ── stage 5+6: select top leads ───────────────────────────────────────────────

def _composite_lead_score(row: Dict[str, Any]) -> float:
    """Combine lead_rank_score with satellite damage confidence.

    Satellite confirmation is a multiplier, not just an additive boost.
    A lead with no satellite confirmation is penalised relative to one that
    has CLIP+spectral evidence — this enforces the requirement that addresses
    are sourced from actual imagery, not from text reports alone.
    """
    base = _to_float(row.get("lead_rank_score") or row.get("severity_score_0_1"), 0.3)
    sat_conf = _to_float(row.get("sentinel2_damage_confidence"), 0.0)
    if sat_conf >= 0.60:
        # Strong satellite confirmation: scale base up by up to 50%
        boost = 1.0 + 0.5 * ((sat_conf - 0.60) / 0.40)
        return base * boost
    elif sat_conf > 0:
        # Weak confirmation: modest additive nudge
        return base + 0.1 * sat_conf
    # No satellite data: pass base through unchanged so we don't drop all forecasts
    return base


def select_top_leads(
    rows: List[Dict[str, Any]],
    income_lookup: Dict[str, float],
    top_n: int,
    min_score: float = 0.05,
) -> List[Dict[str, Any]]:
    """Apply composite scoring, priority sort, and take top N pre-cleared leads.

    Sort order:
      1. Most damage + most recent (recent events with high damage_prob first)
      2. For ties in recency bracket: richest ZIP float to top
      3. Lower damage / older events fall to bottom
    """
    # Update lead_rank_score with satellite boost
    for r in rows:
        r["lead_rank_score"] = _composite_lead_score(r)

    # Filter obvious junk
    eligible = [r for r in rows if _to_float(r.get("lead_rank_score"), 0) >= min_score]
    print(f"  {len(eligible)}/{len(rows)} rows pass min_score={min_score}", flush=True)

    sorted_leads = sort_leads_for_batchdata(eligible, income_lookup)
    return sorted_leads[:top_n]


# ── stage 7: send to BatchData ────────────────────────────────────────────────

def send_to_batchdata(
    leads: List[Dict[str, Any]],
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    if dry_run:
        print(f"  [dry-run] Would send {len(leads)} leads to BatchData skip-trace", flush=True)
        for row in leads:
            row["bd_status"] = "dry_run"
        return leads

    try:
        client = BatchDataClient.from_env()
    except BatchDataError as exc:
        print(f"  BatchData client init failed: {exc}", flush=True)
        for row in leads:
            row["bd_status"] = "client_init_failed"
        return leads

    print(f"  Sending {len(leads)} leads to BatchData…", flush=True)
    return client.skip_trace_leads(leads)


# ── full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    *,
    out_path: Path,
    top_n: int = 100,
    lookback_days: int = 14,
    include_forecast: bool = True,
    dry_run: bool = False,
    states: Sequence[str] = ("OK", "TX"),
    min_score: float = 0.05,
) -> List[Dict[str, Any]]:
    tmp_dir = out_path.parent / ".tmp_pipeline"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1+2: collect
    leads = collect_leads(lookback_days=lookback_days, include_forecast=include_forecast)
    if not leads:
        print("No leads collected — aborting.", flush=True)
        return []

    # Stage 3: Sentinel-2
    print(f"Looking up Sentinel-2 scenes for {len(leads)} leads…", flush=True)
    leads = append_s2_metadata(leads, tmp_dir)

    # Stage 4: satellite damage model (CLIP + spectral change on real band data)
    print("Running CLIP + spectral change model on Sentinel-2 scenes…", flush=True)
    apply_satellite_damage_scores(leads, run_clip=True)

    # Stage 5: income lookup + priority sort
    income_lookup = _build_income_lookup(states)
    print(f"Selecting top {top_n} leads by priority score…", flush=True)
    top_leads = select_top_leads(leads, income_lookup, top_n=top_n, min_score=min_score)
    print(f"  Selected {len(top_leads)} leads", flush=True)

    if not top_leads:
        print("No leads cleared threshold — writing empty output.", flush=True)
        _write_csv(out_path, [])
        return []

    # Stage 6+7: send or dry-run
    results = send_to_batchdata(top_leads, dry_run=dry_run)

    # Write final output
    _write_csv(out_path, results)
    ok_n = sum(1 for r in results if r.get("bd_status") == "ok")
    print(
        f"Wrote {len(results)} rows to {out_path} "
        f"(bd_status=ok: {ok_n}, dry_run={dry_run})",
        flush=True,
    )
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path,
        default=_OUT_DIR / f"sent_leads_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.csv",
    )
    ap.add_argument("--top", dest="top_n", type=int, default=100,
                    help="Maximum leads to send to BatchData")
    ap.add_argument("--lookback", type=int, default=14,
                    help="SPC report lookback window (days)")
    ap.add_argument("--no-forecast", action="store_true",
                    help="Skip HRRR forecast rows; use only actual SPC events")
    ap.add_argument("--dry-run", action="store_true",
                    help="Sort and write CSV without making BatchData API calls")
    ap.add_argument("--min-score", type=float, default=0.05,
                    help="Minimum composite score to include a lead")
    args = ap.parse_args()

    results = run_pipeline(
        out_path=args.out.expanduser().resolve(),
        top_n=args.top_n,
        lookback_days=args.lookback,
        include_forecast=not args.no_forecast,
        dry_run=args.dry_run,
        min_score=args.min_score,
    )
    print(f"Pipeline complete. {len(results)} leads processed.", flush=True)


if __name__ == "__main__":
    main()
