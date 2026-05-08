"""Generate satellite-based verification CSV from Sentinel-2 enriched leads.

This is a practical baseline verifier, not a fully trained damage model.
It consumes rows from *_sentinel2.csv and outputs the verification schema used by
apply_media_verification.py.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image
    import numpy as np
except Exception:  # optional fallback path
    Image = None
    np = None


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _truthy(v: Any) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "y"}


def _fetch_preview_signal(preview_url: str, timeout_sec: float = 20.0) -> Tuple[Optional[float], str]:
    """Return simple texture score [0,1] from preview image edges/contrast."""
    if not preview_url:
        return None, "no_preview_url"
    if Image is None or np is None:
        return None, "pil_numpy_unavailable"
    try:
        req = urllib.request.Request(preview_url, headers={"User-Agent": "QuLab-RoofHunter/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            buf = resp.read()
        img = Image.open(BytesIO(buf)).convert("L")
        # Keep cost low
        img = img.resize((256, 256))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        gx = np.abs(arr[:, 1:] - arr[:, :-1]).mean()
        gy = np.abs(arr[1:, :] - arr[:-1, :]).mean()
        contrast = float(arr.std())
        # Heuristic scale
        raw = 0.55 * float(gx + gy) + 0.45 * contrast
        score = max(0.0, min(1.0, raw * 3.2))
        return score, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"preview_error:{type(exc).__name__}"


def _comment_damage_signal(comment: str) -> float:
    c = (comment or "").lower()
    hints = {
        "destroyed": 0.35,
        "severe damage": 0.28,
        "significant": 0.2,
        "damaged": 0.16,
        "roof": 0.18,
        "baseball": 0.2,
        "softball": 0.22,
        "ef-3": 0.35,
        "ef-2": 0.28,
        "ef-1": 0.18,
    }
    s = 0.0
    for k, w in hints.items():
        if k in c:
            s += w
    m = re.search(r"\b([2-5](?:\.\d+)?)\s*inch\b", c)
    if m:
        s += min(0.25, (float(m.group(1)) - 1.5) * 0.09)
    return max(0.0, min(1.0, s))


def _calc_confidence(row: Dict[str, str], preview_signal: Optional[float]) -> float:
    cloud = _to_float(row.get("sentinel2_cloud_cover_pct"), 100.0)
    cloud_q = max(0.0, min(1.0, 1.0 - (cloud / 100.0)))
    sev = _to_float(row.get("severity_score_0_1"), 0.0)
    rank = min(1.0, _to_float(row.get("lead_rank_score"), 0.0) / 1.8)
    comment_s = _comment_damage_signal(row.get("spc_comments", ""))
    texture = preview_signal if preview_signal is not None else 0.0
    # Weighted baseline tuned to avoid over-approving cloudy/low-evidence scenes.
    conf = 0.28 * sev + 0.2 * rank + 0.22 * cloud_q + 0.2 * comment_s + 0.1 * texture
    return max(0.0, min(1.0, conf))


def _single_property_visible(row: Dict[str, str]) -> bool:
    # Strongest proxy available without parcel polygons:
    # house-level geocode + non-empty property address + acceptable cloud.
    precision = (row.get("geocode_precision") or "").strip().lower()
    cloud = _to_float(row.get("sentinel2_cloud_cover_pct"), 100.0)
    addr = (row.get("property_address") or "").strip()
    return precision == "house" and bool(addr) and cloud <= 35.0


def build_verification_rows(in_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in in_rows:
        lead_id = (r.get("lead_id") or "").strip()
        status = (r.get("sentinel2_status") or "").strip().lower()
        preview_url = (r.get("sentinel2_preview_url") or "").strip()
        acquired = (r.get("sentinel2_acquired_utc") or "").strip()
        stac_url = (r.get("sentinel2_stac_item_url") or "").strip()
        data_url = (r.get("sentinel2_data_url") or "").strip()

        preview_signal, preview_note = _fetch_preview_signal(preview_url) if status == "ok" else (None, "scene_not_ok")
        conf = _calc_confidence(r, preview_signal) if status == "ok" else 0.0
        single_prop = _single_property_visible(r)

        # Conservative gate for auto-verified rows.
        verified = status == "ok" and conf >= 0.74 and single_prop and bool(data_url or stac_url)
        notes = []
        if status != "ok":
            notes.append(f"sentinel2_status={status}")
        if preview_note:
            notes.append(preview_note)
        if not single_prop:
            notes.append("single_property_not_confirmed")
        if conf < 0.74:
            notes.append(f"confidence_below_threshold:{conf:.3f}")

        out.append(
            {
                "lead_id": lead_id,
                "image_verified": "yes" if verified else "no",
                "verification_image_source_type": "satellite",
                "one_property_damage_visible": "yes" if single_prop else "no",
                "visual_damage_confidence_0_1": f"{conf:.3f}",
                "visual_damage_type": "roof_damage_likely" if conf >= 0.78 else "possible_roof_damage",
                "image_capture_timestamp_utc": acquired,
                "image_source_url": stac_url or data_url or preview_url,
                "verification_notes": "; ".join(notes),
            }
        )
    return out


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    p = argparse.ArgumentParser(description="Create satellite verification CSV from *_sentinel2 lead file")
    p.add_argument("--in", dest="in_csv", type=Path, required=True)
    p.add_argument("--out", dest="out_csv", type=Path, required=True)
    args = p.parse_args()

    in_rows = _read_csv(args.in_csv)
    out_rows = build_verification_rows(in_rows)
    fields = [
        "lead_id",
        "image_verified",
        "verification_image_source_type",
        "one_property_damage_visible",
        "visual_damage_confidence_0_1",
        "visual_damage_type",
        "image_capture_timestamp_utc",
        "image_source_url",
        "verification_notes",
    ]
    _write_csv(args.out_csv, out_rows, fields)

    verified_n = sum(1 for r in out_rows if _truthy(r.get("image_verified")))
    print(f"Wrote {len(out_rows)} verification rows to {args.out_csv} (auto-verified={verified_n})")


if __name__ == "__main__":
    main()
