"""Extract labeled building chips from the xView2 dataset for CLIP fine-tuning.

xView2 dataset structure (after extracting the download):
    <xview2_root>/
        train/
            images/   ← *_pre_disaster.png  *_post_disaster.png  (1024×1024 RGB)
            labels/   ← *_pre_disaster.json  *_post_disaster.json
        tier3/        ← same structure
        test/         ← no labels (held-out)

Label JSON format (xy section):
    {"features": {"xy": [
        {"wkt": "POLYGON ((x1 y1, x2 y2, ...))", "properties": {"subtype": "major-damage"}}
    ]}}

Damage class mapping for roof replacement leads:
    no-damage       → 0  (intact — negative class)
    minor-damage    → 0  (not a replacement candidate)
    major-damage    → 1  (replacement candidate — positive class)
    destroyed       → 1  (replacement candidate — positive class)
    un-classified   → skipped

Two extraction modes (set with --mode):
  building  Single building chips: padded bounding box around each polygon.
            Best for fine-grained visual patterns.
  scene     224×224 neighbourhood patches tiled from each scene image.
            Label = 1 if ≥ SCENE_DAMAGE_RATIO buildings in patch are major/destroyed.
            Better match for Sentinel-2 10 m/px resolution (patch ≈ 2 km × 2 km).

Usage:
    python -m roof_hunter.xview2_chip_extractor \\
        --root ~/Downloads/xview2 \\
        --out  roof_hunter/chips/xview2 \\
        --mode scene \\
        --splits train tier3 \\
        --filter-disaster wind,hurricane

    # Then verify the balance:
    python -m roof_hunter.xview2_chip_extractor --root ... --out ... --stats-only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

# ── constants ─────────────────────────────────────────────────────────────────

DAMAGE_TO_LABEL: Dict[str, Optional[int]] = {
    "no-damage":      0,
    "minor-damage":   0,
    "major-damage":   1,
    "destroyed":      1,
    "un-classified":  None,   # skip
}

# Disasters most similar to hail: wind-driven structural damage to rooftops
_WIND_EVENTS = {
    "hurricane", "wind", "tornado", "typhoon", "cyclone",
}

# Minimum building area in pixels to include (filters out tiny polygons)
_MIN_AREA_PX = 64

# Building chip padding (pixels added around the bounding box)
_PAD_PX = 16

# Scene chip size (pixels)
_SCENE_CHIP_PX = 224

# Stride for scene tiling (< chip size = overlapping patches)
_SCENE_STRIDE_PX = 112

# Scene is labelled damaged if this fraction of buildings in window are major/destroyed
_SCENE_DAMAGE_RATIO = 0.10


# ── WKT polygon parser ────────────────────────────────────────────────────────

def _parse_wkt_polygon(wkt: str) -> Optional[List[Tuple[float, float]]]:
    """Parse POLYGON ((x1 y1, x2 y2, ...)) → list of (x, y) tuples, or None."""
    m = re.search(r"POLYGON\s*\(\((.+?)\)\)", wkt, re.IGNORECASE)
    if not m:
        return None
    coords_str = m.group(1)
    pts: List[Tuple[float, float]] = []
    for pair in coords_str.split(","):
        parts = pair.strip().split()
        if len(parts) >= 2:
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    return pts if len(pts) >= 3 else None


def _bbox(pts: List[Tuple[float, float]]) -> Tuple[int, int, int, int]:
    """Return (xmin, ymin, xmax, ymax) bounding box from polygon vertices."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _bbox_area(xmin: int, ymin: int, xmax: int, ymax: int) -> int:
    return max(0, xmax - xmin) * max(0, ymax - ymin)


# ── label JSON parser ─────────────────────────────────────────────────────────

def _load_label(json_path: Path) -> List[Dict[str, Any]]:
    """Return list of {"wkt": str, "subtype": str, "bbox": tuple} from xView2 JSON."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    features = (data.get("features") or {}).get("xy") or []
    buildings: List[Dict[str, Any]] = []
    for f in features:
        props = f.get("properties") or {}
        if props.get("feature_type", "building") != "building":
            continue
        wkt = f.get("wkt") or ""
        pts = _parse_wkt_polygon(wkt)
        if pts is None:
            continue
        subtype = str(props.get("subtype") or "un-classified").lower().strip()
        label = DAMAGE_TO_LABEL.get(subtype)  # None = skip
        bbox = _bbox(pts)
        area = _bbox_area(*bbox)
        buildings.append({
            "wkt": wkt,
            "subtype": subtype,
            "label": label,
            "bbox": bbox,
            "area": area,
        })
    return buildings


# ── disaster filter ───────────────────────────────────────────────────────────

def _is_wind_event(stem: str, filter_terms: Optional[List[str]]) -> bool:
    """Return True if the image filename looks like a wind/hurricane event."""
    if filter_terms is None:
        return True
    sl = stem.lower()
    return any(t in sl for t in filter_terms)


# ── building-chip extraction ──────────────────────────────────────────────────

def extract_building_chips(
    img_path: Path,
    label_json: Path,
    out_dir: Path,
    *,
    filter_terms: Optional[List[str]] = None,
    pad_px: int = _PAD_PX,
    min_area_px: int = _MIN_AREA_PX,
) -> Generator[Dict[str, Any], None, None]:
    """Yield chip metadata dicts; save chips to out_dir/0/ and out_dir/1/."""
    if not _is_wind_event(img_path.stem, filter_terms):
        return
    buildings = _load_label(label_json)
    if not buildings:
        return

    try:
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return

    w, h = img.size

    for i, bld in enumerate(buildings):
        if bld["label"] is None:
            continue
        if bld["area"] < min_area_px:
            continue
        xmin, ymin, xmax, ymax = bld["bbox"]
        # Apply padding, clamp to image bounds
        xmin = max(0, xmin - pad_px)
        ymin = max(0, ymin - pad_px)
        xmax = min(w, xmax + pad_px)
        ymax = min(h, ymax + pad_px)
        if xmax <= xmin or ymax <= ymin:
            continue

        chip = img.crop((xmin, ymin, xmax, ymax)).resize((224, 224))
        label = bld["label"]
        label_dir = out_dir / str(label)
        label_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{img_path.stem}_{i:04d}"
        chip_path = label_dir / f"{stem}.jpg"
        chip.save(chip_path, "JPEG", quality=92)

        yield {
            "chip_path": str(chip_path.relative_to(out_dir)),
            "label": label,
            "subtype": bld["subtype"],
            "source_image": img_path.name,
            "chip_index": i,
        }


# ── scene-chip extraction ─────────────────────────────────────────────────────

def extract_scene_chips(
    img_path: Path,
    label_json: Path,
    out_dir: Path,
    *,
    filter_terms: Optional[List[str]] = None,
    chip_px: int = _SCENE_CHIP_PX,
    stride_px: int = _SCENE_STRIDE_PX,
    damage_ratio_threshold: float = _SCENE_DAMAGE_RATIO,
) -> Generator[Dict[str, Any], None, None]:
    """Tile the scene into overlapping patches; label each patch by damage ratio.

    A patch is labelled 1 (damaged) if ≥ damage_ratio_threshold of the
    buildings whose centroids fall within it are major-damage or destroyed.
    """
    if not _is_wind_event(img_path.stem, filter_terms):
        return
    buildings = _load_label(label_json)
    if not buildings:
        return

    try:
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return

    W, H = img.size

    # Build centroid list for fast patch lookup
    cents: List[Tuple[float, float, int]] = []   # (cx, cy, label)
    for bld in buildings:
        if bld["label"] is None:
            continue
        xmin, ymin, xmax, ymax = bld["bbox"]
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        cents.append((cx, cy, bld["label"]))

    chip_idx = 0
    for top in range(0, H - chip_px + 1, stride_px):
        for left in range(0, W - chip_px + 1, stride_px):
            right = left + chip_px
            bot   = top + chip_px

            # Buildings whose centroid falls in this patch
            in_patch = [
                lbl for (cx, cy, lbl) in cents
                if left <= cx < right and top <= cy < bot
            ]
            if not in_patch:
                continue

            damaged = sum(1 for lbl in in_patch if lbl == 1)
            ratio = damaged / len(in_patch)
            patch_label = 1 if ratio >= damage_ratio_threshold else 0

            chip = img.crop((left, top, right, bot))
            label_dir = out_dir / str(patch_label)
            label_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{img_path.stem}_r{top}_c{left}"
            chip_path = label_dir / f"{stem}.jpg"
            chip.save(chip_path, "JPEG", quality=92)

            yield {
                "chip_path": str(chip_path.relative_to(out_dir)),
                "label": patch_label,
                "damage_ratio": round(ratio, 3),
                "buildings_in_patch": len(in_patch),
                "source_image": img_path.name,
                "chip_index": chip_idx,
            }
            chip_idx += 1


# ── dataset walk ──────────────────────────────────────────────────────────────

def iter_image_label_pairs(
    xview2_root: Path,
    splits: List[str],
    post_only: bool = True,
) -> Generator[Tuple[Path, Path], None, None]:
    """Yield (image_path, label_json_path) pairs from the requested splits."""
    suffix = "_post_disaster"
    for split in splits:
        split_dir = xview2_root / split
        img_dir   = split_dir / "images"
        lbl_dir   = split_dir / "labels"
        if not img_dir.is_dir():
            print(f"  Warning: {img_dir} not found — skipping split '{split}'", flush=True)
            continue
        for img_path in sorted(img_dir.glob("*.png")):
            if post_only and suffix not in img_path.stem:
                continue
            lbl_path = lbl_dir / img_path.with_suffix(".json").name
            if not lbl_path.is_file():
                continue
            yield img_path, lbl_path


# ── manifest writer ───────────────────────────────────────────────────────────

def write_manifest(records: List[Dict[str, Any]], out_dir: Path) -> None:
    path = out_dir / "manifest.csv"
    if not records:
        return
    keys = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)
    print(f"  Manifest: {path}  ({len(records)} chips)", flush=True)


# ── stats ─────────────────────────────────────────────────────────────────────

def print_stats(out_dir: Path) -> None:
    for cls in ("0", "1"):
        d = out_dir / cls
        if d.is_dir():
            n = len(list(d.glob("*.jpg")))
            label = "intact (0)" if cls == "0" else "damaged (1)"
            print(f"    {label}: {n:,} chips")


# ── main ──────────────────────────────────────────────────────────────────────

def run(
    xview2_root: Path,
    out_dir: Path,
    mode: str = "scene",
    splits: Optional[List[str]] = None,
    filter_terms: Optional[List[str]] = None,
    stats_only: bool = False,
) -> None:
    if stats_only:
        print(f"Chip stats in {out_dir}:")
        print_stats(out_dir)
        return

    splits = splits or ["train", "tier3"]
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, Any]] = []
    n_pairs = 0

    for img_path, lbl_path in iter_image_label_pairs(xview2_root, splits):
        n_pairs += 1
        if n_pairs % 50 == 0:
            print(f"  Processing image {n_pairs}… ({len(all_records):,} chips so far)", flush=True)

        if mode == "building":
            gen = extract_building_chips(
                img_path, lbl_path, out_dir,
                filter_terms=filter_terms,
            )
        else:
            gen = extract_scene_chips(
                img_path, lbl_path, out_dir,
                filter_terms=filter_terms,
            )

        for rec in gen:
            all_records.append(rec)

    print(f"\nProcessed {n_pairs} scene images → {len(all_records):,} chips total", flush=True)
    print_stats(out_dir)
    write_manifest(all_records, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="xView2 dataset root directory")
    ap.add_argument("--out", type=Path,
                    default=Path("roof_hunter/chips/xview2"),
                    help="Output directory for chips")
    ap.add_argument("--mode", choices=["building", "scene"], default="scene",
                    help="'scene' patches (recommended) or per-building crops")
    ap.add_argument("--splits", nargs="+", default=["train", "tier3"],
                    help="Dataset splits to use (train / tier3 / test)")
    ap.add_argument(
        "--filter-disaster", default="wind,hurricane,tornado",
        help="Comma-separated disaster keywords to keep (empty = keep all). "
             "Wind/hurricane events are most relevant to hail damage patterns.",
    )
    ap.add_argument("--no-filter", action="store_true",
                    help="Keep all disaster types (ignores --filter-disaster)")
    ap.add_argument("--stats-only", action="store_true",
                    help="Just print class counts from an existing chip directory")
    args = ap.parse_args()

    filter_terms: Optional[List[str]] = None
    if not args.no_filter and args.filter_disaster.strip():
        filter_terms = [t.strip().lower() for t in args.filter_disaster.split(",") if t.strip()]

    run(
        xview2_root=args.root.expanduser().resolve(),
        out_dir=args.out.expanduser().resolve(),
        mode=args.mode,
        splits=args.splits,
        filter_terms=filter_terms,
        stats_only=args.stats_only,
    )


if __name__ == "__main__":
    main()
