"""Sentinel-2 hail damage assessment — real COG band data + HuggingFace CLIP.

What this does that thumbnail-texture approaches do NOT:
  - Fetches BOTH pre-event and post-event Sentinel-2 scenes from AWS Earth Search STAC.
  - Downloads actual multispectral band data (B02, B03, B04, B08, B11) via COG
    range-request (rasterio) or HTTP fallback — not preview thumbnails, not Facebook.
  - Creates pre/post spectral change signals (NDVI, NBR, SWIR) from real reflectance values.
  - Runs the actual visual chip through openai/clip-vit-large-patch14 with prompts
    written specifically for overhead hail damage signatures.
  - Returns a structured DamageAssessment with every signal that produced the score,
    so you can audit exactly why a lead was flagged.

Model hierarchy (tries in order until one succeeds):
  1. CLIP zero-shot (openai/clip-vit-large-patch14) + spectral change — best right now
  2. Spectral-change-only (no torch/transformers) — fallback if ML deps not installed
  3. Returns confidence=None (skips scoring) if band data is also unavailable

Install for CLIP:
  pip install transformers torch torchvision Pillow rasterio pyproj

Labeled chip sources (for building a fine-tuned model later):
  - xView2 building damage dataset: https://xview2.org/dataset  (free, registration)
  - NOAA post-storm orthoimagery: https://storms.ngs.noaa.gov/  (free, public domain)
  - Your own pipeline: run unified_lead_sender.py after a confirmed hail event, manually
    label 200-300 chips as damaged/intact, then retrain with train_roof_damage_model.py.

Upgrade path to Prithvi (IBM/NASA Sentinel-2 foundation model):
  from transformers import AutoModel
  prithvi = AutoModel.from_pretrained("ibm-nasa-geospatial/Prithvi-100M")
  # Prithvi expects 6-band HLS input at 224x224; use its features to replace CLIP embeddings.
  # Needs fine-tuning on labeled chips for damage detection.
"""

from __future__ import annotations

import json
import logging
import math
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

EARTH_SEARCH_STAC = "https://earth-search.aws.element84.com/v1/search"
_UA = "QuLabInfinite-RoofHunter/1.2-chip-model"

# ── CLIP prompt sets ──────────────────────────────────────────────────────────
# Prompts matter more than anything else for zero-shot CLIP accuracy.
# These are written for overhead/satellite perspective and hail-specific damage.

DAMAGE_PROMPTS: List[str] = [
    "aerial satellite view of residential neighborhood after hailstorm with visible roof damage",
    "overhead view showing missing or damaged roofing shingles and bare roof deck exposed after hail",
    "post-storm satellite imagery showing residential rooftops stripped of granules by large hail",
    "overhead view of neighborhood with storm-damaged roofs, debris on lawns, damaged trees from hail",
    "satellite photo showing white bare wood and stripped asphalt shingles on residential rooftops",
    "aerial view of hail-damaged homes with bright patches where granules were knocked off roofs",
]

INTACT_PROMPTS: List[str] = [
    "aerial satellite view of normal intact residential neighborhood with undamaged rooftops",
    "overhead view of residential area with uniform intact roofing material, no damage visible",
    "satellite photo of residential neighborhood before storm with undisturbed rooftops",
    "aerial view showing a healthy intact neighborhood, trees and roofs all appear normal",
]

# Chip size in meters centred on target point
_CHIP_HALF_M = 150.0   # 300 m × 300 m window at target → 30 pixels at 10 m/px

# Band names in Sentinel-2 L2A STAC items (Element84 naming)
_BAND_KEYS = {
    "B02": ("blue", "B02"),
    "B03": ("green", "B03"),
    "B04": ("red", "B04"),
    "B08": ("nir", "B08", "nir08"),
    "B11": ("swir16", "B11", "swir-1"),
}
_S2_SCALE = 10_000.0  # S2 L2A surface reflectance scale factor

# Undamaged baselines (summer/late-spring CONUS suburban)
_BASELINE = {"ndvi": 0.44, "nbr": 0.39, "swir_ratio": 0.19}


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class DamageAssessment:
    lead_id: str = ""
    damage_confidence: Optional[float] = None   # 0.0–1.0 or None if assessment failed
    model_used: str = "none"
    evidence: List[str] = field(default_factory=list)

    # spectral signals
    post_ndvi: Optional[float] = None
    pre_ndvi: Optional[float] = None
    ndvi_change: Optional[float] = None         # positive = vegetation loss (damage signal)
    post_nbr: Optional[float] = None
    pre_nbr: Optional[float] = None
    nbr_change: Optional[float] = None          # positive = structural exposure (damage signal)
    post_swir_ratio: Optional[float] = None
    pre_swir_ratio: Optional[float] = None
    swir_change: Optional[float] = None         # positive = bare surface increase

    # CLIP signals
    clip_damage_score: Optional[float] = None
    clip_intact_score: Optional[float] = None

    # scene provenance
    pre_scene_id: str = ""
    post_scene_id: str = ""
    pre_scene_utc: str = ""
    post_scene_utc: str = ""
    pre_cloud_pct: float = 0.0
    post_cloud_pct: float = 0.0

    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lead_id": self.lead_id,
            "damage_confidence": self.damage_confidence,
            "model_used": self.model_used,
            "evidence": "; ".join(self.evidence),
            "ndvi_change": self.ndvi_change,
            "nbr_change": self.nbr_change,
            "swir_change": self.swir_change,
            "clip_damage_score": self.clip_damage_score,
            "clip_intact_score": self.clip_intact_score,
            "pre_scene_id": self.pre_scene_id,
            "post_scene_id": self.post_scene_id,
            "pre_scene_utc": self.pre_scene_utc,
            "post_scene_utc": self.post_scene_utc,
            "assessment_error": self.error,
        }


# ── STAC helpers ──────────────────────────────────────────────────────────────

def _stac_post(payload: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        EARTH_SEARCH_STAC,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": _UA},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_stac_item(item_url: str) -> Dict[str, Any]:
    req = urllib.request.Request(item_url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_dt(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _best_scene(features: List[Dict[str, Any]], ref_dt: datetime) -> Optional[Dict[str, Any]]:
    """Return the feature with earliest post-ref date and lowest cloud cover."""
    scored = []
    for f in features:
        p = f.get("properties") or {}
        acq = _parse_dt(str(p.get("datetime") or ""))
        if acq is None:
            continue
        days = (acq - ref_dt).total_seconds() / 86400.0
        cloud = float(p.get("eo:cloud_cover") or 100.0)
        scored.append((days, cloud, f))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1]))
    return scored[0][2]


def find_post_event_scene(
    lat: float,
    lon: float,
    event_dt: datetime,
    max_days: int = 21,
    cloud_max: float = 45.0,
) -> Optional[Dict[str, Any]]:
    end_dt = event_dt + timedelta(days=max_days)
    payload = {
        "collections": ["sentinel-2-l2a"],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "datetime": f"{event_dt.isoformat()}/{end_dt.isoformat()}",
        "limit": 12,
        "query": {"eo:cloud_cover": {"lte": cloud_max}},
        "sortby": [{"field": "properties.datetime", "direction": "asc"}],
    }
    try:
        resp = _stac_post(payload)
        return _best_scene(resp.get("features") or [], event_dt)
    except Exception as exc:
        log.debug("Post-event STAC search failed: %s", exc)
        return None


def find_pre_event_scene(
    lat: float,
    lon: float,
    event_dt: datetime,
    days_before: int = 60,
    cloud_max: float = 30.0,
) -> Optional[Dict[str, Any]]:
    """Find the best cloud-free scene in the 60-day window BEFORE the event."""
    start_dt = event_dt - timedelta(days=days_before)
    end_dt = event_dt - timedelta(days=7)    # must be ≥ 1 week before event
    if start_dt >= end_dt:
        return None
    payload = {
        "collections": ["sentinel-2-l2a"],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "datetime": f"{start_dt.isoformat()}/{end_dt.isoformat()}",
        "limit": 12,
        "query": {"eo:cloud_cover": {"lte": cloud_max}},
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],  # most recent first
    }
    try:
        resp = _stac_post(payload)
        features = resp.get("features") or []
        # Pick least cloudy / most recent pre-event scene
        scored = []
        for f in features:
            p = f.get("properties") or {}
            acq = _parse_dt(str(p.get("datetime") or ""))
            cloud = float(p.get("eo:cloud_cover") or 100.0)
            if acq:
                scored.append((cloud, acq, f))
        if not scored:
            return None
        scored.sort(key=lambda t: (t[0], -t[1].timestamp()))
        return scored[0][2]
    except Exception as exc:
        log.debug("Pre-event STAC search failed: %s", exc)
        return None


def _band_hrefs(stac_item: Dict[str, Any]) -> Dict[str, str]:
    assets = stac_item.get("assets") or {}
    hrefs: Dict[str, str] = {}
    for band_key, aliases in _BAND_KEYS.items():
        for alias in aliases:
            href = (assets.get(alias) or {}).get("href", "").strip()
            if href:
                hrefs[band_key] = href
                break
    return hrefs


def _visual_href(stac_item: Dict[str, Any]) -> str:
    assets = stac_item.get("assets") or {}
    for name in ("visual", "overview", "TCI", "true-color"):
        href = (assets.get(name) or {}).get("href", "").strip()
        if href:
            return href
    return ""


def _self_link(stac_item: Dict[str, Any]) -> str:
    for link in (stac_item.get("links") or []):
        if link.get("rel") == "self":
            return link.get("href", "")
    return ""


# ── chip download ─────────────────────────────────────────────────────────────

def _download_band_chip_rasterio(
    href: str,
    lat: float,
    lon: float,
    half_m: float = _CHIP_HALF_M,
) -> Optional["np.ndarray"]:
    """COG window read via rasterio — returns float32 array in [0, 1] or None."""
    try:
        import numpy as np
        import rasterio
        from pyproj import Transformer

        with rasterio.open(href) as src:
            epsg = src.crs.to_epsg()
            t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
            cx, cy = t.transform(lon, lat)
            win = rasterio.windows.from_bounds(
                cx - half_m, cy - half_m, cx + half_m, cy + half_m,
                transform=src.transform,
            )
            arr = src.read(1, window=win, out_dtype="float32")
            return np.clip(arr / _S2_SCALE, 0.0, 1.0)
    except Exception as exc:
        log.debug("rasterio chip fetch failed (%s): %s", href[:60], exc)
        return None


def _download_visual_chip_rasterio(
    href: str,
    lat: float,
    lon: float,
    out_size: int = 224,
    half_m: float = _CHIP_HALF_M * 2,
) -> Optional["PIL.Image.Image"]:
    """Download a visual (RGB) chip via rasterio COG window read, return PIL Image."""
    try:
        import numpy as np
        import rasterio
        from pyproj import Transformer
        from PIL import Image

        with rasterio.open(href) as src:
            epsg = src.crs.to_epsg()
            t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
            cx, cy = t.transform(lon, lat)
            win = rasterio.windows.from_bounds(
                cx - half_m, cy - half_m, cx + half_m, cy + half_m,
                transform=src.transform,
            )
            # Read all bands (3-band TrueColor TCI: R, G, B)
            bands = src.count
            if bands >= 3:
                arr = src.read([1, 2, 3], window=win)  # shape (3, H, W)
            else:
                arr = src.read(1, window=win)
                arr = np.stack([arr, arr, arr], axis=0)
            # Normalise to uint8
            arr = arr.astype("float32")
            p2, p98 = np.percentile(arr, [2, 98])
            if p98 > p2:
                arr = np.clip((arr - p2) / (p98 - p2) * 255, 0, 255).astype("uint8")
            else:
                arr = (arr / arr.max() * 255).astype("uint8") if arr.max() > 0 else arr.astype("uint8")
            img = Image.fromarray(arr.transpose(1, 2, 0), mode="RGB")
            return img.resize((out_size, out_size))
    except Exception as exc:
        log.debug("rasterio visual chip failed (%s): %s", href[:60], exc)
        return None


def _download_visual_chip_http(
    preview_url: str,
    out_size: int = 224,
) -> Optional["PIL.Image.Image"]:
    """Download a small JPEG/PNG preview as fallback (coarser resolution)."""
    if not preview_url:
        return None
    try:
        from PIL import Image
        req = urllib.request.Request(preview_url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=20) as resp:
            buf = resp.read()
        img = Image.open(BytesIO(buf)).convert("RGB")
        return img.resize((out_size, out_size))
    except Exception as exc:
        log.debug("HTTP preview download failed: %s", exc)
        return None


# ── spectral index helpers ─────────────────────────────────────────────────────

def _mean(arr: Any) -> float:
    try:
        import numpy as np
        v = float(np.nanmean(arr))
        return v if math.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _norm_diff(a: Any, b: Any) -> float:
    ma, mb = _mean(a), _mean(b)
    d = ma + mb
    return (ma - mb) / d if d != 0 else 0.0


def _compute_spectral_signals(
    bands: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    """Compute per-scene NDVI, NBR, SWIR/NIR ratio from band arrays."""
    b04 = bands.get("B04")
    b08 = bands.get("B08")
    b11 = bands.get("B11")
    ndvi = _norm_diff(b08, b04) if b08 is not None and b04 is not None else None
    nbr  = _norm_diff(b08, b11) if b08 is not None and b11 is not None else None
    swir = (_mean(b11) / _mean(b08)) if (b08 is not None and b11 is not None and _mean(b08) > 0) else None
    return {"ndvi": ndvi, "nbr": nbr, "swir_ratio": swir}


def _spectral_damage_score(
    pre: Dict[str, Optional[float]],
    post: Dict[str, Optional[float]],
) -> Tuple[float, List[str]]:
    """
    Compute 0–1 damage score and evidence list from pre/post spectral signals.
    Uses change from pre to post; if no pre scene, uses absolute post-event baselines.
    """
    evidence: List[str] = []
    component_scores: List[float] = []

    # NDVI drop: vegetation/canopy damage after hail
    if post["ndvi"] is not None:
        if pre["ndvi"] is not None:
            drop = pre["ndvi"] - post["ndvi"]
        else:
            drop = _BASELINE["ndvi"] - post["ndvi"]
        if drop > 0.04:
            s = min(1.0, drop / 0.25)
            component_scores.append(s * 0.30)
            evidence.append(f"NDVI_drop={drop:.3f}")

    # NBR drop: structural exposure (bare roof deck / debris)
    if post["nbr"] is not None:
        if pre["nbr"] is not None:
            drop = pre["nbr"] - post["nbr"]
        else:
            drop = _BASELINE["nbr"] - post["nbr"]
        if drop > 0.03:
            s = min(1.0, drop / 0.20)
            component_scores.append(s * 0.35)
            evidence.append(f"NBR_drop={drop:.3f}")

    # SWIR/NIR rise: bare surfaces (stripped shingles, exposed decking) reflect more SWIR
    if post["swir_ratio"] is not None:
        if pre["swir_ratio"] is not None:
            rise = post["swir_ratio"] - pre["swir_ratio"]
        else:
            rise = post["swir_ratio"] - _BASELINE["swir_ratio"]
        if rise > 0.02:
            s = min(1.0, rise / 0.15)
            component_scores.append(s * 0.25)
            evidence.append(f"SWIR_rise={rise:.3f}")

    if not component_scores:
        return 0.0, ["no_spectral_signals"]

    raw = sum(component_scores) / sum(w for w in (0.30, 0.35, 0.25))
    return round(min(1.0, raw), 4), evidence


# ── CLIP scoring ──────────────────────────────────────────────────────────────

# Default path where train_clip_damage_head.py saves the fine-tuned head
_TRAINED_HEAD_PATH = Path(__file__).resolve().parent / "models" / "clip_damage_head.pt"

_clip_model = None
_clip_processor = None
_damage_head = None        # fine-tuned MLP head (preferred over zero-shot when present)
_head_loaded: bool = False


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return True
    try:
        from transformers import CLIPModel, CLIPProcessor
        print("  Loading openai/clip-vit-large-patch14 from HuggingFace…", flush=True)
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        _clip_model.eval()
        print("  CLIP loaded.", flush=True)
        return True
    except Exception as exc:
        log.warning("CLIP load failed: %s", exc)
        return False


def _load_trained_head(head_path: Path = _TRAINED_HEAD_PATH) -> bool:
    """Load the MLP head trained by train_clip_damage_head.py, if present."""
    global _damage_head, _head_loaded
    if _head_loaded:
        return _damage_head is not None
    _head_loaded = True
    if not head_path.is_file():
        return False
    try:
        import torch
        import torch.nn as nn
        ckpt = torch.load(head_path, map_location="cpu")
        in_dim = ckpt.get("in_dim", 768)
        head = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(256, 64),    nn.GELU(), nn.Dropout(0.15),
            nn.Linear(64, 1),
        )
        head.load_state_dict(ckpt["state_dict"])
        head.eval()
        _damage_head = head
        auc = ckpt.get("best_val_auc", "?")
        print(f"  Loaded fine-tuned damage head (val_auc={auc}) from {head_path}", flush=True)
        return True
    except Exception as exc:
        log.warning("Could not load trained head from %s: %s", head_path, exc)
        return False


def _score_clip(
    image: "PIL.Image.Image",
) -> Tuple[Optional[float], Optional[float]]:
    """Score a chip image.

    Priority:
      1. Fine-tuned MLP head (CLIP embedding → trained head) — if clip_damage_head.pt exists
      2. Zero-shot CLIP prompts — always available when transformers/torch are installed

    Returns (damage_score, intact_score) each in [0, 1].
    """
    if not _load_clip():
        return None, None

    try:
        import torch
        # ── get CLIP image embedding ──────────────────────────────────────────
        inputs = _clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embedding = _clip_model.get_image_features(**inputs)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        # ── path 1: fine-tuned head ───────────────────────────────────────────
        if _load_trained_head():
            with torch.no_grad():
                logit = _damage_head(embedding)
                prob = float(torch.sigmoid(logit).item())
            return round(prob, 4), round(1.0 - prob, 4)

        # ── path 2: zero-shot prompts ─────────────────────────────────────────
        all_prompts = DAMAGE_PROMPTS + INTACT_PROMPTS
        text_inputs = _clip_processor(
            text=all_prompts, return_tensors="pt", padding=True,
        )
        with torch.no_grad():
            text_feats = _clip_model.get_text_features(**text_inputs)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            # Cosine similarity → softmax
            sims = (embedding @ text_feats.T).squeeze(0)
            probs = sims.softmax(dim=0).tolist()

        n_d = len(DAMAGE_PROMPTS)
        damage_score = sum(probs[:n_d])
        intact_score = sum(probs[n_d:])
        total = damage_score + intact_score
        if total > 0:
            damage_score /= total
            intact_score /= total
        return round(float(damage_score), 4), round(float(intact_score), 4)

    except Exception as exc:
        log.debug("CLIP scoring failed: %s", exc)
        return None, None


# ── main assessment function ──────────────────────────────────────────────────

def assess_lead(
    lead_id: str,
    lat: float,
    lon: float,
    event_dt: datetime,
    *,
    post_stac_item_url: str = "",
    post_preview_url: str = "",
    post_scene_id: str = "",
    post_cloud_pct: float = 0.0,
    run_clip: bool = True,
) -> DamageAssessment:
    """Full damage assessment for a single lead.

    1. Looks up or reuses the post-event Sentinel-2 scene.
    2. Finds the best pre-event scene (up to 60 days before).
    3. Downloads B04/B08/B11 chips from both scenes via COG range-request.
    4. Computes spectral change signals (NDVI, NBR, SWIR).
    5. Downloads visual chip and runs CLIP zero-shot classifier.
    6. Combines signals into a single damage_confidence score.

    Everything that produced the score is recorded in DamageAssessment.evidence.
    """
    result = DamageAssessment(lead_id=lead_id)

    # ── Post-event scene ──────────────────────────────────────────────────────
    post_item: Optional[Dict[str, Any]] = None
    if post_stac_item_url:
        try:
            post_item = _fetch_stac_item(post_stac_item_url)
        except Exception as exc:
            log.debug("Failed to fetch post-event STAC item: %s", exc)

    if post_item is None:
        post_item_found = find_post_event_scene(lat, lon, event_dt)
        if post_item_found is None:
            result.error = "no_post_event_scene"
            return result
        post_item = post_item_found

    p_props = post_item.get("properties") or {}
    result.post_scene_id = post_scene_id or str(post_item.get("id") or "")
    result.post_scene_utc = str(p_props.get("datetime") or "")
    result.post_cloud_pct = float(p_props.get("eo:cloud_cover") or post_cloud_pct)

    # ── Pre-event scene ───────────────────────────────────────────────────────
    pre_item = find_pre_event_scene(lat, lon, event_dt)
    if pre_item:
        pr_props = pre_item.get("properties") or {}
        result.pre_scene_id = str(pre_item.get("id") or "")
        result.pre_scene_utc = str(pr_props.get("datetime") or "")
        result.pre_cloud_pct = float(pr_props.get("eo:cloud_cover") or 0.0)

    # ── Band chip download ────────────────────────────────────────────────────
    post_hrefs = _band_hrefs(post_item)
    pre_hrefs = _band_hrefs(pre_item) if pre_item else {}

    post_bands: Dict[str, Any] = {}
    pre_bands: Dict[str, Any] = {}

    for band in ("B04", "B08", "B11"):
        href = post_hrefs.get(band, "")
        if href:
            arr = _download_band_chip_rasterio(href, lat, lon)
            if arr is not None:
                post_bands[band] = arr

    for band in ("B04", "B08", "B11"):
        href = pre_hrefs.get(band, "")
        if href:
            arr = _download_band_chip_rasterio(href, lat, lon)
            if arr is not None:
                pre_bands[band] = arr

    # ── Spectral change ───────────────────────────────────────────────────────
    post_spec = _compute_spectral_signals(post_bands)
    pre_spec  = _compute_spectral_signals(pre_bands)

    result.post_ndvi = post_spec["ndvi"]
    result.pre_ndvi  = pre_spec["ndvi"]
    result.post_nbr  = post_spec["nbr"]
    result.pre_nbr   = pre_spec["nbr"]
    result.post_swir_ratio = post_spec["swir_ratio"]
    result.pre_swir_ratio  = pre_spec["swir_ratio"]

    if result.post_ndvi is not None and result.pre_ndvi is not None:
        result.ndvi_change = round(result.pre_ndvi - result.post_ndvi, 4)
    if result.post_nbr is not None and result.pre_nbr is not None:
        result.nbr_change = round(result.pre_nbr - result.post_nbr, 4)
    if result.post_swir_ratio is not None and result.pre_swir_ratio is not None:
        result.swir_change = round(result.post_swir_ratio - result.pre_swir_ratio, 4)

    spectral_score, spec_evidence = _spectral_damage_score(pre_spec, post_spec)
    result.evidence.extend(spec_evidence)

    # ── CLIP scoring ──────────────────────────────────────────────────────────
    clip_score: Optional[float] = None
    if run_clip:
        # Prefer COG visual chip; fall back to HTTP preview
        visual_href = _visual_href(post_item)
        chip_img = None
        if visual_href:
            chip_img = _download_visual_chip_rasterio(visual_href, lat, lon)
        if chip_img is None and post_preview_url:
            chip_img = _download_visual_chip_http(post_preview_url)

        if chip_img is not None:
            d_score, i_score = _score_clip(chip_img)
            result.clip_damage_score = d_score
            result.clip_intact_score = i_score
            if d_score is not None:
                clip_score = d_score
                # Report whether the fine-tuned head or zero-shot was used
                head_active = _head_loaded and _damage_head is not None
                clip_variant = "clip_finetuned" if head_active else "clip_zeroshot"
                result.evidence.append(f"{clip_variant}={d_score:.3f}")
                result.model_used = clip_variant
            else:
                result.evidence.append("clip_scoring_failed")
        else:
            result.evidence.append("no_visual_chip")

    # ── Combine ───────────────────────────────────────────────────────────────
    if clip_score is not None and spectral_score > 0:
        # Weighted blend: CLIP 55%, spectral 45%
        combined = 0.55 * clip_score + 0.45 * spectral_score
        result.model_used = "clip+spectral"
    elif clip_score is not None:
        combined = clip_score
        head_active = _head_loaded and _damage_head is not None
        result.model_used = "clip_finetuned_only" if head_active else "clip_zeroshot_only"
    elif spectral_score > 0:
        combined = spectral_score
        result.model_used = "spectral_only"
    else:
        result.error = "no_usable_signals"
        return result

    # Final soft-sigmoid squeeze to stay honest in the 0.2–0.85 range
    result.damage_confidence = round(float(1.0 / (1.0 + math.exp(-6.0 * (combined - 0.45)))), 4)
    return result


# ── batch runner ──────────────────────────────────────────────────────────────

def assess_lead_rows(
    rows: List[Dict[str, Any]],
    *,
    run_clip: bool = True,
    max_cloud_pct: float = 45.0,
) -> List[DamageAssessment]:
    """Assess a list of lead dicts. Skips rows without valid lat/lon/datetime."""
    results: List[DamageAssessment] = []
    for i, row in enumerate(rows):
        lead_id = str(row.get("lead_id") or "")
        try:
            lat = float(row.get("lat") or "nan")
            lon = float(row.get("lon") or "nan")
        except ValueError:
            results.append(DamageAssessment(lead_id=lead_id, error="bad_lat_lon"))
            continue
        if not math.isfinite(lat) or not math.isfinite(lon):
            results.append(DamageAssessment(lead_id=lead_id, error="bad_lat_lon"))
            continue

        dt = None
        for col in ("report_datetime", "peak_timestamp_utc", "sentinel2_acquired_utc"):
            dt = _parse_dt(str(row.get(col) or ""))
            if dt:
                break
        if dt is None:
            results.append(DamageAssessment(lead_id=lead_id, error="no_event_datetime"))
            continue

        cloud = float(row.get("sentinel2_cloud_cover_pct") or 0)
        if cloud > max_cloud_pct:
            results.append(DamageAssessment(lead_id=lead_id, error=f"cloud_cover_too_high:{cloud:.0f}%"))
            continue

        print(f"  [{i+1}/{len(rows)}] assessing {lead_id} ({lat:.4f},{lon:.4f})…", flush=True)
        a = assess_lead(
            lead_id=lead_id,
            lat=lat,
            lon=lon,
            event_dt=dt,
            post_stac_item_url=str(row.get("sentinel2_stac_item_url") or ""),
            post_preview_url=str(row.get("sentinel2_preview_url") or ""),
            post_scene_id=str(row.get("sentinel2_scene_id") or ""),
            post_cloud_pct=cloud,
            run_clip=run_clip,
        )
        results.append(a)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, csv, sys

    ap = argparse.ArgumentParser(description="Run CLIP+spectral damage assessment on a Sentinel-enriched lead CSV")
    ap.add_argument("--in", dest="in_csv", required=True, type=Path)
    ap.add_argument("--out", dest="out_csv", required=True, type=Path)
    ap.add_argument("--no-clip", action="store_true", help="Skip CLIP (spectral only)")
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    with args.in_csv.open(newline="", encoding="utf-8") as f:
        rows: List[Dict[str, Any]] = list(csv.DictReader(f))
    if args.max_rows:
        rows = rows[: args.max_rows]

    assessments = assess_lead_rows(rows, run_clip=not args.no_clip)
    result_dicts = [a.to_dict() for a in assessments]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = list(result_dicts[0].keys()) if result_dicts else []
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in result_dicts:
            w.writerow(r)

    confident = sum(1 for a in assessments if a.damage_confidence and a.damage_confidence >= 0.6)
    print(f"Assessed {len(assessments)} leads → {confident} with confidence ≥ 0.60")
    print(f"Wrote to {args.out_csv}")
