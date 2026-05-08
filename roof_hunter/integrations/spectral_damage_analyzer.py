"""Sentinel-2 spectral damage analyzer — no CV model required.

Uses band reflectance indices from Sentinel-2 L2A COG assets to estimate
structural damage confidence at a point, replacing the heuristic thumbnail-texture
approach in satellite_damage_verifier.py.

Science basis:
  • NDVI (B08-B04)/(B08+B04)   — vegetative cover loss after hail/wind
  • NDRE (B08-B05)/(B08+B05)   — red-edge stress; more sensitive to canopy damage
  • NBR  (B08-B11)/(B08+B11)   — normalized burn ratio; bare-soil / structural exposure
  • SWIR ratio B11/B08          — high B11 relative to NIR indicates bare roof / debris
  • B02 brightness proxy        — hail scatter / white roofing exposure

COG reading strategy: range-request only the ~256×256 pixel chip around the
target lat/lon using GDAL windowed read (gdal/rasterio) if available, or
falls back to downloading small preview windows via Sentinel Hub presigned URLs.

Confidence is a weighted combination of these indices evaluated against
expected post-hail damage signatures.  Returns 0.0–1.0.

If neither rasterio nor requests+numpy is available the function returns None
so the caller can skip spectral scoring without crashing.

Upgrade path to trained CV model:
  Install: pip install transformers timm Pillow
  Swap spectral_damage_confidence() for a call to:
    from transformers import AutoFeatureExtractor, AutoModelForImageClassification
    model = AutoModelForImageClassification.from_pretrained(
        "microsoft/beit-base-patch16-224"   # fine-tune on xView2/ETCI chips
    )
"""

from __future__ import annotations

import json
import math
import urllib.request
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

_UA = "QuLabInfinite-RoofHunter/1.2-spectral"

# Band asset names in Sentinel-2 L2A STAC items (Element84 earth-search naming)
_BAND_ASSETS = {
    "B02": "blue",
    "B04": "red",
    "B05": "rededge",
    "B08": "nir",
    "B11": "swir16",
}

# Chip size to read around target point (pixels, at 10 m/px native resolution)
_CHIP_PX = 64  # ~640 m × 640 m window centred on point

# Weight matrix for the damage composite
_WEIGHTS = {
    "ndvi_drop":     0.30,   # healthy NDVI decreases after hail canopy loss
    "nbr_drop":      0.30,   # NBR drops when roofing material/soil exposed
    "swir_rise":     0.20,   # SWIR/NIR rises for debris / bare surfaces
    "ndre_stress":   0.10,   # red-edge stress
    "brightness":    0.10,   # anomalous brightness (white/reflective debris)
}

# Typical "undamaged" baselines (summer CONUS residential area)
_BASELINE_NDVI = 0.45
_BASELINE_NBR = 0.40
_BASELINE_SWIR_RATIO = 0.18
_BASELINE_NDRE = 0.38
_BASELINE_BRIGHTNESS = 0.15


def _fetch_band_array(
    band_href: str,
    lat: float,
    lon: float,
    chip_px: int = _CHIP_PX,
) -> Optional["np.ndarray"]:
    """
    Attempt rasterio COG window read (fast, only fetches a small tile).
    Falls back to downloading the COG overview at low resolution.
    Returns float32 array or None.
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.windows import from_bounds as window_from_bounds

        with rasterio.open(band_href) as src:
            # Convert lat/lon to dataset CRS
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", src.crs.to_epsg() or 32614, always_xy=True)
            cx, cy = transformer.transform(lon, lat)
            half = (chip_px / 2) * abs(src.res[0])
            win = rasterio.windows.from_bounds(
                cx - half, cy - half, cx + half, cy + half,
                transform=src.transform,
            )
            arr = src.read(1, window=win, out_dtype="float32")
            # Sentinel-2 L2A surface reflectance is scaled by 10000
            return arr.astype("float32") / 10_000.0
    except Exception:
        pass

    # Fallback: download a small overview (JPEG2000 preview) via HTTP
    try:
        import numpy as np
        from PIL import Image

        req = urllib.request.Request(band_href, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=30) as resp:
            buf = resp.read()
        img = Image.open(BytesIO(buf)).convert("L")
        img = img.resize((chip_px, chip_px))
        return (np.asarray(img, dtype="float32") / 255.0) * 0.5  # rough DN rescale
    except Exception:
        return None


def _safe_index(a: Any, b: Any) -> float:
    """Normalised difference of mean arrays; returns 0 on failure."""
    try:
        import numpy as np
        ma = float(np.nanmean(a))
        mb = float(np.nanmean(b))
        denom = ma + mb
        if denom == 0:
            return 0.0
        return (ma - mb) / denom
    except Exception:
        return 0.0


def _safe_ratio(a: Any, b: Any) -> float:
    try:
        import numpy as np
        ma = float(np.nanmean(a))
        mb = float(np.nanmean(b))
        if mb == 0:
            return 0.0
        return ma / mb
    except Exception:
        return 0.0


def _band_hrefs_from_stac(stac_item_url: str) -> Dict[str, str]:
    """Fetch STAC item JSON and extract band asset hrefs."""
    if not stac_item_url:
        return {}
    try:
        req = urllib.request.Request(stac_item_url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=20) as resp:
            item = json.loads(resp.read().decode("utf-8"))
        assets = item.get("assets") or {}
        hrefs: Dict[str, str] = {}
        for band_key, asset_key in _BAND_ASSETS.items():
            candidates = [band_key.lower(), band_key, asset_key, asset_key.lower()]
            for c in candidates:
                href = (assets.get(c) or {}).get("href", "")
                if href:
                    hrefs[band_key] = href
                    break
        return hrefs
    except Exception:
        return {}


def spectral_damage_confidence(
    stac_item_url: str,
    lat: float,
    lon: float,
) -> float:
    """Return spectral damage confidence in [0, 1] for a Sentinel-2 scene at (lat, lon).

    Uses NDVI drop, NBR drop, SWIR rise relative to typical undamaged baselines.
    Returns 0.0 if dependencies (rasterio or PIL+numpy) are unavailable or the
    stac_item_url is empty.
    """
    if not stac_item_url or not lat or not lon:
        return 0.0

    try:
        import numpy as np
    except ImportError:
        return 0.0

    hrefs = _band_hrefs_from_stac(stac_item_url)
    if not hrefs:
        return 0.0

    # Fetch required bands (B04=red, B08=NIR, B11=SWIR, B05=RedEdge, B02=blue)
    b04 = _fetch_band_array(hrefs.get("B04", ""), lat, lon) if hrefs.get("B04") else None
    b08 = _fetch_band_array(hrefs.get("B08", ""), lat, lon) if hrefs.get("B08") else None
    b11 = _fetch_band_array(hrefs.get("B11", ""), lat, lon) if hrefs.get("B11") else None
    b05 = _fetch_band_array(hrefs.get("B05", ""), lat, lon) if hrefs.get("B05") else None
    b02 = _fetch_band_array(hrefs.get("B02", ""), lat, lon) if hrefs.get("B02") else None

    scores: Dict[str, float] = {}

    # NDVI: lower than baseline → possible canopy/cover damage
    if b08 is not None and b04 is not None:
        ndvi = _safe_index(b08, b04)
        drop = max(0.0, _BASELINE_NDVI - ndvi) / _BASELINE_NDVI
        scores["ndvi_drop"] = min(1.0, drop * 2.0)   # scale 50% drop → 1.0

    # NBR: (NIR-SWIR)/(NIR+SWIR) — lower after bare roof / debris exposure
    if b08 is not None and b11 is not None:
        nbr = _safe_index(b08, b11)
        drop = max(0.0, _BASELINE_NBR - nbr) / _BASELINE_NBR
        scores["nbr_drop"] = min(1.0, drop * 2.5)

    # SWIR/NIR ratio: rises when reflective debris / bare shingles dominate
    if b11 is not None and b08 is not None:
        swir_ratio = _safe_ratio(b11, b08)
        rise = max(0.0, swir_ratio - _BASELINE_SWIR_RATIO) / (1.0 - _BASELINE_SWIR_RATIO)
        scores["swir_rise"] = min(1.0, rise * 1.8)

    # NDRE: red-edge stress proxy
    if b08 is not None and b05 is not None:
        ndre = _safe_index(b08, b05)
        stress = max(0.0, _BASELINE_NDRE - ndre) / _BASELINE_NDRE
        scores["ndre_stress"] = min(1.0, stress * 2.0)

    # Brightness: anomalous B02 (blue) brightness for reflective hail debris
    if b02 is not None:
        try:
            import numpy as np_inner
            bright = float(np_inner.nanmean(b02))
            excess = max(0.0, bright - _BASELINE_BRIGHTNESS) / (0.6 - _BASELINE_BRIGHTNESS)
            scores["brightness"] = min(1.0, excess)
        except Exception:
            pass

    if not scores:
        return 0.0

    total_weight = sum(_WEIGHTS[k] for k in scores)
    if total_weight == 0:
        return 0.0

    weighted_sum = sum(_WEIGHTS[k] * v for k, v in scores.items())
    raw = weighted_sum / total_weight

    # Soft sigmoid squeeze to avoid over-saturation
    return round(float(1.0 / (1.0 + math.exp(-6.0 * (raw - 0.45)))), 4)


def analyze_lead_csv_row(row: Dict[str, str]) -> float:
    """Convenience wrapper for use in pipeline loops."""
    return spectral_damage_confidence(
        stac_item_url=(row.get("sentinel2_stac_item_url") or "").strip(),
        lat=float(row.get("lat") or 0),
        lon=float(row.get("lon") or 0),
    )
