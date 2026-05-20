"""OpenStreetMap Nominatim reverse-geocode helpers (nearest address hint near report lat/lon).

Use per https://operations.osmfoundation.org/policies/nominatim/ : identify the application,
stay at ~1 request/second, and cache aggressively.

**Imprecision sources (historical / mitigated):**
- Rounding coordinates before the HTTP request (~1.1 m at 5 decimals) biased results toward
  a grid cell center; we now query with full-precision lat/lon while keeping a rounded cache key.
- Low Nominatim ``zoom`` returns coarser administrative/road context; we default to zoom 18 and
  optionally retry at 19 when no ``house_number`` is returned.
- OSM reverse returns the *nearest* mapped feature to the point, not a parcel or "worst-hit"
  structure; storm lat/lon may sit on a road centroid, park, or building footprint without
  rooftop-level addressing. True property-at-risk usually needs parcel/CAD boundaries.

**Optional heavy approaches (not implemented):**
- Bearing-based interpolation along the nearest road segment from the storm centroid.
- Parcel / rooftop APIs (typically licensed).

``addressdetails=1`` is passed explicitly (jsonv2 includes structured ``address``, but this keeps
intent clear for maintainers).

A cache miss may perform **up to two** throttled HTTP calls (~2 s worst case) when
``try_refine_missing_house_number`` is enabled, to chase a finer zoom level without exploding QPS.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_USER_AGENT = "QuLabInfinite-RoofHunter/1.0 (parcel-prep; respectful use of Nominatim)"
_MIN_INTERVAL_SEC = 1.05
_last_call_lock = threading.Lock()
_last_call_monotonic = 0.0


def _throttle() -> None:
    global _last_call_monotonic
    with _last_call_lock:
        now = time.monotonic()
        wait = _MIN_INTERVAL_SEC - (now - _last_call_monotonic)
        if wait > 0:
            time.sleep(wait)
        _last_call_monotonic = time.monotonic()


def _fetch_reverse_json(
    lat: float,
    lon: float,
    *,
    zoom: int,
    timeout_sec: float = 20.0,
) -> Dict[str, Any]:
    qs = urllib.parse.urlencode(
        {
            "lat": lat,
            "lon": lon,
            "format": "jsonv2",
            "zoom": zoom,
            "addressdetails": "1",
        }
    )
    url = f"https://nominatim.openstreetmap.org/reverse?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    _throttle()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return {"error": str(e.code), "display_name": ""}
    except urllib.error.URLError as e:
        return {"error": str(e.reason), "display_name": ""}
    except Exception as exc:  # noqa: BLE001 — surface to caller column
        return {"error": str(exc), "display_name": ""}

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "invalid_json", "display_name": ""}

    if payload.get("error"):
        return {"error": payload.get("error"), "display_name": ""}
    return payload


def _normalize_house_number(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    return str(raw).strip()


def _road_from_address(addr: Dict[str, Any]) -> str:
    return (
        addr.get("road")
        or addr.get("pedestrian")
        or addr.get("path")
        or addr.get("footway")
        or addr.get("residential")
        or ""
    )


def _locality_from_address(addr: Dict[str, Any]) -> str:
    return (
        addr.get("hamlet")
        or addr.get("village")
        or addr.get("town")
        or addr.get("city")
        or addr.get("municipality")
        or ""
    )


def _poi_keys_from_address(addr: Dict[str, Any]) -> List[str]:
    keys = (
        "amenity",
        "shop",
        "tourism",
        "historic",
        "leisure",
        "building",
        "man_made",
        "office",
    )
    out: List[str] = []
    for k in keys:
        v = addr.get(k)
        if v:
            out.append(str(v))
    return out


def _classify_precision_and_note(
    payload: Dict[str, Any],
    *,
    house_number: str,
    road: str,
    locality: str,
) -> Tuple[str, str]:
    """Return (reverse_geocode_precision, reverse_geocode_confidence_note)."""

    addr = payload.get("address") or {}
    clas = str(payload.get("class") or "")
    typ = str(payload.get("type") or "")
    name = str(payload.get("name") or "")
    poi_addr = _poi_keys_from_address(addr)

    has_hn = bool(house_number and house_number.strip())
    has_road = bool(road and road.strip())
    has_loc = bool(locality and locality.strip())

    if has_hn and has_road:
        return "house", ""

    if has_hn and not has_road:
        return (
            "road",
            "OSM returned a house or unit reference without an adjacent mapped road name; "
            "refine manually or via parcel GIS.",
        )

    # Building-like mapped feature (centroid may be POI/roof — still not a validated parcel).
    poi_like_class = clas in {"building", "amenity", "shop", "tourism", "historic", "office", "man_made"}
    if poi_like_class or poi_addr:
        cn = (
            "OSM resolved to a mapped building or POI centroid without a street house number "
            "(or digits are absent in OSM at this coordinate). Confirm with parcel GIS or imagery."
        )
        return "building", cn

    if has_road or clas == "highway":
        cn = (
            "OSM nearest address lacks a distinct house_number at this storm coordinate; "
            "likely snapped along a road or path. Use parcel search or street-level review for "
            "the impacted structure."
        )
        return "road", cn

    if has_loc or clas == "boundary" or typ in {"administrative", "postcode"}:
        cn = (
            "Reverse geocode is locality- or administrative-level only at this coordinate; "
            "not building-specific."
        )
        return "locality", cn

    if name:
        cn = (
            "OSM returned a named place without standard street-address components at this coordinate; "
            "treat as a weak hint for manual lookup."
        )
        return "place", cn

    return (
        "unknown",
        "Could not confidently classify reverse-geocode granularity; refine with parcel layers or aerial context.",
    )


def _enrich_payload_to_record(
    payload: Dict[str, Any],
    *,
    zoom_used: int,
    refinement_note_parts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the flat integration record exported to CSV callers."""

    if payload.get("error"):
        return {
            "display_name": payload.get("display_name") or "",
            "road": "",
            "house_number": "",
            "city": "",
            "state": "",
            "postcode": "",
            "county_osm": "",
            "suggested_search_address": "",
            "amenity_near": "",
            "reverse_geocode_precision": "",
            "reverse_geocode_confidence_note": "",
            "nominatim_zoom_used": zoom_used,
            "error": payload.get("error"),
        }

    addr = payload.get("address") or {}
    display = payload.get("display_name") or ""
    road = _road_from_address(addr)
    hn_st = _normalize_house_number(addr.get("house_number"))
    locality = _locality_from_address(addr)

    line_parts: List[str] = []
    if hn_st and road:
        line_parts.append(f"{hn_st} {road}".strip())
    elif road:
        line_parts.append(road)
    elif hn_st:
        line_parts.append(hn_st)

    suggested = ""
    if line_parts:
        tail = locality or ""
        if tail:
            suggested = f"{line_parts[0]}, {tail}".strip(", ")
        else:
            suggested = line_parts[0]

    amenity_near = ""
    poi_bits = []
    poi_bits.extend(_poi_keys_from_address(addr))
    if payload.get("name"):
        poi_bits.append(str(payload.get("name")))
    if poi_bits:
        amenity_near = "; ".join(dict.fromkeys(poi_bits))

    precision, conf = _classify_precision_and_note(
        payload,
        house_number=hn_st,
        road=road,
        locality=locality,
    )
    refin = refinement_note_parts or []
    extra_bits = [x for x in refin if x]
    if extra_bits:
        sep = " " if conf else ""
        conf = f"{conf}{sep}{' '.join(extra_bits)}".strip()

    out: Dict[str, Any] = {
        "display_name": display,
        "road": road or "",
        "house_number": hn_st or "",
        "city": locality or "",
        "state": addr.get("state") or "",
        "postcode": addr.get("postcode") or "",
        "county_osm": addr.get("county") or "",
        "suggested_search_address": suggested,
        "amenity_near": amenity_near,
        "reverse_geocode_precision": precision,
        "reverse_geocode_confidence_note": conf if conf else "",
        "nominatim_zoom_used": zoom_used,
    }
    return out


def reverse_geocode(
    lat: float,
    lon: float,
    *,
    zoom: int = 18,
    timeout_sec: float = 20.0,
    try_refine_missing_house_number: bool = True,
) -> Dict[str, Any]:
    """Return flattened Nominatim reverse-geocode fields for ``lat``, ``lon`` (full precision).

    When ``try_refine_missing_house_number`` is True and the first zoom=18 response has no
    ``house_number`` but no fatal error, a second request at zoom=19 is attempted. Trade-offs:
    zoom 19 is more granular and can attach a rooftop-level address digit when OSM encodes it;
    it can also return a less street-oriented object (fine POI polygons), so callers should
    read ``reverse_geocode_precision`` and confidence notes."""

    zoom_primary = zoom
    p1 = _fetch_reverse_json(lat, lon, zoom=zoom_primary, timeout_sec=timeout_sec)

    refinement_parts: List[str] = []

    err1 = p1.get("error")
    if err1:
        return _enrich_payload_to_record(p1, zoom_used=zoom_primary, refinement_note_parts=refinement_parts)

    addr1 = p1.get("address") or {}
    hn1 = _normalize_house_number(addr1.get("house_number"))

    if try_refine_missing_house_number and not hn1 and zoom_primary != 19:
        p2 = _fetch_reverse_json(lat, lon, zoom=19, timeout_sec=timeout_sec)
        if p2.get("error"):
            refinement_parts.append("Zoom 19 retry failed or errored; using primary zoom response.")
        else:
            addr2 = p2.get("address") or {}
            hn2 = _normalize_house_number(addr2.get("house_number"))
            if hn2:
                refinement_parts.append("House number surfaced only after Nominatim zoom 19 retry.")
                return _enrich_payload_to_record(
                    p2,
                    zoom_used=19,
                    refinement_note_parts=refinement_parts,
                )
            chosen = _pick_more_specific_payload(p1, p2)
            zw = 19 if chosen is p2 else zoom_primary
            if chosen is p2 and chosen is not p1:
                refinement_parts.append("Used finer Nominatim zoom (19) for more specific mapped context.")
            return _enrich_payload_to_record(chosen, zoom_used=zw, refinement_note_parts=refinement_parts)

    return _enrich_payload_to_record(p1, zoom_used=zoom_primary, refinement_note_parts=refinement_parts)


def _pick_more_specific_payload(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """If neither payload clearly wins by house_number, prefer the one with finer address parts."""

    def score(p: Dict[str, Any]) -> int:
        addr = p.get("address") or {}
        s = 0
        if addr.get("house_number"):
            s += 40
        if _road_from_address(addr):
            s += 10
        if addr.get("postcode"):
            s += 2
        clas = str(p.get("class") or "")
        if clas in {"building", "house"}:
            s += 15
        if p.get("name"):
            s += 3
        return s

    return a if score(a) >= score(b) else b


def load_json_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): v for k, v in data.items() if isinstance(v, dict)}
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def save_json_cache(path: Path, mapping: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def rounded_key(lat: float, lon: float) -> Tuple[str, Tuple[float, float]]:
    rl = round(lat, 5)
    rr = round(lon, 5)
    return f"{rl},{rr}", (rl, rr)


def _cache_missing_new_fields(cached: Dict[str, Any]) -> bool:
    """Stale disk entries from older Roof Hunter builds lack precision metadata."""

    return "reverse_geocode_precision" not in cached


_DISK_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
_DISK_CACHE_PATH: Optional[Path] = None


def _ensure_disk_cache(cache_path: Path) -> Dict[str, Dict[str, Any]]:
    global _DISK_CACHE, _DISK_CACHE_PATH
    if _DISK_CACHE is None or _DISK_CACHE_PATH != cache_path:
        _DISK_CACHE = load_json_cache(cache_path)
        _DISK_CACHE_PATH = cache_path
    return _DISK_CACHE


def reverse_geocode_cached(
    lat: float,
    lon: float,
    *,
    memory: Dict[str, Dict[str, Any]],
    cache_path: Optional[Path] = None,
    try_refine_missing_house_number: bool = True,
) -> Dict[str, Any]:
    """Reverse-geocode with disk+memory JSON cache keyed by 5-decimal rounded coordinates.

    The **HTTP reverse lookup uses full-precision** ``lat``/``lon`` so storm coordinates from SPC
    are not pre-quantised; only the cache key shares a ~1.1 m grid."""

    key, _coords = rounded_key(lat, lon)
    if key in memory and not _cache_missing_new_fields(memory[key]):
        return dict(memory[key])
    if cache_path is not None:
        disk = _ensure_disk_cache(cache_path)
        if key in disk and not _cache_missing_new_fields(disk[key]):
            memory[key] = dict(disk[key])
            return dict(memory[key])

    # Full-precision coordinate query; throttle + optional second call happen inside reverse_geocode.
    res = reverse_geocode(
        lat,
        lon,
        try_refine_missing_house_number=try_refine_missing_house_number,
    )
    memory[key] = res
    if cache_path is not None:
        disk = _ensure_disk_cache(cache_path)
        disk[key] = res
        save_json_cache(cache_path, disk)
    return dict(res)
