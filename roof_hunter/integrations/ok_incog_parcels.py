"""Point-in-parcel lookup for select Oklahoma counties via INCOG ArcGIS REST.

Coverage (free public services via INCOG): Creek, Osage, Rogers, Tulsa, Wagoner. Extend
``COUNTY_TO_SERVICE`` when you add more vetted county REST endpoints.

Uses ``outFields=*`` — some INCOG layers error if field lists include invalid names.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Tuple

_INCOG_BASE = "https://map11.incog.org/arcgis11wa/rest/services"

# Uppercase county name as in roof_hunter lead CSVs (no "County" suffix)
COUNTY_TO_SERVICE: Dict[str, str] = {
    "TULSA": f"{_INCOG_BASE}/Parcels_TulsaCo/FeatureServer/0/query",
    "ROGERS": f"{_INCOG_BASE}/Parcels_RogersCo/FeatureServer/0/query",
    "CREEK": f"{_INCOG_BASE}/Parcels_CreekCo/FeatureServer/0/query",
    "WAGONER": f"{_INCOG_BASE}/Parcels_WagonerCo/FeatureServer/0/query",
    "OSAGE": f"{_INCOG_BASE}/Parcels_OsageCo/FeatureServer/0/query",
}


@dataclass
class ParcelLookupResult:
    matched: bool
    parcel_attributes: Dict[str, Any]
    arcgis_error: str
    service_url: str


def normalize_county(county: str) -> str:
    c = (county or "").strip().upper().replace(" COUNTY", "")
    return c


def _http_get_json(url: str, *, timeout: float = 60.0) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "QuLabInfinite-RoofHunter/parcel-enrich"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def query_parcel_point(
    lat: float,
    lon: float,
    county: str,
    *,
    sleep_s: float = 0.0,
) -> ParcelLookupResult:
    """Return parcel feature attributes containing ``lat``/``lon`` or unmatched shell."""
    cty = normalize_county(county)
    svc = COUNTY_TO_SERVICE.get(cty)
    if not svc:
        return ParcelLookupResult(False, {}, "no_incog_layer_for_county", "")
    params = {
        "f": "json",
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "false",
    }
    url = f"{svc}?{urllib.parse.urlencode(params)}"
    try:
        if sleep_s > 0:
            time.sleep(sleep_s)
        data = _http_get_json(url, timeout=90.0)
    except urllib.error.HTTPError as e:
        return ParcelLookupResult(False, {}, f"http_{e.code}", svc)
    except Exception as e:  # noqa: BLE001 — surface to CSV
        return ParcelLookupResult(False, {}, str(e), svc)

    err = data.get("error")
    if err:
        msg = err.get("message", "arcgis_error") if isinstance(err, dict) else str(err)
        return ParcelLookupResult(False, {}, msg, svc)

    feats = data.get("features") or []
    if not feats:
        return ParcelLookupResult(False, {}, "no_parcel_polygon", svc)

    attrs = feats[0].get("attributes") or {}
    return ParcelLookupResult(True, attrs, "", svc)


def flatten_for_batchdata(attrs: Dict[str, Any]) -> Dict[str, str]:
    """Map INCOG assessor fields to generic BatchData / skip-trace style columns."""
    empty = {
        "batch_apn": "",
        "batch_owner_name": "",
        "batch_property_address": "",
        "batch_property_city": "",
        "batch_property_zip": "",
        "batch_mailing_line1": "",
        "batch_mailing_city": "",
        "batch_mailing_state": "",
        "batch_mailing_zip": "",
        "parcel_year_built": "",
        "parcel_total_acct_value": "",
        "parcel_legal": "",
    }
    if not attrs:
        return empty

    # Osage County layer uses Assessor / CAMA field names.
    if attrs.get("CountyFips") == "40113" or ("OwnerName" in attrs and "MailingCty" in attrs):
        apn = attrs.get("ParcelId") or attrs.get("county_id") or ""
        owner = attrs.get("OwnerName") or ""
        site = (attrs.get("AdrLabel") or "").strip()
        if not site:
            parts = [
                str(attrs.get("AdrNum") or "").strip(),
                (attrs.get("PreDir") or "").strip(),
                (attrs.get("PstrNam") or "").strip(),
            ]
            site = " ".join(p for p in parts if p and p != "0").strip()
        city = (attrs.get("AdrCity") or "").strip()
        zraw = attrs.get("AdrZip5")
        zip_s = str(zraw) if zraw not in (None, "", 0) else ""
        mail1 = (attrs.get("MailingAd1") or "").strip()
        mail_city = (attrs.get("MailingCty") or "").strip()
        mail_state = (attrs.get("MailingSt") or "").strip()
        mail_zip = str(attrs.get("MailingZip") or "").strip()
        total_val = attrs.get("TotalValue")
        return {
            "batch_apn": str(apn or "").strip(),
            "batch_owner_name": str(owner or "").strip(),
            "batch_property_address": site,
            "batch_property_city": city,
            "batch_property_zip": zip_s,
            "batch_mailing_line1": mail1,
            "batch_mailing_city": mail_city,
            "batch_mailing_state": mail_state,
            "batch_mailing_zip": mail_zip,
            "parcel_year_built": "",
            "parcel_total_acct_value": str(total_val if total_val is not None else "").strip(),
            "parcel_legal": str(attrs.get("ParcelLgl") or "").strip(),
        }

    owner = attrs.get("Owner") or attrs.get("Name1") or ""
    apn = attrs.get("ParcelNo") or attrs.get("AccountNo") or ""
    site = attrs.get("PropertyAddress") or ""
    city = attrs.get("PropertyCity") or attrs.get("SiteCity") or attrs.get("City") or ""
    z = attrs.get("PropertyZIP")
    if z is None:
        z = attrs.get("ZIPCode")
    zip_s = str(z) if z not in (None, "") else ""
    mail1 = attrs.get("Address1") or ""
    mail_city = attrs.get("City") or ""
    mail_state = attrs.get("State") or ""
    mail_zip = attrs.get("ZIPCode") or ""
    return {
        "batch_apn": str(apn or "").strip(),
        "batch_owner_name": str(owner or "").strip(),
        "batch_property_address": str(site or "").strip(),
        "batch_property_city": str(city or "").strip(),
        "batch_property_zip": zip_s.strip(),
        "batch_mailing_line1": str(mail1 or "").strip(),
        "batch_mailing_city": str(mail_city or "").strip(),
        "batch_mailing_state": str(mail_state or "").strip(),
        "batch_mailing_zip": str(mail_zip or "").strip(),
        "parcel_year_built": str(attrs.get("YearBuilt") or "").strip(),
        "parcel_total_acct_value": str(attrs.get("TotalAcctValue") or "").strip(),
        "parcel_legal": str(attrs.get("Legal") or "").strip(),
    }


def parcel_attrs_summary(attrs: Dict[str, Any]) -> Tuple[str, str]:
    """Human-readable site line + owner for quick review."""
    fd = flatten_for_batchdata(attrs)
    line = ", ".join(
        x
        for x in (
            fd["batch_property_address"],
            fd["batch_property_city"],
            "OK",
            fd["batch_property_zip"],
        )
        if x
    )
    return line, fd["batch_owner_name"]
