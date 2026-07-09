"""BatchData.com API client for property lookup and skip-trace.

Reads BATCHDATA_API_KEY from the .env file (never commit keys to git).

Endpoint base and paths are set as module-level constants so they can be
confirmed against https://developer.batchdata.com before use.  The skip-trace
endpoint in particular may require a plan that includes it.

Usage example:
    from roof_hunter.batchdata_client import BatchDataClient
    client = BatchDataClient.from_env()
    results = client.skip_trace_leads(rows)     # list of dicts with batch_* cols
    client.export_response_csv(results, Path("output/skiptrace_response.csv"))
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ── API surface (verify against https://developer.batchdata.com) ──────────────
_BASE_URL = "https://api.batchdata.com/api/v1"
_SKIP_TRACE_EP = f"{_BASE_URL}/person/skip-trace"       # POST – batch skip trace
_PROPERTY_LOOKUP_EP = f"{_BASE_URL}/property/lookup"    # POST – property detail by address
_PROPERTY_SEARCH_EP = f"{_BASE_URL}/property/search"    # POST – search by criteria

_USER_AGENT = "QuLabInfinite-RoofHunter/1.0"
_DEFAULT_RATE_SEC = 0.25   # 4 req/s conservative default
_MAX_BATCH = 50             # maximum records per single skip-trace request


class BatchDataError(RuntimeError):
    pass


class BatchDataClient:
    """Minimal BatchData.com API wrapper sufficient for lead skip-tracing."""

    def __init__(self, api_key: str, *, rate_sec: float = _DEFAULT_RATE_SEC):
        if not api_key:
            raise BatchDataError("BATCHDATA_API_KEY is empty – set it in .env")
        self._api_key = api_key
        self._rate_sec = max(0.0, rate_sec)
        self._last_call: float = 0.0

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls, *, env_var: str = "BATCHDATA_API_KEY") -> "BatchDataClient":
        """Load API key from environment (uses python-dotenv if available)."""
        try:
            from dotenv import load_dotenv  # type: ignore[import]
            load_dotenv()
        except ImportError:
            pass
        key = os.environ.get(env_var, "").strip()
        if not key:
            raise BatchDataError(
                f"Environment variable {env_var} not set. "
                "Copy .env.secure.example to .env and fill in your BatchData key."
            )
        return cls(key)

    # ── request helpers ───────────────────────────────────────────────────────

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        wait = self._rate_sec - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _post(self, url: str, payload: Dict[str, Any], *, timeout: float = 60.0) -> Dict[str, Any]:
        self._throttle()
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-api-key": self._api_key,
                "User-Agent": _USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body_snippet = ""
            try:
                body_snippet = e.read().decode("utf-8", errors="replace")[:400]
            except Exception:
                pass
            raise BatchDataError(f"HTTP {e.code} from {url}: {body_snippet}") from e
        except urllib.error.URLError as e:
            raise BatchDataError(f"Network error contacting {url}: {e.reason}") from e
        return json.loads(raw)

    # ── property lookup ───────────────────────────────────────────────────────

    def lookup_property(
        self,
        address: str,
        city: str = "",
        state: str = "",
        zip_code: str = "",
    ) -> Dict[str, Any]:
        """Single property lookup by address string."""
        payload: Dict[str, Any] = {"address": address}
        if city:
            payload["city"] = city
        if state:
            payload["state"] = state
        if zip_code:
            payload["zip"] = zip_code
        return self._post(_PROPERTY_LOOKUP_EP, payload)

    # ── skip-trace (batch) ────────────────────────────────────────────────────

    def skip_trace_leads(
        self,
        rows: Sequence[Dict[str, str]],
        *,
        max_batch: int = _MAX_BATCH,
        address_col: str = "batch_property_address",
        city_col: str = "batch_property_city",
        state_col: str = "batch_mailing_state",
        zip_col: str = "batch_property_zip",
        name_col: str = "batch_owner_name",
        apn_col: str = "batch_apn",
    ) -> List[Dict[str, Any]]:
        """Skip-trace a list of lead rows.  Returns enriched response dicts.

        Each output dict carries all original lead fields plus BatchData response
        keys (``bd_phone_1``, ``bd_email_1``, ``bd_owner_full_name``, etc.) and
        ``bd_status`` / ``bd_error`` for diagnostics.
        """
        results: List[Dict[str, Any]] = []
        chunks = _chunk(list(rows), max_batch)

        for batch_idx, chunk in enumerate(chunks):
            records = []
            for r in chunk:
                rec: Dict[str, Any] = {}
                addr = (r.get(address_col) or "").strip()
                name = (r.get(name_col) or "").strip()
                apn = (r.get(apn_col) or "").strip()
                if addr:
                    rec["address"] = addr
                    rec["city"] = (r.get(city_col) or "").strip()
                    rec["state"] = (r.get(state_col) or "").strip()
                    rec["zip"] = (r.get(zip_col) or "").strip()
                if name:
                    rec["name"] = name
                if apn:
                    rec["apn"] = apn
                # Attach lead metadata for correlation
                rec["_lead_id"] = r.get("lead_id", "")
                records.append(rec)

            payload = {"requests": records}
            try:
                resp = self._post(_SKIP_TRACE_EP, payload)
                resp_records = resp.get("data", {}).get("results") or resp.get("results") or []
                for orig, returned in _zip_response(chunk, resp_records):
                    enriched = dict(orig)
                    enriched.update(_flatten_skip_trace_result(returned))
                    enriched["bd_status"] = "ok"
                    enriched["bd_error"] = ""
                    results.append(enriched)
            except BatchDataError as exc:
                for orig in chunk:
                    enriched = dict(orig)
                    enriched["bd_status"] = "error"
                    enriched["bd_error"] = str(exc)
                    results.append(enriched)

        return results

    # ── output helpers ────────────────────────────────────────────────────────

    @staticmethod
    def export_response_csv(results: List[Dict[str, Any]], path: Path) -> None:
        if not results:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(dict.fromkeys(k for r in results for k in r))
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k, "") for k in fieldnames})


# ── priority sort ─────────────────────────────────────────────────────────────

_RECENCY_HALFLIFE_DAYS = 10.0   # recent events decay with 10-day half-life
_WEALTH_WEIGHT = 0.25            # portion of composite score from ZIP wealth


def priority_score(
    lead_rank_score: float,
    days_since_event: float,
    zip_median_income: float,
    *,
    income_floor: float = 30_000,
    income_ceil: float = 180_000,
) -> float:
    """Composite priority score for BatchData submission ordering.

    Sort descending.  Scoring captures three axes:
      1. Damage magnitude  — lead_rank_score (higher = more damage/vulnerability)
      2. Recency           — exponential decay over RECENCY_HALFLIFE_DAYS
      3. ZIP wealth        — linear 0-1 scale between income_floor / income_ceil

    Formula:
        recency = exp(-days / halflife)
        wealth  = clamp((income - floor) / (ceil - floor), 0, 1)
        score   = lead_rank_score * recency * (1 - WEALTH_WEIGHT + WEALTH_WEIGHT * wealth)
    """
    recency = math.exp(-max(0.0, days_since_event) / _RECENCY_HALFLIFE_DAYS)
    income_norm = max(0.0, min(1.0, (zip_median_income - income_floor) / (income_ceil - income_floor)))
    wealth_mult = 1.0 - _WEALTH_WEIGHT + _WEALTH_WEIGHT * income_norm
    return max(0.0, lead_rank_score * recency * wealth_mult)


def sort_leads_for_batchdata(
    rows: List[Dict[str, Any]],
    income_lookup: Dict[str, float],
    *,
    reference_date: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Sort lead rows into BatchData submission order.

    Priority: most damage + most recent first → as recency falls, richest ZIPs
    float up → lower damage last.

    Args:
        rows:          Lead dicts.  Expected keys: ``lead_rank_score``,
                       ``report_datetime`` (ISO), ``batch_property_zip``.
        income_lookup: zip → median household income (from Census ACS or pre-built table).
        reference_date: Treated as "now" for recency calculation.  Defaults to UTC now.
    """
    ref = reference_date or datetime.now(timezone.utc)
    scored: List[tuple[float, Dict[str, Any]]] = []
    for row in rows:
        lrs = _to_float(row.get("lead_rank_score") or row.get("severity_score_0_1"), 0.5)
        rdt_s = str(row.get("report_datetime") or "")
        days_old = _days_since(rdt_s, ref)
        zip_code = str(row.get("batch_property_zip") or row.get("inferred_zip") or "")
        income = income_lookup.get(zip_code, 55_000.0)
        ps = priority_score(lrs, days_old, income)
        scored.append((ps, row))
    scored.sort(key=lambda t: -t[0])
    return [r for _, r in scored]


# ── private helpers ───────────────────────────────────────────────────────────

def _chunk(lst: List, n: int) -> List[List]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def _zip_response(
    originals: List[Dict[str, str]],
    returned: List[Dict[str, Any]],
) -> List[tuple[Dict[str, str], Dict[str, Any]]]:
    """Pair original rows with response records; fill blanks if counts mismatch."""
    pairs = []
    for i, orig in enumerate(originals):
        resp = returned[i] if i < len(returned) else {}
        pairs.append((orig, resp))
    return pairs


def _flatten_skip_trace_result(result: Dict[str, Any]) -> Dict[str, str]:
    """Extract useful contact fields from a BatchData skip-trace result record."""
    phones = result.get("phones") or result.get("phoneNumbers") or []
    emails = result.get("emails") or result.get("emailAddresses") or []
    owner = result.get("fullName") or result.get("name") or ""
    return {
        "bd_owner_full_name": str(owner).strip(),
        "bd_phone_1": str(phones[0] if phones else "").strip(),
        "bd_phone_2": str(phones[1] if len(phones) > 1 else "").strip(),
        "bd_email_1": str(emails[0] if emails else "").strip(),
        "bd_do_not_call": str(result.get("doNotCall") or "").strip(),
        "bd_property_value": str(result.get("estimatedValue") or result.get("value") or "").strip(),
        "bd_year_built": str(result.get("yearBuilt") or "").strip(),
        "bd_raw_json": json.dumps(result, ensure_ascii=False),
    }


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _days_since(dt_str: str, ref: datetime) -> float:
    if not dt_str:
        return 999.0
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (ref - dt).total_seconds() / 86400.0)
    except ValueError:
        return 999.0


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python batchdata_client.py <leads_csv> [--dry-run]")
        sys.exit(1)
    dry = "--dry-run" in sys.argv
    leads_path = Path(sys.argv[1])
    with leads_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows from {leads_path}")
    if dry:
        print("Dry-run: skipping API calls.  Priority score for first 5 rows:")
        for r in rows[:5]:
            ps = priority_score(_to_float(r.get("lead_rank_score"), 0.5), 3.0, 75_000)
            print(f"  lead_id={r.get('lead_id')} score={ps:.4f}")
    else:
        client = BatchDataClient.from_env()
        results = client.skip_trace_leads(rows[:10])
        out = leads_path.with_name(leads_path.stem + "_skiptrace.csv")
        BatchDataClient.export_response_csv(results, out)
        print(f"Wrote {len(results)} rows to {out}")
