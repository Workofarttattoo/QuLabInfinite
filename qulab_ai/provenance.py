
"""
Provenance stamping: SHA-256 hashing of inputs/outputs + minimal citation block.
"""
import hashlib, json, datetime
from typing import Any, Dict

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def stamp(record: Dict[str, Any]) -> Dict[str, Any]:
    """Attach timestamp and digest over the JSON-normalized record (without the digest)."""
    rec = dict(record)
    rec["timestamp_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
    tmp = dict(rec)
    tmp.pop("digest", None)
    raw = json.dumps(tmp, sort_keys=True).encode("utf-8")
    rec["digest"] = sha256_bytes(raw)
    return rec

def citation_block(**kwargs) -> Dict[str, Any]:
    """
    Minimal structured citations. Example:
    citation_block(doi="10.1038/s41597-025-04720-7", source="QCML Scientific Data 2025")
    """
    return {k: v for k, v in kwargs.items() if v is not None}
