from __future__ import annotations
from typing import Dict
import json, pathlib, time

# Minimal local registry (swap with your QuLab DB)
REG_PATH = pathlib.Path('data/registry.jsonl')
REG_PATH.parent.mkdir(parents=True, exist_ok=True)

def register_dataset(name: str, path: str, kind: str, fingerprint: str, meta: Dict) -> Dict:
    entry = {
        "name": name,
        "path": path,
        "kind": kind,
        "fingerprint": fingerprint,
        "meta": meta,
        "registered_at": time.time(),
    }
    with REG_PATH.open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + "\n")
    return entry
