from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime


class Registry:
    def __init__(self, path: str = "data/registry.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add_entry(self, name: str, filepath: str, num_rows: int) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "dataset": name,
            "path": str(filepath),
            "rows": int(num_rows),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
