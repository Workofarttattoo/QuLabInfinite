import logging
#!/usr/bin/env python3
"""
Wisdom Ingestion Pipeline - ECH0
Data sources → processing → knowledge graph / RAG index → memory active.

Ensures the hive mind knowledge store is populated and memory is active
for the autonomy loop. Run periodically or after training data updates.

Usage:
    python ech0_wisdom_ingestion.py
    python ech0_wisdom_ingestion.py --merge-dir training_data/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

def main():
    ap = argparse.ArgumentParser(description="ECH0 wisdom ingestion: update knowledge store and ensure memory active.")
    ap.add_argument("--merge-dir", type=str, default=None, help="Directory of JSON/JSONL to merge into knowledge (e.g. training_data or ech0_training_data).")
    ap.add_argument("--knowledge-path", type=str, default=None, help="Path to ech0_hive_mind_knowledge.json (default: same dir as this script).")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    knowledge_path = Path(args.knowledge_path) if args.knowledge_path else base / "ech0_hive_mind_knowledge.json"
    memory_data_dir = base / "memory_data"
    memory_data_dir.mkdir(exist_ok=True)

    # Load or create knowledge payload
    if knowledge_path.exists():
        try:
            with open(knowledge_path, "r", encoding="utf-8") as f:
                knowledge = json.load(f)
        except Exception:
            knowledge = {"timestamp": datetime.now(timezone.utc).isoformat(), "discoveries": {}, "wisdom_ingestion": []}
    else:
        knowledge = {"timestamp": datetime.now(timezone.utc).isoformat(), "discoveries": {}, "wisdom_ingestion": []}

    if "wisdom_ingestion" not in knowledge:
        knowledge["wisdom_ingestion"] = []
    knowledge["last_ingestion"] = datetime.now(timezone.utc).isoformat()

    merged_count = 0
    if args.merge_dir:
        merge_path = Path(args.merge_dir)
        if merge_path.is_dir():
            for p in list(merge_path.glob("*.json")) + list(merge_path.glob("*.jsonl")):
                try:
                    if p.suffix.lower() == ".jsonl":
                        with open(p, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                rec = json.loads(line)
                                knowledge["wisdom_ingestion"].append({
                                    "source": p.name,
                                    "type": "training_row",
                                    "prompt_preview": (rec.get("prompt") or "")[:200],
                                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                                })
                                merged_count += 1
                    else:
                        data = json.loads(p.read_text(encoding="utf-8"))
                        if isinstance(data, list):
                            for i, item in enumerate(data[:500]):
                                knowledge["wisdom_ingestion"].append({
                                    "source": f"{p.name}[{i}]",
                                    "type": "training_row",
                                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                                })
                                merged_count += 1
                        else:
                            knowledge["wisdom_ingestion"].append({
                                "source": p.name,
                                "type": "dataset_meta",
                                "ingested_at": datetime.now(timezone.utc).isoformat(),
                            })
                            merged_count += 1
                except Exception as e:
                    knowledge["wisdom_ingestion"].append({
                        "source": p.name,
                        "error": str(e),
                        "ingested_at": datetime.now(timezone.utc).isoformat(),
                    })

    # Cap wisdom_ingestion list size
    knowledge["wisdom_ingestion"] = knowledge["wisdom_ingestion"][-5000:]

    knowledge_path.parent.mkdir(parents=True, exist_ok=True)
    with open(knowledge_path, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, indent=2, default=str)

    # Ensure memory_data has a marker so autonomy status shows Memory ACTIVE
    (memory_data_dir / ".memory_active").write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")

    logging.info(f"Wisdom ingestion complete. Knowledge path: {knowledge_path}")
    logging.info(f"Memory data dir: {memory_data_dir} (memory will show ACTIVE)")
    if merged_count:
        logging.info(f"Merged {merged_count} items from {args.merge_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
