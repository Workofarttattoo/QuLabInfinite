"""
Agent data prep utility

Purpose:
- Aggregate interaction logs and domain artifacts into a JSONL dataset
  for prompt/model fine-tuning (tool-use, safety).

What it does:
- Scans known log/data locations (if present) and emits standardized
  records to `training_data/agent_ft_dataset.jsonl`.
- Records include: source file, tool_name (if known), user prompt,
  assistant response, labels (success/failure), and optional citations.

Usage:
  python scripts/agent_data_prep.py --out training_data/agent_ft_dataset.jsonl

This script is resilient to missing files and will skip absent sources.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "training_data" / "agent_ft_dataset.jsonl"


def read_json_lines(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_json(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def iter_sources() -> Iterable[Path]:
    candidates = [
        ROOT / "ech0_lab_results",
        ROOT / "logs" / "ech0_mcp_server.log",
        ROOT / "logs" / "ech0_mcp_lite.log",
        ROOT / "ech0_marketing_results.json",
        ROOT / "ech0_hive_mind_knowledge.json",
    ]
    for base in candidates:
        if base.is_dir():
            for file in base.glob("*.json"):
                yield file
        elif base.is_file():
            yield base


def normalize_record(raw: Dict, source: str) -> Optional[Dict]:
    """
    Normalize heterogeneous records to a common schema.
    Expected keys (if present): prompt/input, response/output, tool, success.
    """
    prompt = raw.get("prompt") or raw.get("input") or raw.get("question")
    response = raw.get("response") or raw.get("output") or raw.get("answer")
    tool = raw.get("tool") or raw.get("tool_name")
    success = raw.get("success")

    if not prompt and not response:
        return None

    return {
        "source": source,
        "prompt": prompt,
        "response": response,
        "tool": tool,
        "success": success,
        "meta": {k: v for k, v in raw.items() if k not in {"prompt", "input", "question", "response", "output", "answer", "tool", "tool_name", "success"}},
    }


def build_dataset(out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for src in iter_sources():
            if src.suffix == ".log":
                # logs are not structured; skip for now
                continue
            if src.suffix == ".jsonl":
                iterator = read_json_lines(src)
            else:
                payload = read_json(src)
                if payload is None:
                    continue
                iterator = payload if isinstance(payload, list) else [payload]

            for record in iterator:
                normalized = normalize_record(record, source=str(src.relative_to(ROOT)))
                if not normalized:
                    continue
                handle.write(json.dumps(normalized) + "\n")
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate agent logs into JSONL for fine-tuning.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSONL path")
    args = parser.parse_args()

    out_path = Path(args.out).expanduser().resolve()
    total = build_dataset(out_path)
    print(f"[info] wrote {total} records to {out_path}")


if __name__ == "__main__":
    main()
