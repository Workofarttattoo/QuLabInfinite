#!/usr/bin/env python3
"""Reproducible benchmark runner for ARC and MMLU subsets.

This tool is designed for auditability over leaderboard optimization:
- logs exact prompts
- logs exact model outputs
- logs hardware information
- writes deterministic, shareable artifacts
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class BenchmarkExample:
    example_id: str
    question: str
    choices: List[str]
    answer_key: str
    subject: Optional[str] = None


def _run_cmd(cmd: List[str]) -> Dict[str, Any]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "command": cmd,
            "returncode": out.returncode,
            "stdout": out.stdout.strip(),
            "stderr": out.stderr.strip(),
        }
    except FileNotFoundError:
        return {"command": cmd, "returncode": 127, "stdout": "", "stderr": "command not found"}


def collect_hardware() -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version,
            "cpu_count": os.cpu_count(),
        },
        "commands": {
            "lscpu": _run_cmd(["lscpu"]),
            "nvidia_smi": _run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
            "free": _run_cmd(["free", "-h"]),
        },
    }
    return data


def _load_from_hf(dataset_name: str, config_name: str, split: str):
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "datasets package is required. Install with: pip install datasets"
        ) from exc

    return load_dataset(dataset_name, config_name, split=split)


def load_arc_subset(limit: int) -> List[BenchmarkExample]:
    ds = _load_from_hf("allenai/ai2_arc", "ARC-Challenge", "test")
    examples: List[BenchmarkExample] = []
    for row in ds.select(range(min(limit, len(ds)))):
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        # normalize into A/B/C/... order
        choice_map = {label.strip(): text.strip() for label, text in zip(labels, texts)}
        canonical_labels = ["A", "B", "C", "D", "E"]
        choices = [choice_map[label] for label in canonical_labels if label in choice_map]
        answer_key = row["answerKey"].strip()

        examples.append(
            BenchmarkExample(
                example_id=str(row["id"]),
                question=row["question"].strip(),
                choices=choices,
                answer_key=answer_key,
                subject="arc_challenge",
            )
        )
    return examples


def load_mmlu_subset(limit: int, subject: Optional[str]) -> List[BenchmarkExample]:
    ds = _load_from_hf("cais/mmlu", "all", "test")
    if subject:
        ds = ds.filter(lambda x: x["subject"] == subject)

    examples: List[BenchmarkExample] = []
    for row in ds.select(range(min(limit, len(ds)))):
        choices = [str(c).strip() for c in row["choices"]]
        answer_idx = int(row["answer"])
        answer_key = chr(ord("A") + answer_idx)
        examples.append(
            BenchmarkExample(
                example_id=str(row.get("id", hashlib.sha256(row["question"].encode()).hexdigest()[:12])),
                question=row["question"].strip(),
                choices=choices,
                answer_key=answer_key,
                subject=row.get("subject"),
            )
        )
    return examples


def format_prompt(example: BenchmarkExample) -> str:
    lines = [
        "Answer the multiple-choice question with a single letter (A, B, C, D, or E).",
        "",
        f"Question: {example.question}",
        "Choices:",
    ]
    for i, choice in enumerate(example.choices):
        label = chr(ord("A") + i)
        lines.append(f"{label}. {choice}")
    lines.append("")
    lines.append("Final answer:")
    return "\n".join(lines)


def run_model(prompt: str, model_command: Optional[str]) -> str:
    if not model_command:
        return "A"  # deterministic fallback; caller should provide real model command for meaningful scores.

    proc = subprocess.run(
        model_command,
        input=prompt,
        capture_output=True,
        text=True,
        shell=True,
        check=False,
    )
    output = (proc.stdout or "").strip()
    if not output and proc.stderr:
        output = proc.stderr.strip()
    return output


def extract_choice_label(output: str) -> Optional[str]:
    match = re.search(r"\b([A-E])\b", output.upper())
    return match.group(1) if match else None


def evaluate(examples: Iterable[BenchmarkExample], model_command: Optional[str]) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    correct = 0

    for idx, ex in enumerate(examples, start=1):
        prompt = format_prompt(ex)
        raw_output = run_model(prompt, model_command)
        pred = extract_choice_label(raw_output)
        is_correct = pred == ex.answer_key
        if is_correct:
            correct += 1

        records.append(
            {
                "index": idx,
                "example_id": ex.example_id,
                "subject": ex.subject,
                "prompt": prompt,
                "raw_output": raw_output,
                "predicted_label": pred,
                "answer_key": ex.answer_key,
                "correct": is_correct,
            }
        )

    total = len(records)
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "records": records,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible ARC/MMLU benchmark subset")
    parser.add_argument("--benchmark", choices=["arc", "mmlu"], required=True)
    parser.add_argument("--limit", type=int, default=100, help="Number of samples to run")
    parser.add_argument("--mmlu-subject", default=None, help="Optional MMLU subject filter")
    parser.add_argument(
        "--model-command",
        default=None,
        help="Shell command that reads prompt from stdin and writes answer text to stdout",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/repro_benchmark",
        help="Root output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{args.benchmark}_{timestamp}"
    run_dir = Path(args.output_dir) / run_id

    if args.benchmark == "arc":
        examples = load_arc_subset(args.limit)
    else:
        examples = load_mmlu_subset(args.limit, args.mmlu_subject)

    hardware = collect_hardware()
    result = evaluate(examples, args.model_command)

    config = {
        "benchmark": args.benchmark,
        "limit": args.limit,
        "mmlu_subject": args.mmlu_subject,
        "model_command": args.model_command,
        "run_id": run_id,
    }

    prompts_rows = [
        {
            "index": r["index"],
            "example_id": r["example_id"],
            "subject": r["subject"],
            "prompt": r["prompt"],
        }
        for r in result["records"]
    ]
    outputs_rows = [
        {
            "index": r["index"],
            "example_id": r["example_id"],
            "raw_output": r["raw_output"],
            "predicted_label": r["predicted_label"],
            "answer_key": r["answer_key"],
            "correct": r["correct"],
        }
        for r in result["records"]
    ]

    write_json(run_dir / "config.json", config)
    write_json(run_dir / "hardware.json", hardware)
    write_json(run_dir / "summary.json", {
        "run_id": run_id,
        "benchmark": args.benchmark,
        "samples": result["total"],
        "correct": result["correct"],
        "accuracy": result["accuracy"],
    })
    write_json(run_dir / "full_results.json", result)
    write_jsonl(run_dir / "prompts.jsonl", prompts_rows)
    write_jsonl(run_dir / "outputs.jsonl", outputs_rows)

    print(f"Run complete: {run_dir}")
    print(f"Accuracy: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")


if __name__ == "__main__":
    main()
