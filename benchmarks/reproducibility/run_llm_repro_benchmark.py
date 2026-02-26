#!/usr/bin/env python3
"""Reproducible LLM benchmark runner for MMLU and ARC.

This script prioritizes reproducibility over leaderboard formatting by logging:
- exact prompts
- exact model outputs
- exact hardware/runtime metadata

It also creates a GitHub-ready reproducibility bundle so others can rerun the
same evaluation with a single command.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import random
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible MMLU/ARC subset benchmarks.")
    parser.add_argument("--dataset", choices=["mmlu", "arc_challenge", "arc_easy"], required=True)
    parser.add_argument("--mmlu-subject", default="college_physics", help="MMLU subject when --dataset=mmlu")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=50, help="Number of examples to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", required=True, help="Model identifier for metadata")
    parser.add_argument(
        "--backend",
        choices=["ollama", "command"],
        default="ollama",
        help="Inference backend. 'command' uses --command-template.",
    )
    parser.add_argument(
        "--command-template",
        default="",
        help=(
            "Template for custom backend command. Must include {prompt}. "
            "Example: 'python my_runner.py --model X --prompt {prompt}'"
        ),
    )
    parser.add_argument("--output-dir", default="benchmark_runs")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-shuffle", action="store_true", help="Keep original dataset order")
    return parser.parse_args()


def _try_run(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(command, text=True).strip()
    except Exception:
        return "unavailable"


def get_hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version.replace("\n", " "),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "git_commit": _try_run(["git", "rev-parse", "HEAD"]),
        "git_branch": _try_run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }

    if shutil.which("nvidia-smi"):
        try:
            smi = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                text=True,
            ).strip()
            info["nvidia_smi"] = smi
        except Exception:
            info["nvidia_smi"] = "unavailable"
    else:
        info["nvidia_smi"] = "not_installed"

    return info


def load_examples(dataset_name: str, split: str, mmlu_subject: str, limit: int, seed: int, no_shuffle: bool) -> List[Dict[str, Any]]:
    from datasets import load_dataset  # lazy import

    if dataset_name == "mmlu":
        ds = load_dataset("cais/mmlu", mmlu_subject, split=split)
    elif dataset_name == "arc_challenge":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    else:
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)

    if not no_shuffle:
        ds = ds.shuffle(seed=seed)

    return [ds[i] for i in range(min(limit, len(ds)))]


def build_prompt(example: Dict[str, Any], dataset_name: str) -> Tuple[str, str, str]:
    if dataset_name == "mmlu":
        choices = example["choices"]
        answer_idx = int(example["answer"])
        answer_key = chr(65 + answer_idx)
        rendered_choices = "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices))
        prompt = (
            "You are taking a multiple-choice benchmark. "
            "Return only the final answer letter.\n\n"
            f"Question:\n{example['question']}\n\n"
            f"Choices:\n{rendered_choices}\n\n"
            "Answer:"
        )
        return prompt, answer_key, example["question"]

    # ARC format
    question = example["question"]
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]
    rendered_choices = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    answer_key = str(example["answerKey"]).strip()
    prompt = (
        "You are taking a multiple-choice benchmark. "
        "Return only the final answer label.\n\n"
        f"Question:\n{question}\n\n"
        f"Choices:\n{rendered_choices}\n\n"
        "Answer:"
    )
    return prompt, answer_key, question


def run_inference(prompt: str, backend: str, model: str, command_template: str, timeout: int) -> str:
    if backend == "ollama":
        command = ["ollama", "run", model, prompt]
    else:
        if "{prompt}" not in command_template:
            raise ValueError("--command-template must include {prompt} placeholder")
        command_text = command_template.format(prompt=shlex.quote(prompt))
        command = shlex.split(command_text)

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"Model command failed ({completed.returncode}): {stderr}")

    return completed.stdout.strip()


def normalize_pred(raw_output: str) -> str:
    text = raw_output.strip()
    if not text:
        return ""
    first_token = text.split()[0].strip().strip(".\n\t")
    return first_token.upper()


def ensure_repro_bundle(run_dir: Path) -> None:
    repo_dir = run_dir / "repro_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)

    readme = repo_dir / "README.md"
    readme.write_text(
        "# Reproducibility Bundle\n\n"
        "This folder is GitHub-ready. Commit it to a public repository to make the run reproducible.\n\n"
        "## Included\n"
        "- `hardware.json`: machine/runtime metadata\n"
        "- `run_manifest.json`: benchmark arguments and dataset selection\n"
        "- `predictions.jsonl`: exact prompts, outputs, and labels\n"
        "- `summary.json`: aggregate accuracy\n\n"
        "## Rerun\n"
        "Use the exact command from `run_manifest.json.command`.\n",
        encoding="utf-8",
    )



def main() -> int:
    args = parse_args()

    out_root = Path(args.output_dir)
    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"{args.dataset}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    hardware = get_hardware_info()
    (run_dir / "hardware.json").write_text(json.dumps(hardware, indent=2), encoding="utf-8")

    command = " ".join(shlex.quote(x) for x in sys.argv)
    manifest = {
        "command": f"python {command}",
        "dataset": args.dataset,
        "mmlu_subject": args.mmlu_subject,
        "split": args.split,
        "limit": args.limit,
        "seed": args.seed,
        "model": args.model,
        "backend": args.backend,
        "command_template": args.command_template,
        "timeout": args.timeout,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    examples = load_examples(args.dataset, args.split, args.mmlu_subject, args.limit, args.seed, args.no_shuffle)

    results: List[Dict[str, Any]] = []
    correct = 0
    random.seed(args.seed)

    for idx, example in enumerate(examples):
        prompt, gold, question = build_prompt(example, args.dataset)
        try:
            raw_output = run_inference(prompt, args.backend, args.model, args.command_template, args.timeout)
            pred = normalize_pred(raw_output)
            is_correct = pred == str(gold).upper()
        except Exception as exc:
            raw_output = f"<ERROR> {exc}"
            pred = ""
            is_correct = False

        if is_correct:
            correct += 1

        results.append(
            {
                "index": idx,
                "question": question,
                "prompt": prompt,
                "gold": gold,
                "raw_output": raw_output,
                "pred": pred,
                "correct": is_correct,
            }
        )

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    accuracy = correct / len(results) if results else 0.0
    summary = {
        "num_examples": len(results),
        "num_correct": correct,
        "accuracy": accuracy,
        "dataset": args.dataset,
        "model": args.model,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    ensure_repro_bundle(run_dir)
    for file_name in ["hardware.json", "run_manifest.json", "predictions.jsonl", "summary.json"]:
        shutil.copy2(run_dir / file_name, run_dir / "repro_repo" / file_name)

    print(json.dumps(summary, indent=2))
    print(f"\nReproducibility bundle ready at: {run_dir / 'repro_repo'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
