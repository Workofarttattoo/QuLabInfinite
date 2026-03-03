#!/usr/bin/env python3
"""Leaderboard-oriented eval harness with reproducible fallback.

This runner is designed to make leaderboard-style evaluations easy to launch
while preserving reproducibility artifacts. It prefers lm-eval and falls back
to the local reproducibility runner when lm-eval is unavailable.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class ReproTaskSpec:
    task_name: str
    dataset: str
    mmlu_subject: Optional[str]
    gorilla_file: Optional[str] = None


def _try_run(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(command, text=True).strip()
    except Exception:
        return "unavailable"


def collect_hardware() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version.replace("\n", " "),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
        "git_commit": _try_run(["git", "rev-parse", "HEAD"]),
        "git_branch": _try_run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }
    if shutil.which("nvidia-smi"):
        info["nvidia_smi"] = _try_run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
        )
    else:
        info["nvidia_smi"] = "not_installed"
    return info


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leaderboard-style evaluations with lm-eval and reproducibility fallback."
    )
    parser.add_argument("--model", required=True, help="Model name (for Ollama and metadata).")
    parser.add_argument(
        "--tasks",
        default="arc_challenge,arc_easy,mmlu,winogrande,truthfulqa_mc1,gorilla_exec_simple",
        help=(
            "Comma-separated task list. lm-eval task names are passed through. "
            "Fallback supports arc_challenge, arc_easy, mmlu, mmlu_<subject>, "
            "winogrande/winnogrande, truthfulqa(_mc1), and gorilla(_exec_simple)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "lm_eval", "repro"],
        default="auto",
        help="Runner selection. auto tries lm-eval first, then reproducibility fallback.",
    )
    parser.add_argument("--limit", type=int, default=25, help="Sample limit per task.")
    parser.add_argument("--seed", type=int, default=42, help="Dataset shuffle seed for fallback runner.")
    parser.add_argument("--split", default="test", help="Dataset split for fallback runner.")
    parser.add_argument("--mmlu-subject", default="college_physics", help="Default MMLU subject for fallback.")
    parser.add_argument(
        "--gorilla-file",
        default="BFCL_v3_exec_simple.json",
        help="BFCL JSONL file used by fallback when running Gorilla tasks.",
    )
    parser.add_argument("--backend", choices=["ollama", "command"], default="ollama", help="Fallback backend.")
    parser.add_argument("--command-template", default="", help="Fallback command template for --backend command.")
    parser.add_argument("--timeout", type=int, default=120, help="Per-request timeout for fallback runner.")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable dataset shuffling in fallback runner.")
    parser.add_argument("--num-fewshot", type=int, default=0, help="Few-shot count for lm-eval.")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1", help="OpenAI-compatible endpoint.")
    parser.add_argument("--lm-eval-timeout", type=int, default=180, help="Max runtime in seconds for lm-eval call.")
    parser.add_argument("--output-dir", default="benchmark_runs/leaderboard", help="Root output directory.")
    return parser.parse_args()


def _parse_tasks_csv(tasks_csv: str) -> List[str]:
    tasks = [x.strip() for x in tasks_csv.split(",") if x.strip()]
    if not tasks:
        raise ValueError("At least one task is required.")
    return tasks


def _task_to_repro_spec(task_name: str, default_mmlu_subject: str) -> Optional[ReproTaskSpec]:
    if task_name in {"arc_challenge", "arc_easy"}:
        return ReproTaskSpec(task_name=task_name, dataset=task_name, mmlu_subject=None)
    if task_name == "mmlu":
        return ReproTaskSpec(task_name=task_name, dataset="mmlu", mmlu_subject=default_mmlu_subject)
    if task_name.startswith("mmlu_"):
        subject = task_name[len("mmlu_") :]
        if subject:
            return ReproTaskSpec(task_name=task_name, dataset="mmlu", mmlu_subject=subject)
    if task_name in {"winogrande", "winnogrande"}:
        return ReproTaskSpec(task_name=task_name, dataset="winogrande", mmlu_subject=None)
    if task_name in {"truthfulqa", "truthfulqa_mc1"}:
        return ReproTaskSpec(task_name=task_name, dataset="truthfulqa_mc1", mmlu_subject=None)
    if task_name in {"gorilla", "gorilla_exec_simple"}:
        return ReproTaskSpec(
            task_name=task_name,
            dataset="gorilla_exec_simple",
            mmlu_subject=None,
            gorilla_file="BFCL_v3_exec_simple.json",
        )
    return None


def _find_json_result_file(output_dir: Path) -> Optional[Path]:
    json_files = sorted(output_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return json_files[0] if json_files else None


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def run_lm_eval(
    *,
    run_dir: Path,
    model: str,
    tasks: List[str],
    num_fewshot: int,
    limit: int,
    base_url: str,
    timeout_s: int,
) -> Dict[str, Any]:
    lm_eval_dir = run_dir / "lm_eval"
    lm_eval_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = lm_eval_dir / "stdout.log"
    stderr_path = lm_eval_dir / "stderr.log"

    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "local-chat-completions",
        "--model_args",
        f"model={model},base_url={base_url}",
        "--tasks",
        ",".join(tasks),
        "--num_fewshot",
        str(num_fewshot),
        "--limit",
        str(limit),
        "--log_samples",
        "--show_config",
        "--output_path",
        str(lm_eval_dir),
    ]
    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "EMPTY")

    result: Dict[str, Any] = {
        "runner": "lm_eval",
        "command": shlex.join(command),
        "output_dir": str(lm_eval_dir),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
            env=env,
        )
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        result["returncode"] = completed.returncode

        json_result = _find_json_result_file(lm_eval_dir)
        if completed.returncode == 0 and json_result is not None:
            result["ok"] = True
            result["result_json"] = str(json_result)
            try:
                payload = json.loads(json_result.read_text(encoding="utf-8"))
                result["metrics"] = payload.get("results", {})
            except Exception as exc:  # noqa: BLE001
                result["metrics_parse_error"] = str(exc)
        else:
            result["ok"] = False
            result["error"] = "lm-eval exited non-zero or did not produce a JSON result file."
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(_to_text(exc.stdout), encoding="utf-8")
        stderr_text = _to_text(exc.stderr) + f"\n<TIMEOUT> exceeded {timeout_s}s"
        stderr_path.write_text(stderr_text, encoding="utf-8")
        result["ok"] = False
        result["returncode"] = None
        result["error"] = f"lm-eval timed out after {timeout_s}s."

    return result


def _discover_new_run_dir(output_root: Path, dataset_prefix: str, previous: set[str]) -> Optional[Path]:
    candidates = sorted(
        [p for p in output_root.glob(f"{dataset_prefix}_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    for candidate in reversed(candidates):
        if candidate.name not in previous:
            return candidate
    return candidates[-1] if candidates else None


def run_repro_fallback(
    *,
    run_dir: Path,
    model: str,
    tasks: List[str],
    split: str,
    limit: int,
    seed: int,
    mmlu_subject: str,
    gorilla_file: str,
    backend: str,
    command_template: str,
    timeout: int,
    no_shuffle: bool,
) -> Dict[str, Any]:
    repro_root = run_dir / "repro_runs"
    repro_root.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_llm_repro_benchmark.py")
    per_task: List[Dict[str, Any]] = []

    for task in tasks:
        spec = _task_to_repro_spec(task, mmlu_subject)
        if spec is None:
            per_task.append(
                {
                    "task": task,
                    "ok": False,
                    "error": "Task unsupported by fallback runner.",
                }
            )
            continue

        before = {p.name for p in repro_root.glob(f"{spec.dataset}_*") if p.is_dir()}
        command = [
            sys.executable,
            str(runner),
            "--dataset",
            spec.dataset,
            "--split",
            split,
            "--limit",
            str(limit),
            "--seed",
            str(seed),
            "--model",
            model,
            "--backend",
            backend,
            "--timeout",
            str(timeout),
            "--output-dir",
            str(repro_root),
        ]
        if spec.dataset == "mmlu" and spec.mmlu_subject:
            command.extend(["--mmlu-subject", spec.mmlu_subject])
        if spec.dataset == "gorilla_exec_simple":
            command.extend(["--gorilla-file", spec.gorilla_file or gorilla_file])
        if backend == "command" and command_template:
            command.extend(["--command-template", command_template])
        if no_shuffle:
            command.append("--no-shuffle")

        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        task_log_dir = run_dir / "logs" / task
        task_log_dir.mkdir(parents=True, exist_ok=True)
        (task_log_dir / "stdout.log").write_text(completed.stdout or "", encoding="utf-8")
        (task_log_dir / "stderr.log").write_text(completed.stderr or "", encoding="utf-8")

        run_path = _discover_new_run_dir(repro_root, spec.dataset, before)
        summary_path = run_path / "summary.json" if run_path else None
        summary_payload: Optional[Dict[str, Any]] = None
        if summary_path and summary_path.exists():
            try:
                summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                summary_payload = None

        per_task.append(
            {
                "task": task,
                "dataset": spec.dataset,
                "mmlu_subject": spec.mmlu_subject,
                "ok": completed.returncode == 0 and summary_payload is not None,
                "returncode": completed.returncode,
                "command": shlex.join(command),
                "run_dir": str(run_path) if run_path else None,
                "summary": summary_payload,
            }
        )

    accuracies = [
        float(item["summary"]["accuracy"])
        for item in per_task
        if item.get("ok") and item.get("summary") and "accuracy" in item["summary"]
    ]
    aggregate = {
        "num_tasks": len(per_task),
        "num_success": sum(1 for item in per_task if item.get("ok")),
        "macro_accuracy": (sum(accuracies) / len(accuracies)) if accuracies else None,
    }
    return {"runner": "repro", "output_dir": str(repro_root), "aggregate": aggregate, "tasks": per_task}


def main() -> int:
    args = parse_args()
    tasks = _parse_tasks_csv(args.tasks)

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_dir) / f"leaderboard_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "hardware.json", collect_hardware())
    _write_json(
        run_dir / "manifest.json",
        {
            "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
            "command": " ".join(shlex.quote(x) for x in sys.argv),
            "args": vars(args),
            "tasks": tasks,
        },
    )

    lm_eval_result: Optional[Dict[str, Any]] = None
    repro_result: Optional[Dict[str, Any]] = None
    selected_runner = None

    if args.mode in {"auto", "lm_eval"}:
        lm_eval_result = run_lm_eval(
            run_dir=run_dir,
            model=args.model,
            tasks=tasks,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            base_url=args.base_url,
            timeout_s=args.lm_eval_timeout,
        )
        if lm_eval_result.get("ok"):
            selected_runner = "lm_eval"
        elif args.mode == "lm_eval":
            selected_runner = "lm_eval_failed"

    if selected_runner is None and args.mode in {"auto", "repro"}:
        repro_result = run_repro_fallback(
            run_dir=run_dir,
            model=args.model,
            tasks=tasks,
            split=args.split,
            limit=args.limit,
            seed=args.seed,
            mmlu_subject=args.mmlu_subject,
            gorilla_file=args.gorilla_file,
            backend=args.backend,
            command_template=args.command_template,
            timeout=args.timeout,
            no_shuffle=args.no_shuffle,
        )
        selected_runner = "repro"

    summary = {
        "run_dir": str(run_dir),
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "selected_runner": selected_runner,
        "model": args.model,
        "tasks": tasks,
        "lm_eval": lm_eval_result,
        "repro": repro_result,
    }
    _write_json(run_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
