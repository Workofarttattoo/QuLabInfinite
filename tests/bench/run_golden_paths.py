#!/usr/bin/env python3
"""Utility to sanity-check benchmark registry entries.

This is a placeholder runner: it loads each YAML definition and verifies the
required keys exist. Real calibration routines should replace this checker
with model execution + acceptance tests. The script exits with code 1 if any
benchmark definition is malformed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REQUIRED_KEYS = {"id", "summary", "inputs", "criteria", "data_ref", "engine"}


def iter_benchmarks(root: Path):
    for path in root.rglob("*.yaml"):
        yield path


def validate_file(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        data = yaml.safe_load(path.read_text())
    except Exception as exc:  # pragma: no cover - YAML parse error
        return [f"{path}: failed to parse ({exc})"]

    if not isinstance(data, dict):
        return [f"{path}: expected mapping root"]

    missing = REQUIRED_KEYS - data.keys()
    if missing:
        errors.append(f"{path}: missing keys {sorted(missing)}")

    if "criteria" in data and not isinstance(data["criteria"], dict):
        errors.append(f"{path}: criteria must be mapping")

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate benchmark registry metadata and optionally execute calibrations."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run each benchmark's calibration script after validation.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit immediately with status 1 if any calibration fails.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Request JSON output from calibration scripts instead of human-readable text.",
    )
    return parser.parse_args()


def run_calibration(script_path: Path, strict: bool, json_mode: bool) -> int:
    command = [sys.executable, str(script_path)]
    if strict:
        command.append("--strict")
    if json_mode:
        command.append("--json")

    result = subprocess.run(command, capture_output=not json_mode, text=True)

    if json_mode:
        # In JSON mode the child printed to stdout; propagate stderr if any.
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    else:
        if result.stdout:
            print(result.stdout.rstrip())
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)

    return result.returncode


def main() -> int:
    args = parse_args()

    root = Path(__file__).resolve().parent
    errors: list[str] = []
    benchmark_payloads: list[tuple[Path, dict]] = []
    for yaml_path in iter_benchmarks(root):
        validation_errors = validate_file(yaml_path)
        errors.extend(validation_errors)
        if not validation_errors:
            with yaml_path.open("r") as handle:
                data = yaml.safe_load(handle)
            benchmark_payloads.append((yaml_path, data))

    if errors:
        for msg in errors:
            print(f"[fail] {msg}")
        return 1

    print("[ok] Benchmark registry definitions look sane.")

    if not args.execute:
        return 0

    print("[info] Executing calibration scripts…")
    exit_code = 0
    for yaml_path, metadata in benchmark_payloads:
        engine = metadata.get("engine", {})
        script_ref = engine.get("calibration_script")
        if not script_ref:
            print(f"[warn] {yaml_path}: no calibration_script defined; skipping.")
            continue

        script_path = (yaml_path.parent / script_ref).resolve()
        if not script_path.exists():
            print(f"[fail] {yaml_path}: calibration script {script_path} missing.")
            exit_code = 1
            if args.strict:
                break
            continue

        print(f"[run] {metadata['id']} → {script_path}")
        rc = run_calibration(script_path, strict=args.strict, json_mode=args.json)
        if rc != 0:
            exit_code = rc
            if args.strict:
                break

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
