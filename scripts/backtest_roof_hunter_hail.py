#!/usr/bin/env python3
"""Backtest Roof Hunter hail predictions against known historical outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hail_model.backtest import RoofHunterBacktester, load_backtest_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="CSV, JSON, or JSONL file with known hail outcomes.")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional XGBoost model path. Defaults to dual-pol hail intelligence only.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used for binary metrics.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON report path. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Exclude per-record comparisons from the report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_backtest_records(args.input)
    report = RoofHunterBacktester(model_path=args.model_path).backtest(
        records,
        threshold=args.threshold,
    )
    payload = report.to_dict(include_records=not args.summary_only)
    text = json.dumps(payload, indent=2) + "\n"

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote backtest report to {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
