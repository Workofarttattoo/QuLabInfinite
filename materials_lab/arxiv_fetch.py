#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

arxiv_fetch - Part of Materials Lab
"""

"""
Fetch experimental material property snippets from arXiv.

This script is not a substitute for manual curation; it simply collects
candidate sentences containing numeric property statements so domain experts
can validate and ingest them into the QuLab Infinite database.

Usage
-----
  python materials_lab/arxiv_fetch.py --material "SS 304" --keywords tensile,yield --limit 10

Outputs a JSON structure with entries containing:
  - material: material label used in the query
  - title: paper title
  - published: ISO timestamp
  - summary_snippet: text excerpt containing numeric facts
  - link: URL to the arXiv abstract
  - query: the actual search query executed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Adjust path to import from chemistry_lab
sys.path.append(str(Path(__file__).resolve().parents[1]))

from chemistry_lab.datasets.arxiv import load_arxiv_materials


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch arXiv summaries for material property data points.")
    parser.add_argument("--material", required=True, help="Material name, e.g. 'SS 304' or 'Ti-6Al-4V'.")
    parser.add_argument(
        "--keywords",
        default="tensile,strength,thermal",
        help="Comma-separated keywords to include in the query.",
    )
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of arXiv entries to fetch (default: 10).")
    parser.add_argument("--output", help="Optional path to write JSON output.")
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keywords = [kw.strip() for kw in args.keywords.split(",") if kw.strip()]
    
    df = load_arxiv_materials(args.material, keywords, args.limit)
    
    # The dataframe is converted to a list of dictionaries for JSON serialization
    snippets = df.to_dict('records')

    payload = {"material": args.material, "keywords": keywords, "results": snippets}

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(payload, fh, indent=2)

    if args.json or not args.output:
        print(json.dumps(payload, indent=2))
    else:
        print(f"[info] Wrote {len(snippets)} entries to {args.output}")


if __name__ == "__main__":
    main()
