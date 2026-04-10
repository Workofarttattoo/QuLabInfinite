#!/usr/bin/env python3
"""
Fetch research papers from arXiv based on search queries.

This script searches for papers on arXiv and provides a list of relevant
publications with their titles, authors, summaries, and links.

Usage
-----
  python scripts/fetch_arxiv_research.py --query "quantum algorithms" --limit 10
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List
from urllib.parse import quote_plus

import feedparser


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Fetch research papers from arXiv.")
    parser.add_argument("--query", required=True, help="The search query for arXiv.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of entries to fetch.")
    parser.add_argument("--output", help="Optional path to write JSON output.")
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout.")
    return parser.parse_args()


def fetch_entries(query: str, limit: int) -> feedparser.FeedParserDict:
    """Fetches entries from the arXiv API."""
    base = "https://export.arxiv.org/api/query"
    url = f"{base}?search_query={quote_plus(query)}&start=0&max_results={limit}"
    return feedparser.parse(url)


def process_entries(feed: feedparser.FeedParserDict) -> List[Dict[str, str]]:
    """Processes the feed entries into a list of dictionaries."""
    results: List[Dict[str, str]] = []
    for entry in feed.entries:
        authors = ", ".join(author.name for author in entry.authors)
        results.append(
            {
                "title": entry.title,
                "authors": authors,
                "published": entry.published,
                "summary": entry.summary,
                "link": entry.link,
            }
        )
    return results


def main() -> None:
    """Main function to fetch and process arXiv data."""
    args = parse_args()
    feed = fetch_entries(args.query, args.limit)
    results = process_entries(feed)

    payload = {"query": args.query, "results": results}

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(payload, fh, indent=2)

    if args.json or not args.output:
        print(json.dumps(payload, indent=2))
    else:
        print(f"[info] Wrote {len(results)} entries to {args.output}")


if __name__ == "__main__":
    main()
