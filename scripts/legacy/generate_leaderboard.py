#!/usr/bin/env python3
"""
Generate SIMTEST leaderboard from aggregated test results.

Usage:
    python scripts/generate_leaderboard.py --results results/ --out leaderboard.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Any
from datetime import datetime

from scripts.simtest_cli import _expand_globs, _load_json

def _repo_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, os.pardir))


def compute_certification_level(stats: Dict[str, Any], total_mandatory: int) -> str:
    """Determine certification level based on pass rates."""
    pass_rate = stats.get("pass_rate", 0.0)
    domains_covered = stats.get("domains_covered", set())
    
    if pass_rate >= 0.95 and len(domains_covered) >= 5:
        return "gold"
    elif pass_rate >= 0.90 and len(domains_covered) >= 2:
        return "silver"
    elif pass_rate >= 0.80:
        return "bronze"
    return "unqualified"


def aggregate_results(result_files: List[str]) -> Dict[str, Any]:
    """Aggregate test results by engine."""
    engine_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "total": 0,
        "pass": 0,
        "fail": 0,
        "error": 0,
        "by_domain": defaultdict(lambda: {"total": 0, "pass": 0, "fail": 0, "error": 0}),
        "domains_covered": set(),
        "tests": [],
        "avg_duration_s": 0.0,
        "last_run": None
    })
    
    mandatory_tests = {
        "materials_si_formation_energy_v1",
        "chemistry_h2o_formation_energy_v1",
        "mech_cantilever_304ss_v1",
        "thermal_rod_transient_v1",
        "cfd_lid_driven_cavity_Re100_v1"
    }
    
    for result_path in result_files:
        try:
            result = _load_json(result_path)
            engine_info = result.get("engine", {})
            engine_name = engine_info.get("name", "unknown")
            engine_version = engine_info.get("version", "unknown")
            engine_key = f"{engine_name}:{engine_version}"
            
            stats = engine_stats[engine_key]
            stats["engine_name"] = engine_name
            stats["engine_version"] = engine_version
            
            # Update domain coverage
            test_id = result.get("test_id", "")
            domain = test_id.split("_")[0] if "_" in test_id else "unknown"
            stats["domains_covered"].add(domain)
            
            # Update counts
            status = result.get("status", "error")
            stats[status] = stats.get(status, 0) + 1
            stats["total"] += 1
            stats["by_domain"][domain][status] += 1
            stats["by_domain"][domain]["total"] += 1
            
            # Track mandatory tests
            if test_id in mandatory_tests:
                stats["mandatory_passed"] = stats.get("mandatory_passed", 0)
                if status == "pass":
                    stats["mandatory_passed"] += 1
                stats["mandatory_total"] = stats.get("mandatory_total", 0) + 1
            
            # Duration
            duration = result.get("duration_s", 0.0)
            if duration > 0:
                stats["avg_duration_s"] = (
                    (stats["avg_duration_s"] * (stats["total"] - 1) + duration) / stats["total"]
                )
            
            # Last run timestamp
            prov = result.get("provenance", {})
            timestamp_str = prov.get("timestamp_utc", "")
            if timestamp_str:
                if stats["last_run"] is None or timestamp_str > stats["last_run"]:
                    stats["last_run"] = timestamp_str
            
            # Track test
            stats["tests"].append({
                "test_id": test_id,
                "status": status,
                "duration_s": duration
            })
            
        except Exception as e:
            print(f"Error processing {result_path}: {e}", file=sys.stderr)
            continue
    
    # Compute pass rates and certification
    leaderboard: List[Dict[str, Any]] = []
    for engine_key, stats in engine_stats.items():
        stats["pass_rate"] = stats["pass"] / stats["total"] if stats["total"] > 0 else 0.0
        stats["mandatory_pass_rate"] = (
            stats["mandatory_passed"] / stats["mandatory_total"]
            if stats.get("mandatory_total", 0) > 0 else 0.0
        )
        stats["domains_covered"] = list(stats["domains_covered"])
        stats["by_domain"] = dict(stats["by_domain"])
        
        # Compute certification
        mandatory_total = stats.get("mandatory_total", 5)
        stats["certification_level"] = compute_certification_level(stats, mandatory_total)
        
        leaderboard.append(stats)
    
    # Sort by certification level (gold > silver > bronze) then pass rate
    cert_order = {"gold": 3, "silver": 2, "bronze": 1, "unqualified": 0}
    leaderboard.sort(
        key=lambda x: (cert_order.get(x.get("certification_level", "unqualified"), 0), x.get("pass_rate", 0.0)),
        reverse=True
    )
    
    return leaderboard


def main():
    parser = argparse.ArgumentParser(description="Generate SIMTEST leaderboard")
    parser.add_argument("--results", nargs="*", default=["results/**/*.json"], help="Result files/dirs/globs")
    parser.add_argument("--out", default="leaderboard.json", help="Output JSON path")
    args = parser.parse_args()
    
    repo_root = _repo_root()
    result_files = _expand_globs(args.results) if args.results else []
    
    if not result_files:
        print("No result files found", file=sys.stderr)
        return 1
    
    print(f"Processing {len(result_files)} result files...", file=sys.stderr)
    leaderboard = aggregate_results(result_files)
    
    output = {
        "version": "1.0.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_engines": len(leaderboard),
        "entries": leaderboard
    }
    
    out_path = os.path.join(repo_root, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, indent=2, fp=f)
    
    print(f"Generated leaderboard with {len(leaderboard)} engines", file=sys.stderr)
    print(f"  Gold: {sum(1 for e in leaderboard if e.get('certification_level') == 'gold')}", file=sys.stderr)
    print(f"  Silver: {sum(1 for e in leaderboard if e.get('certification_level') == 'silver')}", file=sys.stderr)
    print(f"  Bronze: {sum(1 for e in leaderboard if e.get('certification_level') == 'bronze')}", file=sys.stderr)
    print(f"Wrote: {out_path}", file=sys.stderr)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

