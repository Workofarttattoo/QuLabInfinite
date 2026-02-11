#!/usr/bin/env python3
from __future__ import annotations
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

catalog_inspector - Part of Materials Lab
"""

"""
Materials catalogue inspector.

Provides lightweight analytics over the QuLab Infinite materials database:
- summary statistics
- property distributions
- top-N queries
- data quality checks for missing/zero values

Because the database is intended to be factual, this tool does not modify any
records; it only reports metrics so humans can cross-check against reference
sources. Use it interactively or from CI to monitor data quality.
"""


import argparse
import json
from statistics import mean
from typing import Dict, Iterable, List, Tuple

from materials_database import MaterialProperties, MaterialsDatabase


NUMERIC_PROPERTIES = [
    "density",
    "youngs_modulus",
    "tensile_strength",
    "yield_strength",
    "thermal_conductivity",
    "specific_heat",
    "thermal_expansion",
    "electrical_resistivity",
    "electrical_conductivity",
    "cost_per_kg",
]


def summarise_catalogue(db: MaterialsDatabase) -> Dict[str, object]:
    """Return overall statistics for the catalogue."""
    stats = db.get_statistics()
    category_counts = stats["categories"]

    property_stats = {
        prop: _property_stats(db.materials.values(), prop)
        for prop in NUMERIC_PROPERTIES
    }

    return {
        "total_materials": stats["total_materials"],
        "categories": category_counts,
        "property_stats": property_stats,
    }


def _property_stats(materials: Iterable[MaterialProperties], prop: str) -> Dict[str, float]:
    values = [
        getattr(mat, prop)
        for mat in materials
        if getattr(mat, prop, 0) not in (None, 0)
    ]
    if not values:
        return {"count": 0}

    return {
        "count": len(values),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(mean(values)),
    }


def top_materials(db: MaterialsDatabase, prop: str, limit: int, descending: bool = True) -> List[Dict[str, object]]:
    """Return top materials by property."""
    if prop not in NUMERIC_PROPERTIES:
        raise ValueError(f"Unsupported property '{prop}'. Choose from {NUMERIC_PROPERTIES}.")

    materials = [
        (getattr(mat, prop, None), mat)
        for mat in db.materials.values()
        if getattr(mat, prop, None) not in (None, 0)
    ]
    materials.sort(key=lambda item: item[0], reverse=descending)

    top = []
    for value, mat in materials[:limit]:
        top.append({
            "name": mat.name,
            "category": mat.category,
            "subcategory": mat.subcategory,
            prop: float(value),
            "density": float(mat.density),
            "tensile_strength": float(mat.tensile_strength),
            "thermal_conductivity": float(mat.thermal_conductivity),
            "cost_per_kg": float(mat.cost_per_kg),
        })
    return top


def missing_value_report(db: MaterialsDatabase) -> Dict[str, int]:
    """Count zero/None entries for critical properties."""
    counts: Dict[str, int] = {prop: 0 for prop in NUMERIC_PROPERTIES}

    for mat in db.materials.values():
        for prop in NUMERIC_PROPERTIES:
            value = getattr(mat, prop, None)
            if value is None or value == 0:
                counts[prop] += 1

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the QuLab Infinite materials catalogue.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--summary", action="store_true", help="Print overall catalogue summary.")
    parser.add_argument("--top", metavar="PROPERTY", help=f"Show top materials for property in {NUMERIC_PROPERTIES}.")
    parser.add_argument("--limit", type=int, default=10, help="Limit for --top query (default: 10).")
    parser.add_argument("--ascending", action="store_true", help="Use ascending sort for --top.")
    parser.add_argument("--missing", action="store_true", help="Report missing/zero value counts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db = MaterialsDatabase()

    output: Dict[str, object] = {}

    if args.summary:
        output["summary"] = summarise_catalogue(db)

    if args.top:
        output["top"] = top_materials(db, args.top, args.limit, descending=not args.ascending)

    if args.missing:
        output["missing_values"] = missing_value_report(db)

    if not output:
        output["summary"] = summarise_catalogue(db)

    if args.json:
        print(json.dumps(output, indent=2))
    else:
        _print_human_readable(output)


def _print_human_readable(data: Dict[str, object]) -> None:
    if "summary" in data:
        summary = data["summary"]
        print("=== Catalogue Summary ===")
        print(f"Total materials: {summary['total_materials']}")
        print("By category:")
        for category, count in summary["categories"].items():
            print(f"  {category:<15} {count:>6}")
        print("\nKey property statistics:")
        for prop, stats in summary["property_stats"].items():
            if stats.get("count", 0) == 0:
                continue
            print(f"  {prop:<25} n={stats['count']:<5} min={stats['min']:.3g}  mean={stats['mean']:.3g}  max={stats['max']:.3g}")
        print()

    if "top" in data:
        print("=== Top Materials ===")
        for entry in data["top"]:
            name = entry.pop("name")
            print(f"- {name}")
            for key, value in entry.items():
                print(f"    {key}: {value}")
        print()

    if "missing_values" in data:
        print("=== Missing or Zero Values ===")
        for prop, count in data["missing_values"].items():
            print(f"  {prop:<25} {count}")


if __name__ == "__main__":
    main()
