"""
QuLabInfinite CLI.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="qulab",
        description="QuLabInfinite — Infinite Scientific Simulation Platform",
    )
    subparsers = parser.add_subparsers(dest="command")

    # qulab list
    list_parser = subparsers.add_parser("list", help="List available labs")
    list_parser.add_argument("--category", "-c", help="Filter by category")
    list_parser.add_argument("--medical", action="store_true", help="Show only medical labs")

    # qulab serve
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", "-p", type=int, default=8000, help="Port")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # qulab run
    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument("lab", help="Lab name")
    run_parser.add_argument("--spec", "-s", help="JSON experiment spec")
    run_parser.add_argument("--file", "-f", help="JSON file with experiment spec")

    # qulab info
    info_parser = subparsers.add_parser("info", help="Show platform info")

    args = parser.parse_args()

    if args.command == "list":
        from qulab.core.registry import LabRegistry

        registry = LabRegistry()
        registry.auto_discover()

        if args.medical:
            labs = registry.list_medical()
            print(f"\n🏥 Medical Labs ({len(labs)}):")
        elif args.category:
            labs = registry.list_by_category(args.category)
            print(f"\n📂 {args.category} Labs ({len(labs)}):")
        else:
            labs = registry.list_labs()
            print(f"\n🔬 All Labs ({len(labs)}):")

        for lab in labs:
            meta = registry.get_metadata(lab)
            desc = f" — {meta.description}" if meta and meta.description else ""
            icon = "🏥" if meta and meta.is_medical else "🔬"
            print(f"  {icon} {lab}{desc}")

    elif args.command == "serve":
        import uvicorn

        uvicorn.run(
            "qulab.api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )

    elif args.command == "run":
        from qulab.core.simulator import UnifiedSimulator

        sim = UnifiedSimulator()
        spec = {}
        if args.spec:
            spec = json.loads(args.spec)
        elif args.file:
            with open(args.file) as f:
                spec = json.load(f)

        results = sim.run_simulation(args.lab, spec)
        print(json.dumps(results, indent=2, default=str))

    elif args.command == "info":
        from qulab.core.simulator import UnifiedSimulator

        sim = UnifiedSimulator()
        summary = sim.summary()
        print("\n═══════════════════════════════════════")
        print("  QuLabInfinite — Scientific Platform")
        print("═══════════════════════════════════════")
        print(f"  Version:    {summary.get('version', '1.0.0')}")
        print(f"  Total Labs: {summary['total_labs']}")
        print(f"  Categories: {len(summary['categories'])}")
        print(f"  Medical:    {len(summary['medical_labs'])}")
        print("═══════════════════════════════════════\n")

        for cat, labs in sorted(summary["categories"].items()):
            print(f"  [{cat}] {len(labs)} labs")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
