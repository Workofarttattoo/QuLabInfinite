"""
Entry point for the ECH0 Autonomous Research Service.

Usage:
    python -m qulab_research [--port PORT] [--host HOST]
                             [--lab-root PATH] [--backend URL]
    qulab-research           (via pyproject.toml script)
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(
        description="ECH0 Autonomous Research Service"
    )
    p.add_argument("--port", type=int, default=8090, help="Port to listen on (default: 8090)")
    p.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    p.add_argument(
        "--lab-root",
        default=None,
        help="Path to QuLabInfinite repository root (auto-detected if omitted)",
    )
    p.add_argument(
        "--backend",
        default="http://127.0.0.1:8000",
        help="qulab-mcp backend URL (default: http://127.0.0.1:8000)",
    )
    args = p.parse_args()

    if args.lab_root:
        os.environ["QULAB_LAB_ROOT"] = args.lab_root

    if args.backend:
        os.environ["QULAB_BACKEND_URL"] = args.backend

    import uvicorn  # type: ignore

    uvicorn.run(
        "qulab_research.app:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
