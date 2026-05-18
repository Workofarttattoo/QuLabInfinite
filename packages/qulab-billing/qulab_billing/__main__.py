"""
QuLab Billing Proxy entry point.

Usage:
  qulab-billing                         # default port 8080
  qulab-billing --port 9000
  qulab-billing --backend http://localhost:8000
  qulab-billing --host 127.0.0.1 --port 8080
"""
from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    p = argparse.ArgumentParser(
        description="QuLab Billing Proxy — pay-per-use MCP gateway",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("QULAB_BILLING_PORT", "8080")),
        help="Port to listen on (default: 8080)",
    )
    p.add_argument(
        "--host",
        default=os.environ.get("QULAB_BILLING_HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)",
    )
    p.add_argument(
        "--backend",
        default=None,
        help="Backend MCP server URL (default: http://127.0.0.1:8000)",
    )
    args = p.parse_args()

    if args.backend:
        os.environ["QULAB_BACKEND_URL"] = args.backend

    uvicorn.run(
        "qulab_billing.app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
