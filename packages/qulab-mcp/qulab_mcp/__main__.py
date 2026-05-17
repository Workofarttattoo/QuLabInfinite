"""Entry point: python -m qulab_mcp  OR  qulab-mcp (via pyproject script).

Usage:
  qulab-mcp                          # stdio mode (Claude Desktop / claude CLI)
  qulab-mcp --http                   # HTTP+SSE on 127.0.0.1:8000
  qulab-mcp --http --host 0.0.0.0    # expose on all interfaces
  qulab-mcp --http --port 9000       # custom port
  QULAB_API_KEY=secret qulab-mcp --http  # require Bearer token
"""
import argparse
import asyncio
import sys


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="qulab-mcp", description="QuLab Infinite MCP server")
    p.add_argument("--http", action="store_true", help="Run HTTP+SSE transport (default: stdio)")
    p.add_argument("--host", default="127.0.0.1", help="HTTP host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    return p.parse_args()


def main() -> None:
    args = _parse()
    if args.http:
        from qulab_mcp.server import run_http
        asyncio.run(run_http(host=args.host, port=args.port))
    else:
        from qulab_mcp.server import run
        asyncio.run(run())


if __name__ == "__main__":
    main()
