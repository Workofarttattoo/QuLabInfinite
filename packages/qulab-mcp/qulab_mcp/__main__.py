"""Entry point: python -m qulab_mcp  OR  qulab-mcp (via pyproject script)."""
import asyncio
import sys


def main() -> None:
    from qulab_mcp.server import run
    asyncio.run(run())


if __name__ == "__main__":
    main()
