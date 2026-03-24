import logging
"""Canonical runtime entrypoint for QuLabInfinite."""

import argparse
import json
from typing import Any, Dict

from core.unified_simulator import get_simulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QuLabInfinite runtime")
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("list", help="List all registered tools")

    run_parser = subcommands.add_parser("run", help="Run a tool and emit canonical JSON artifact")
    run_parser.add_argument("tool", help="Registered tool name")
    run_parser.add_argument("--payload", default="{}", help="JSON payload for the tool")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    simulator = get_simulator()

    if args.command == "list":
        logging.info(json.dumps(simulator.list_labs(), sort_keys=True, indent=2))
        return

    payload: Dict[str, Any] = json.loads(args.payload)
    logging.info(simulator.run_simulation_artifact(args.tool, payload))


if __name__ == "__main__":
    main()
