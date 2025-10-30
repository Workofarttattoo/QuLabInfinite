#!/usr/bin/env python3
"""Convenience wrapper for the chemistry lab CLI."""

from __future__ import annotations

import sys

from chemistry_lab.cli import main as chemistry_cli_main


def main(argv: list[str] | None = None) -> int:
    return chemistry_cli_main(argv or sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
