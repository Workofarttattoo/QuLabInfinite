#!/usr/bin/env python3
"""
List available MatGL pretrained model names bundled with the installed package.

Usage:
    python scripts/matgl_list_models.py

Notes:
- This simply enumerates the directory names under matgl.pretrained_models.
- Names can be passed directly to matgl.load_model, e.g., matgl.load_model("M3GNet-MP-2018.6.1-Eform").
"""

from __future__ import annotations

import sys
from importlib.resources import files


def main():
    try:
        models_dir = files("matgl") / "pretrained_models"
    except Exception as exc:  # pragma: no cover - defensive for missing pkg
        sys.stderr.write(f"Unable to locate matgl pretrained models: {exc}\n")
        sys.exit(1)

    if not models_dir.exists():
        sys.stderr.write(f"Pretrained models directory not found: {models_dir}\n")
        sys.exit(1)

    names = sorted([p.name for p in models_dir.iterdir() if p.is_dir()])
    for name in names:
        print(name)


if __name__ == "__main__":
    main()
