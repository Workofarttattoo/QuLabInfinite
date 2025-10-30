#!/usr/bin/env python3
"""Compute SHA-256 hashes for top-level files inside the OpenQDC cache.

This avoids shipping the multi-GB raw datasets in the repository while still
providing reproducible fingerprints of the local cache state.
"""
from __future__ import annotations

import hashlib
import pathlib
from typing import Iterable

CACHE_ROOT = pathlib.Path.home() / ".cache" / "openqdc"
OUTPUT = pathlib.Path("data/raw/quantum/openqdc_cache_hashes.txt")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def iter_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not CACHE_ROOT.exists():
        raise SystemExit(f"No OpenQDC cache found at {CACHE_ROOT}. Run `openqdc download ...` first.")

    with OUTPUT.open("w") as out:
        out.write(f"# SHA-256 hashes for files under {CACHE_ROOT}\n")
        for file_path in iter_files(CACHE_ROOT):
            digest = sha256(file_path)
            out.write(f"{digest}  {file_path}\n")
    print(f"Wrote {OUTPUT} (${OUTPUT.stat().st_size/1024:.1f} KiB)")


if __name__ == "__main__":
    main()
