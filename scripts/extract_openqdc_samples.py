#!/usr/bin/env python3
"""Extract small CSV samples from locally cached OpenQDC datasets."""
from __future__ import annotations

import argparse
import csv
import itertools
import pathlib
from typing import Iterable

import numpy as np

try:
    import openqdc
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "openqdc Python package is required. Install via `pip install openqdc`."
    ) from exc

CACHE_DIR = pathlib.Path.home() / ".cache" / "openqdc"
OUTPUT_DIR = pathlib.Path("data/raw/quantum")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "qmugs": (openqdc.QMugs, "qmugs_sample.csv"),
    "spice": (openqdc.Spice, "spice_sample.csv"),
    "qm7x": (openqdc.QM7X, "qm7x_sample.csv"),
}

DEFAULT_SAMPLES = 64


def load_dataset(name: str, cls):
    cache_path = CACHE_DIR / name
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Expected OpenQDC cache at {cache_path}. Run `openqdc download {name}` first."
        )
    return cls(cache_dir=str(cache_path), skip_statistics=True)


def format_energy(values) -> tuple[float | None, float | None]:
    if values is None:
        return (None, None)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr[None]
    if arr.size == 0:
        return (None, None)
    if arr.size == 1:
        return (float(arr[0]), None)
    return (float(arr[0]), float(arr[1]))


def write_dataset(dataset, out_file: pathlib.Path, n_samples: int) -> None:
    with out_file.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "index",
                "name",
                "subset",
                "n_atoms",
                "energy_0",
                "energy_1",
                "formation_energy_0",
                "formation_energy_1",
            ]
        )
        for idx, sample in enumerate(itertools.islice(dataset, n_samples)):
            e0 = format_energy(sample.get("energies"))
            fe = format_energy(sample.get("formation_energies"))
            writer.writerow(
                [
                    idx,
                    sample.get("name"),
                    sample.get("subset"),
                    int(sample.get("n_atoms", 0)),
                    e0[0],
                    e0[1],
                    fe[0],
                    fe[1],
                ]
            )
    print(f"Wrote {out_file} ({out_file.stat().st_size/1024:.1f} KiB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASETS.keys()),
        help="Subset of datasets to export (default: all available)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="Number of rows per dataset (default: 64)",
    )
    args = parser.parse_args()

    for name in args.datasets:
        if name not in DATASETS:
            print(f"[warn] Unknown dataset '{name}', skipping")
            continue
        cls, filename = DATASETS[name]
        try:
            dataset = load_dataset(name, cls)
        except FileNotFoundError as exc:
            print(f"[warn] {exc}")
            continue
        out_path = OUTPUT_DIR / filename
        write_dataset(dataset, out_path, args.samples)


if __name__ == "__main__":
    main()
