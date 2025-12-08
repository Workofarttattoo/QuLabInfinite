#!/usr/bin/env python3
"""
Formation/total energy inference with MatGL (DGL backend).

Defaults to the pretrained M3GNet formation energy model. You can also point to
other MatGL models such as MEGNet formation energy, QET universal potential, or
CHGNet PES variants.

Examples:
    MATGL_BACKEND=DGL python scripts/matgl_dgl_form_energy.py --cif struct.cif
    MATGL_BACKEND=DGL python scripts/matgl_dgl_form_energy.py --cif struct.cif \
        --model MEGNet-MP-2018.6.1-Eform
    MATGL_BACKEND=DGL python scripts/matgl_dgl_form_energy.py --cif struct.cif \
        --model QET-MatQ-PES
    MATGL_BACKEND=DGL python scripts/matgl_dgl_form_energy.py --cif struct.cif \
        --model CHGNet-MatPES-r2SCAN-2025.2.10-2.7M-PES

Notes:
- Set MATGL_BACKEND=DGL to force the DGL implementation.
- Energy units are eV; per-atom energy is eV/atom.
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import torch
from pymatgen.core import Structure

import matgl
from matgl.ext.pymatgen import Structure2Graph


def load_model(name: str):
    # Ensure we stay on the DGL backend unless user overrides externally.
    os.environ.setdefault("MATGL_BACKEND", "DGL")
    return matgl.load_model(name)


def energy_only(struct: Structure, model) -> float:
    conv = Structure2Graph(element_types=model.model.element_types, cutoff=model.model.cutoff)
    g, lat, state = conv.get_graph(struct)
    energy, _, _, _ = model(g, torch.tensor(lat, dtype=matgl.float_th), torch.tensor(state, dtype=matgl.float_th))
    return float(energy.detach().cpu().numpy())


def main():
    parser = argparse.ArgumentParser(description="Run MatGL formation/total energy inference (DGL backend).")
    parser.add_argument("--cif", required=True, help="Path to CIF file.")
    parser.add_argument(
        "--model",
        default="M3GNet-MP-2018.6.1-Eform",
        help=(
            "Pretrained model name. Examples: "
            "M3GNet-MP-2018.6.1-Eform (default), "
            "MEGNet-MP-2018.6.1-Eform, "
            "QET-MatQ-PES, "
            "CHGNet-MatPES-r2SCAN-2025.2.10-2.7M-PES, "
            "CHGNet-MatPES-PBE-2025.2.10-2.7M-PES."
        ),
    )
    args = parser.parse_args()

    struct = Structure.from_file(args.cif)
    model = load_model(args.model)
    energy = energy_only(struct, model)

    np.set_printoptions(precision=6, suppress=True)
    print(f"backend: {os.environ.get('MATGL_BACKEND', 'DGL')}")
    print(f"model: {args.model}")
    print(f"total_energy_eV: {energy:.6f}")
    print(f"per_atom_energy_eV: {energy / len(struct):.6f}")


if __name__ == "__main__":
    main()
