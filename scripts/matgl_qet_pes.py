#!/usr/bin/env python3
"""
QET universal PES inference (energy/forces/stress) using MatGL (DGL backend).

Example:
    MATGL_BACKEND=DGL python scripts/matgl_qet_pes.py --cif struct.cif

Outputs:
- total_energy_eV
- per_atom_energy_eV
- forces_eV_per_A (n_atoms x 3)
- stress_GPa (3x3)
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
    os.environ.setdefault("MATGL_BACKEND", "DGL")
    return matgl.load_model(name)


def build_graph(struct: Structure, model):
    conv = Structure2Graph(element_types=model.model.element_types, cutoff=model.model.cutoff)
    g, lat, state = conv.get_graph(struct)
    return g, torch.tensor(lat, dtype=matgl.float_th), torch.tensor(state, dtype=matgl.float_th)


def main():
    parser = argparse.ArgumentParser(description="Run QET universal PES (DGL backend) via MatGL.")
    parser.add_argument("--cif", required=True, help="Path to CIF file.")
    parser.add_argument(
        "--model",
        default="QET-MatQ-PES",
        help="Pretrained model name (default QET-MatQ-PES).",
    )
    args = parser.parse_args()

    struct = Structure.from_file(args.cif)
    model = load_model(args.model)
    g, lat, state = build_graph(struct, model)

    energy, forces, stresses, _ = model(g, lat, state)

    energy = energy.detach().cpu().numpy().item()
    forces = forces.detach().cpu().numpy()
    stresses = stresses.detach().cpu().numpy()

    np.set_printoptions(precision=6, suppress=True)
    print(f"backend: {os.environ.get('MATGL_BACKEND', 'DGL')}")
    print(f"model: {args.model}")
    print(f"total_energy_eV: {energy:.6f}")
    print(f"per_atom_energy_eV: {energy / len(struct):.6f}")
    print("forces_eV_per_A (n_atoms x 3):")
    print(forces)
    print("stress_GPa (3x3):")
    print(stresses)


if __name__ == "__main__":
    main()
