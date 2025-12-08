#!/usr/bin/env python3
"""
Minimal PyG TensorNet PES inference.

Loads a CIF, converts to graph with the pretrained TensorNet-MatPES-PBE-v2025.1-PES,
and prints total energy (eV), per-atom energy (eV/atom), forces, and stress.

Usage:
    MATGL_BACKEND=PYG python scripts/matgl_pyg_infer.py --cif path/to/struct.cif

Optionally override model name:
    MATGL_BACKEND=PYG python scripts/matgl_pyg_infer.py --cif struct.cif --model TensorNet-MatPES-r2SCAN-v2025.1-PES
"""

import argparse
import numpy as np
import torch
from pymatgen.core import Structure

import matgl
from matgl.ext.pymatgen import Structure2Graph


def load_model(name: str):
    model = matgl.load_model(name)
    return model


def build_graph(struct: Structure, model) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    conv = Structure2Graph(element_types=model.model.element_types, cutoff=model.model.cutoff)
    g, lat, state = conv.get_graph(struct)
    return g, torch.tensor(lat, dtype=matgl.float_th), torch.tensor(state, dtype=matgl.float_th)


def main():
    parser = argparse.ArgumentParser(description="Run MatGL TensorNet PES inference (PyG backend).")
    parser.add_argument("--cif", required=True, help="Path to CIF file.")
    parser.add_argument(
        "--model",
        default="TensorNet-MatPES-PBE-v2025.1-PES",
        help="Pretrained model name (default: TensorNet-MatPES-PBE-v2025.1-PES).",
    )
    args = parser.parse_args()

    struct = Structure.from_file(args.cif)
    model = load_model(args.model)
    g, lat, state = build_graph(struct, model)

    energy, forces, stresses, _ = model(g, lat, state)

    energy = energy.detach().cpu().numpy().item()
    forces = forces.detach().cpu().numpy()
    stresses = stresses.detach().cpu().numpy()

    print(f"model: {args.model}")
    print(f"total_energy_eV: {energy:.6f}")
    print(f"per_atom_energy_eV: {energy / len(struct):.6f}")
    print("forces_eV_per_A (n_atoms x 3):")
    np.set_printoptions(precision=6, suppress=True)
    print(forces)
    print("stress_GPa (3x3):")
    print(stresses)


if __name__ == "__main__":
    main()
