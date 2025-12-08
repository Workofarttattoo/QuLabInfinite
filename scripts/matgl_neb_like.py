#!/usr/bin/env python3
"""
Minimal NEB-like barrier estimate using TensorNet PES (PyG backend).

Given initial and final CIFs, linearly interpolate images, evaluate PES energies,
and report a crude barrier (max(image) - min(initial, final)).

Usage:
    MATGL_BACKEND=PYG python scripts/matgl_neb_like.py \
        --initial initial.cif --final final.cif \
        --n-images 5

This is a quick heuristic, not a replacement for full NEB/DFT.
"""

import argparse
import numpy as np
import torch
from pymatgen.core import Structure

import matgl
from matgl.ext.pymatgen import Structure2Graph


def load_model(name: str):
    return matgl.load_model(name)


def energy_only(struct: Structure, model) -> float:
    conv = Structure2Graph(element_types=model.model.element_types, cutoff=model.model.cutoff)
    g, lat, state = conv.get_graph(struct)
    energy, _, _, _ = model(g, torch.tensor(lat, dtype=matgl.float_th), torch.tensor(state, dtype=matgl.float_th))
    return float(energy.detach().cpu().numpy())


def interpolate_structures(struct_a: Structure, struct_b: Structure, n_images: int) -> list[Structure]:
    if len(struct_a) != len(struct_b):
        raise ValueError("Initial and final structures must have the same number of atoms.")
    images = []
    for t in np.linspace(0, 1, n_images + 2):  # include endpoints
        coords = (1 - t) * struct_a.frac_coords + t * struct_b.frac_coords
        img = Structure(struct_a.lattice, struct_a.species, coords, coords_are_cartesian=False)
        images.append(img)
    return images


def main():
    parser = argparse.ArgumentParser(description="NEB-like barrier estimate via MatGL TensorNet PES (PyG backend).")
    parser.add_argument("--initial", required=True, help="Initial CIF.")
    parser.add_argument("--final", required=True, help="Final CIF.")
    parser.add_argument("--n-images", type=int, default=5, help="Number of interpolated images (default 5).")
    parser.add_argument(
        "--model",
        default="TensorNet-MatPES-PBE-v2025.1-PES",
        help="Pretrained model name (default TensorNet-MatPES-PBE-v2025.1-PES).",
    )
    args = parser.parse_args()

    model = load_model(args.model)
    s0 = Structure.from_file(args.initial)
    s1 = Structure.from_file(args.final)
    images = interpolate_structures(s0, s1, args.n_images)

    energies = [energy_only(s, model) for s in images]
    e0, e1 = energies[0], energies[-1]
    barrier = max(energies) - min(e0, e1)

    print(f"model: {args.model}")
    for i, e in enumerate(energies):
        tag = "initial" if i == 0 else ("final" if i == len(energies) - 1 else f"img{i}")
        print(f"{tag}: {e:.6f} eV")
    print(f"barrier_estimate_eV: {barrier:.6f}")


if __name__ == "__main__":
    main()
