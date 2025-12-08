#!/usr/bin/env python3
"""
Compute adsorption energy using TensorNet PES (PyG backend).

ΔE_ads = E(slab+adsorbate) - E(slab) - E(adsorbate)

Usage:
    MATGL_BACKEND=PYG python scripts/matgl_adsorption_energy.py \
        --slab slab.cif --ads ads.cif --combined slab_plus_ads.cif

Options:
    --model TensorNet-MatPES-PBE-v2025.1-PES (default)
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


def main():
    parser = argparse.ArgumentParser(description="Adsorption energy via MatGL TensorNet PES (PyG backend).")
    parser.add_argument("--slab", required=True, help="Path to slab CIF.")
    parser.add_argument("--ads", required=True, help="Path to adsorbate CIF.")
    parser.add_argument("--combined", required=True, help="Path to combined slab+adsorbate CIF.")
    parser.add_argument(
        "--model",
        default="TensorNet-MatPES-PBE-v2025.1-PES",
        help="Pretrained model name (default TensorNet-MatPES-PBE-v2025.1-PES).",
    )
    args = parser.parse_args()

    model = load_model(args.model)
    e_slab = energy_only(Structure.from_file(args.slab), model)
    e_ads = energy_only(Structure.from_file(args.ads), model)
    e_combined = energy_only(Structure.from_file(args.combined), model)

    delta = e_combined - e_slab - e_ads

    print(f"model: {args.model}")
    print(f"E_slab_eV: {e_slab:.6f}")
    print(f"E_ads_eV: {e_ads:.6f}")
    print(f"E_combined_eV: {e_combined:.6f}")
    print(f"delta_E_ads_eV: {delta:.6f}")


if __name__ == "__main__":
    main()
