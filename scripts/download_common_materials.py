#!/usr/bin/env python3
"""
Download Common Materials from Materials Project

Downloads a curated set of 100 common, well-characterized materials
for validation and testing purposes.

Usage:
    export MP_API_KEY='your-api-key-here'
    python download_common_materials.py

Output:
    - materials_project_100_common.json: Full dataset
    - materials_project_summary.txt: Human-readable summary
"""

import sys
import os
from pathlib import Path
import json
import logging
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from materials_lab.materials_project_client import MaterialsProjectClient, MPMaterialData
from materials_lab.materials_database import MaterialProperties


# Curated list of 100 common materials (by Materials Project ID)
COMMON_MATERIALS = [
    # === PURE ELEMENTS (25) ===
    ("mp-149", "Silicon", "semiconductor"),
    ("mp-13", "Iron", "metal"),
    ("mp-30", "Copper", "metal"),
    ("mp-23", "Aluminum", "metal"),
    ("mp-72", "Titanium", "metal"),
    ("mp-911", "Nickel", "metal"),
    ("mp-54", "Tungsten", "metal"),
    ("mp-79", "Chromium", "metal"),
    ("mp-8", "Magnesium", "metal"),
    ("mp-48", "Gold", "metal"),
    ("mp-126", "Silver", "metal"),
    ("mp-124", "Platinum", "metal"),
    ("mp-35", "Zinc", "metal"),
    ("mp-96", "Molybdenum", "metal"),
    ("mp-90", "Zirconium", "metal"),
    ("mp-47", "Lead", "metal"),
    ("mp-20", "Cobalt", "metal"),
    ("mp-101", "Vanadium", "metal"),
    ("mp-81", "Manganese", "metal"),
    ("mp-11", "Carbon (graphite)", "ceramic"),
    ("mp-66", "Diamond", "ceramic"),
    ("mp-568842", "Boron", "ceramic"),
    ("mp-12", "Germanium", "semiconductor"),
    ("mp-134", "Tin", "metal"),
    ("mp-153", "Tantalum", "metal"),

    # === COMMON OXIDES (20) ===
    ("mp-1143", "SiO2 (Quartz)", "ceramic"),
    ("mp-19770", "Al2O3 (Alumina/Sapphire)", "ceramic"),
    ("mp-1245", "TiO2 (Rutile)", "ceramic"),
    ("mp-2657", "Fe2O3 (Hematite)", "ceramic"),
    ("mp-18905", "Fe3O4 (Magnetite)", "ceramic"),
    ("mp-1487", "ZnO", "ceramic"),
    ("mp-1265", "CuO", "ceramic"),
    ("mp-19399", "MgO (Periclase)", "ceramic"),
    ("mp-1216", "NiO", "ceramic"),
    ("mp-1096", "Cr2O3", "ceramic"),
    ("mp-1840", "ZrO2 (Zirconia)", "ceramic"),
    ("mp-19306", "CeO2 (Ceria)", "ceramic"),
    ("mp-3749", "SnO2", "ceramic"),
    ("mp-3827", "WO3", "ceramic"),
    ("mp-19399", "CaO", "ceramic"),
    ("mp-1792", "V2O5", "ceramic"),
    ("mp-19395", "MnO", "ceramic"),
    ("mp-22598", "CoO", "ceramic"),
    ("mp-19326", "Y2O3 (Yttria)", "ceramic"),
    ("mp-19306", "BaO", "ceramic"),

    # === SEMICONDUCTORS (15) ===
    ("mp-2534", "GaAs", "semiconductor"),
    ("mp-804", "GaN", "semiconductor"),
    ("mp-20305", "InP", "semiconductor"),
    ("mp-2490", "CdTe", "semiconductor"),
    ("mp-10695", "InAs", "semiconductor"),
    ("mp-10044", "AlAs", "semiconductor"),
    ("mp-8062", "GaP", "semiconductor"),
    ("mp-10695", "InSb", "semiconductor"),
    ("mp-2691", "CdS", "semiconductor"),
    ("mp-1138", "ZnS", "semiconductor"),
    ("mp-2133", "ZnSe", "semiconductor"),
    ("mp-1018063", "CdSe", "semiconductor"),
    ("mp-9754", "PbS", "semiconductor"),
    ("mp-19717", "PbSe", "semiconductor"),
    ("mp-21276", "PbTe", "semiconductor"),

    # === CERAMICS & REFRACTORIES (20) ===
    ("mp-2133", "Si3N4 (Silicon Nitride)", "ceramic"),
    ("mp-1029", "SiC (Silicon Carbide)", "ceramic"),
    ("mp-1243", "AlN (Aluminum Nitride)", "ceramic"),
    ("mp-636", "TiN (Titanium Nitride)", "ceramic"),
    ("mp-1818", "TiC (Titanium Carbide)", "ceramic"),
    ("mp-11714", "WC (Tungsten Carbide)", "ceramic"),
    ("mp-2074", "BN (Boron Nitride)", "ceramic"),
    ("mp-1700", "B4C (Boron Carbide)", "ceramic"),
    ("mp-1639", "ZrC", "ceramic"),
    ("mp-1857", "ZrN", "ceramic"),
    ("mp-1178", "HfC", "ceramic"),
    ("mp-1688", "HfN", "ceramic"),
    ("mp-1184", "TaC", "ceramic"),
    ("mp-1653", "NbC", "ceramic"),
    ("mp-1673", "VC", "ceramic"),
    ("mp-1672", "VN", "ceramic"),
    ("mp-1007", "Cr3C2", "ceramic"),
    ("mp-570349", "Mo2C", "ceramic"),
    ("mp-1009930", "MoN", "ceramic"),
    ("mp-1181", "TaN", "ceramic"),

    # === COMMON ALLOYS & COMPOUNDS (20) ===
    ("mp-568", "Fe3C (Cementite in Steel)", "metal"),
    ("mp-20194", "Fe7C3", "metal"),
    ("mp-1094", "CuZn (Brass)", "metal"),
    ("mp-1141", "Cu3Sn (Bronze)", "metal"),
    ("mp-2490", "NiAl", "metal"),
    ("mp-1547", "Ni3Al", "metal"),
    ("mp-1183", "TiAl", "metal"),
    ("mp-1548", "Ti3Al", "metal"),
    ("mp-1185", "FeAl", "metal"),
    ("mp-1598", "Fe3Al", "metal"),
    ("mp-1200", "CoAl", "metal"),
    ("mp-1595", "Co3Al", "metal"),
    ("mp-1080", "CuAl2", "metal"),
    ("mp-1228", "Mg2Si", "metal"),
    ("mp-1547", "AlLi", "metal"),
    ("mp-1651", "ZrAl3", "metal"),
    ("mp-1844", "NiTi (Nitinol)", "metal"),
    ("mp-20661", "Ni3Ti", "metal"),
    ("mp-1186", "FeNi (Invar)", "metal"),
    ("mp-2055", "FeNi3 (Permalloy)", "metal"),
]


def download_materials(count: int = 100) -> List[MPMaterialData]:
    """
    Download common materials from Materials Project

    Args:
        count: Number of materials to download

    Returns:
        List of MPMaterialData objects
    """
    # Check API key
    if not os.environ.get("MP_API_KEY"):
        print("\n❌ Error: MP_API_KEY environment variable not set")
        print("\n📝 To get a free API key:")
        print("   1. Go to https://materialsproject.org/api")
        print("   2. Sign up or log in")
        print("   3. Copy your API key")
        print("   4. Set it: export MP_API_KEY='your-key-here'")
        print("\n💡 Add to ~/.bashrc or ~/.zshrc to persist across sessions:")
        print("   echo 'export MP_API_KEY=\"your-key-here\"' >> ~/.bashrc")
        sys.exit(1)

    logging.info("Initializing Materials Project client...")
    client = MaterialsProjectClient()

    materials = []
    failed = []

    print(f"\n🔬 Downloading {count} common materials from Materials Project...")
    print("=" * 80)

    for i, (mp_id, name, category) in enumerate(COMMON_MATERIALS[:count], 1):
        print(f"\n[{i}/{count}] Fetching {name} ({mp_id})...", end=" ")

        try:
            material = client.get_material(mp_id, use_cache=True)

            if material:
                materials.append(material)
                print(f"✓ {material.formula}")
                print(f"        ρ={material.density:.2f} g/cm³, Eg={material.band_gap:.2f} eV, "
                      f"ΔH={material.formation_energy_per_atom:.3f} eV/atom")
            else:
                failed.append((mp_id, name, "Not found"))
                print("✗ Not found")

        except Exception as e:
            failed.append((mp_id, name, str(e)))
            print(f"✗ Error: {e}")

    return materials, failed


def save_dataset(materials: List[MPMaterialData], output_dir: str = "./data"):
    """Save materials dataset to JSON"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save full dataset
    dataset_file = output_path / "materials_project_100_common.json"

    dataset = {
        "source": "Materials Project",
        "license": "CC-BY-4.0",
        "count": len(materials),
        "materials": [
            {
                "mp_id": mat.mp_id,
                "formula": mat.formula,
                "formation_energy_per_atom": mat.formation_energy_per_atom,
                "band_gap": mat.band_gap,
                "density": mat.density,
                "volume": mat.volume,
                "nsites": mat.nsites,
                "space_group": mat.space_group,
                "is_stable": mat.is_stable,
                "theoretical": mat.theoretical,
                "structure": mat.structure,
            }
            for mat in materials
        ]
    }

    with open(dataset_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✓ Dataset saved to: {dataset_file}")

    # Save summary
    summary_file = output_path / "materials_project_summary.txt"

    with open(summary_file, 'w') as f:
        f.write("MATERIALS PROJECT - 100 COMMON MATERIALS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total materials: {len(materials)}\n")
        f.write(f"Source: Materials Project (https://materialsproject.org)\n")
        f.write(f"License: CC-BY-4.0\n\n")

        # Statistics
        stable_count = sum(1 for m in materials if m.is_stable)
        theoretical_count = sum(1 for m in materials if m.theoretical)
        metals = sum(1 for m in materials if m.band_gap < 0.1)
        semiconductors = sum(1 for m in materials if 0.1 <= m.band_gap <= 3.0)
        insulators = sum(1 for m in materials if m.band_gap > 3.0)

        f.write(f"Statistics:\n")
        f.write(f"  - Stable materials: {stable_count}\n")
        f.write(f"  - Theoretical (no experimental data): {theoretical_count}\n")
        f.write(f"  - Metals (Eg < 0.1 eV): {metals}\n")
        f.write(f"  - Semiconductors (0.1 < Eg < 3.0 eV): {semiconductors}\n")
        f.write(f"  - Insulators (Eg > 3.0 eV): {insulators}\n\n")

        f.write("MATERIALS LIST:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'#':<4} {'MP-ID':<15} {'Formula':<20} {'Density':<12} {'Band Gap':<12} {'Stable':<8}\n")
        f.write("-" * 80 + "\n")

        for i, mat in enumerate(materials, 1):
            stable_str = "Yes" if mat.is_stable else "No"
            f.write(f"{i:<4} {mat.mp_id:<15} {mat.formula:<20} {mat.density:>8.2f} g/cm³ "
                    f"{mat.band_gap:>8.2f} eV   {stable_str:<8}\n")

    print(f"✓ Summary saved to: {summary_file}")

    return dataset_file, summary_file


def main():
    """Main execution"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n" + "=" * 80)
    print("MATERIALS PROJECT - COMMON MATERIALS DOWNLOAD")
    print("=" * 80)

    # Download materials
    materials, failed = download_materials(count=100)

    print("\n" + "=" * 80)
    print(f"✓ Downloaded {len(materials)} materials")

    if failed:
        print(f"⚠  Failed: {len(failed)} materials")
        print("\nFailed materials:")
        for mp_id, name, error in failed:
            print(f"  - {name} ({mp_id}): {error}")

    # Save dataset
    if materials:
        dataset_file, summary_file = save_dataset(materials)

        print("\n" + "=" * 80)
        print("DATASET READY!")
        print("=" * 80)
        print(f"\n📁 Files created:")
        print(f"   - {dataset_file}")
        print(f"   - {summary_file}")

        print(f"\n📊 Statistics:")
        print(f"   - Total materials: {len(materials)}")
        print(f"   - Stable: {sum(1 for m in materials if m.is_stable)}")
        print(f"   - Metals: {sum(1 for m in materials if m.band_gap < 0.1)}")
        print(f"   - Semiconductors: {sum(1 for m in materials if 0.1 <= m.band_gap <= 3.0)}")
        print(f"   - Insulators: {sum(1 for m in materials if m.band_gap > 3.0)}")

        print(f"\n✅ Ready for validation and testing!")

    else:
        print("\n❌ No materials downloaded. Check your API key and connection.")
        sys.exit(1)


if __name__ == "__main__":
    main()
