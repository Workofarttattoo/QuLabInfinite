from __future__ import annotations
from typing import Iterable
import os
from ..schemas import RecordMaterial, Provenance
from pymatgen.ext.matproj import MPRester
import argparse
from .base import DataSource

class MaterialsProjectSource(DataSource):
    name = "materials_project"
    description = "Loads material data from the Materials Project."

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument("--material-id", required=True, help="Materials Project ID (e.g., mp-149 for silicon)")

    def load(self, args: argparse.Namespace) -> Iterable[RecordMaterial]:
        material_id = args.material_id
        api_key = os.environ.get("MP_API_KEY")
        if not api_key:
            raise ValueError("MP_API_KEY environment variable not set. Get a key from materialsproject.org.")

        with MPRester(api_key) as mpr:
            doc = mpr.materials.get_data_by_id(material_id)

            if not doc:
                raise ValueError(f"Material {material_id} not found in the Materials Project.")

            provenance = Provenance(
                source="Materials Project",
                url=f"https://materialsproject.org/materials/{material_id}",
                license="CC-BY-4.0",
                notes=f"Data for {doc.formula_pretty} ({material_id}).",
            )

            record = RecordMaterial(
                substance=doc.formula_pretty,
                material_id=material_id,
                structure=doc.structure.as_dict(),
                formation_energy_per_atom_ev=doc.formation_energy_per_atom,
                band_gap_ev=doc.band_gap,
                density_g_cm3=doc.density,
                volume_a3_per_atom=doc.volume / doc.nsites,
                tags=[
                    f"space_group:{doc.spacegroup.symbol}",
                ],
                provenance=provenance,
            )
            yield record
