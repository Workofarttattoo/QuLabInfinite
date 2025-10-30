from __future__ import annotations
from typing import Iterable
import requests
from ..schemas import RecordMaterial, Provenance
from pymatgen.core import Structure, Lattice

BASE_URL = "http://oqmd.org/optimade/structures"

def load_material_by_filter(oqmd_filter: str) -> Iterable[RecordMaterial]:
    """
    Load material data from the OQMD for a given filter.
    See https://static.oqmd.org/static/docs/restful.html for filter syntax.
    """
    
    response = requests.get(BASE_URL, params={"filter": oqmd_filter})
    response.raise_for_status()
    data = response.json()

    if not data or not data.get("data"):
        return

    for entry in data["data"]:
        attributes = entry.get("attributes", {})
        formula = attributes.get("chemical_formula_descriptive", "unknown")
        
        provenance = Provenance(
            source="OQMD (optimade)",
            url=f"http://oqmd.org/materials/entry/{entry['id']}",
            license="ODbL-1.0",
            notes=f"Data for {formula} (OQMD ID: {entry['id']}).",
        )

        # Construct pymatgen structure
        try:
            lattice = Lattice(attributes['lattice_vectors'])
            species = [s['name'] for s in attributes['species']]
            sites = attributes['cartesian_site_positions']
            structure = Structure(lattice, species, sites)
        except (KeyError, TypeError):
            # some entries might be missing structural info
            continue

        record = RecordMaterial(
            substance=formula,
            material_id=str(entry["id"]),
            structure=structure.as_dict(),
            formation_energy_per_atom_ev=attributes.get("formation_energy_per_atom"),
            tags=[
                f"nelements:{attributes.get('nelements')}",
                f"nsites:{attributes.get('nsites')}",
            ],
            provenance=provenance,
        )

        yield record
