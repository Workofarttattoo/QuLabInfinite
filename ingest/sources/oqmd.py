from __future__ import annotations
from typing import Iterable, cast
import requests
from ..schemas import RecordChem, Provenance, RecordMaterial

BASE_URL = "http://oqmd.org/oqmdapi"

def search_oqmd(query_params: dict) -> Iterable[RecordMaterial]:
    """
    Search the OQMD database with given query parameters.
    Example: `search_oqmd({'element_set': '(Fe,O)', 'stability': '>0'})`
    """
    response = requests.get(f"{BASE_URL}/calculation", params=query_params)
    response.raise_for_status()
    data = response.json()

    if data['meta']['results_returned'] == 0:
        return

    for entry in data['data']:
        yield _parse_oqmd_entry(entry)

def get_material_by_id(oqmd_id: int) -> RecordMaterial:
    """
    Retrieve a specific material from OQMD by its ID.
    """
    response = requests.get(f"{BASE_URL}/formation_energy/{oqmd_id}")
    response.raise_for_status()
    data = response.json()
    
    if not data or not data.get('data'):
        raise ValueError(f"Material with OQMD ID {oqmd_id} not found.")

    return _parse_oqmd_entry(data['data'])


def _parse_oqmd_entry(entry_data: dict) -> RecordMaterial:
    """
    Parses a single entry from an OQMD API response into a RecordMaterial object.
    """
    provenance = Provenance(
        source="OQMD",
        url=f"http://oqmd.org/materials/entry/{entry_data['id']}",
        license="CC-BY-4.0",
        notes=f"OQMD Entry ID: {entry_data['id']}",
    )

    # Some OQMD entries may not have all fields. We'll provide defaults.
    tags = []
    if entry_data.get('spacegroup'):
        tags.append(f"space_group:{entry_data['spacegroup']}")
    if entry_data.get('prototype'):
        tags.append(f"prototype:{entry_data['prototype']}")

    record = RecordMaterial(
        substance=entry_data.get('composition_generic', 'Unknown'),
        material_id=f"oqmd:{entry_data['id']}",
        structure=None, # OQMD API for calculations does not directly provide structure.
                      # Fetching structure requires another endpoint or parsing from POSCAR if available.
        formation_energy_per_atom_ev=entry_data.get('delta_e'),
        band_gap_ev=entry_data.get('band_gap'),
        volume_a3_per_atom=entry_data.get('volume_pa'),
        tags=tags,
        provenance=provenance,
    )
    return record
