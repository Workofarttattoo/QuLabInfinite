import os
import sys
from argparse import Namespace
import json

# Add project root to path to allow absolute imports
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from ingest.pipeline import IngestionPipeline, PydanticValidator
from ingest.schemas import RecordMaterial
from pymatgen.ext.matproj import MPRester

def ingest_by_chemsys(api_key: str, chemsys: str, output_path: str):
    """
    Programmatically ingests all materials for a given chemical system.

    Args:
        api_key: Your Materials Project API key.
        chemsys: The chemical system to search for (e.g., "Fe-Cr-Ni-Mo").
        output_path: The path to the output .jsonl file.
    """
    os.environ["MP_API_KEY"] = api_key

    with MPRester(api_key) as mpr:
        print(f"Fetching all materials for {chemsys} from Materials Project... This may take some time.")
        
        # Search for all materials in the specified chemical system
        docs = mpr.search("summary", chemsys=chemsys)

        def record_generator(docs):
            for doc in docs:
                try:
                    provenance = {
                        "source": "Materials Project",
                        "url": f"https://materialsproject.org/materials/{doc['material_id']}",
                        "license": "CC-BY-4.0",
                        "notes": f"Data for {doc['formula_pretty']} ({doc['material_id']}).",
                    }
                    
                    record = RecordMaterial(
                        substance=doc['formula_pretty'],
                        material_id=doc['material_id'],
                        structure=doc['structure'].as_dict(),
                        formation_energy_per_atom_ev=doc['formation_energy_per_atom'],
                        band_gap_ev=doc['band_gap'],
                        density_g_cm3=doc['density'],
                        volume_a3_per_atom=doc['volume'] / doc['nsites'],
                        tags=[f"space_group:{doc['symmetry']['symbol']}"],
                        provenance=provenance,
                    )
                    yield record
                except Exception as e:
                    print(f"Skipping record due to parsing error: {e}")
                    continue

        pipeline = IngestionPipeline(processors=[PydanticValidator(schemas=[RecordMaterial])])
        
        print(f"Ingesting {len(docs)} materials to {output_path}...")
        path = pipeline.run(record_generator(docs), output_path)
        print(f"Successfully ingested materials to: {path}")

if __name__ == "__main__":
    API_KEY = "YI56kyKUPqFbyAmZXwgWy6lde3THG45I"
    if not API_KEY:
        raise ValueError("Please provide your Materials Project API key.")

    # --- Define the chemical system to ingest ---
    CHEM_SYS = "Fe-Cr-Ni-Mo"
    # -----------------------------------------

    OUTPUT_PATH = f"downloads/{CHEM_SYS}.jsonl"
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    ingest_by_chemsys(API_KEY, CHEM_SYS, OUTPUT_PATH)
