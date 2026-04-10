import os
import sys
from argparse import Namespace

# Add project root to path to allow absolute imports
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from ingest.pipeline import IngestionPipeline, PydanticValidator
from ingest.plugins import PLUGIN_REGISTRY
from ingest.schemas import RecordMaterial

def ingest_all_materials(api_key: str, output_path: str):
    """
    Programmatically ingests all materials from the Materials Project.

    Args:
        api_key: Your Materials Project API key.
        output_path: The path to the output .jsonl file.
    """
    os.environ["MP_API_KEY"] = api_key

    # Get the Materials Project plugin
    plugin_class = PLUGIN_REGISTRY.get("materials_project")
    if not plugin_class:
        raise RuntimeError("Materials Project plugin not found.")

    # Simulate command-line arguments - no specific material_id to get all
    args = Namespace()

    plugin_instance = plugin_class()
    # Modify the load method call if needed to signify 'all'
    # Assuming the plugin's load method handles fetching all when no ID is given
    # This might require a change in the plugin itself.
    # For now, let's see how it behaves. The plugin will need to be adapted.
    
    # Let's adapt the plugin call first. The plugin expects a material_id.
    # The plugin needs to be modified.
    # It's better to call the mpr search directly here.
    
    from pymatgen.ext.matproj import MPRester

    with MPRester(api_key) as mpr:
        print("Fetching all materials from Materials Project... This will take a long time.")
        all_docs = mpr.search("summary")

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
                except Exception:
                    # Skip records that fail to parse
                    continue

        pipeline = IngestionPipeline(processors=[PydanticValidator(schemas=[RecordMaterial])])
        
        print(f"Ingesting all materials to {output_path}...")
        path = pipeline.run(record_generator(all_docs), output_path)
        print(f"Successfully ingested materials to: {path}")


if __name__ == "__main__":
    API_KEY = "YI56kyKUPqFbyAmZXwgWy6lde3THG45I"
    if not API_KEY:
        raise ValueError("Please provide your Materials Project API key.")

    OUTPUT_PATH = "materials_lab/data/materials_project_all.jsonl"
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    ingest_all_materials(API_KEY, OUTPUT_PATH)
