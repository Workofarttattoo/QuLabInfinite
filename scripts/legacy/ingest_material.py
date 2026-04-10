import os
import sys
from argparse import Namespace

# Add project root to path to allow absolute imports
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from ingest.pipeline import IngestionPipeline, PydanticValidator
from ingest.plugins import PLUGIN_REGISTRY
from ingest.schemas import RecordMaterial

def ingest_material(api_key: str, material_id: str, output_path: str):
    """
    Programmatically ingests a single material from the Materials Project.

    Args:
        api_key: Your Materials Project API key.
        material_id: The ID of the material to ingest (e.g., "mp-149").
        output_path: The path to the output .jsonl file.
    """
    os.environ["MP_API_KEY"] = api_key

    # Get the Materials Project plugin
    plugin_class = PLUGIN_REGISTRY.get("materials_project")
    if not plugin_class:
        raise RuntimeError("Materials Project plugin not found.")

    # Simulate command-line arguments
    args = Namespace(material_id=material_id)

    plugin_instance = plugin_class()
    records = plugin_instance.load(args)

    # Set up and run the ingestion pipeline
    pipeline = IngestionPipeline(processors=[PydanticValidator(schemas=[RecordMaterial])])
    
    print(f"Ingesting {material_id} from Materials Project...")
    path = pipeline.run(records, output_path)
    print(f"Successfully ingested material to: {path}")

if __name__ == "__main__":
    API_KEY = "YI56kyKUPqFbyAmZXwgWy6lde3THG45I"
    if not API_KEY:
        raise ValueError("Please provide your Materials Project API key.")

    MATERIAL_ID = "mp-149"  # Silicon, a common test case
    OUTPUT_PATH = "materials_lab/data/materials_project_expansion.jsonl"
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    ingest_material(API_KEY, MATERIAL_ID, OUTPUT_PATH)
