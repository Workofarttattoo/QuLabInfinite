import os
import sys
from argparse import Namespace
import json

# Add project root to path to allow absolute imports
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from ingest.pipeline import IngestionPipeline, PydanticValidator
from ingest.plugins import PLUGIN_REGISTRY
from ingest.schemas import RecordMaterial  # This might need to change for synthesis data

def ingest_synthesis_data(api_key: str, output_path: str):
    """
    Programmatically ingests all synthesis data from the Materials Project.

    Args:
        api_key: Your Materials Project API key.
        output_path: The path to the output .jsonl file.
    """
    os.environ["MP_API_KEY"] = api_key
    
    from pymatgen.ext.matproj import MPRester

    with MPRester(api_key) as mpr:
        print("Fetching all synthesis data from Materials Project... This may take some time.")
        
        # The synthesis endpoint might return a different structure.
        # We will dump it directly to a file to inspect.
        all_synthesis_docs = mpr.synthesis.search()

        print(f"Ingesting all synthesis data to {output_path}...")
        with open(output_path, 'w') as f:
            for doc in all_synthesis_docs:
                # The doc from search might not be a simple dict,
                # we need to convert it to a dict to be JSON serializable.
                # Assuming it has a .dict() or similar method, like pydantic models.
                if hasattr(doc, 'dict'):
                    f.write(doc.json() + '\n')
                elif hasattr(doc, '__dict__'):
                     f.write(json.dumps(doc.__dict__) + '\n')
                else:
                    # Fallback for unknown object types
                    try:
                        f.write(json.dumps(str(doc)) + '\n')
                    except TypeError:
                        print(f"Could not serialize document of type {type(doc)}")
                        continue

        print(f"Successfully ingested synthesis data to: {output_path}")


if __name__ == "__main__":
    API_KEY = "YI56kyKUPqFbyAmZXwgWy6lde3THG45I"
    if not API_KEY:
        raise ValueError("Please provide your Materials Project API key.")

    OUTPUT_PATH = "downloads/synthesis.jsonl"
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    ingest_synthesis_data(API_KEY, OUTPUT_PATH)
