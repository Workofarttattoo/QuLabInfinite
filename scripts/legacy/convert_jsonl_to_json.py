import json
import os

def convert_jsonl_to_json(jsonl_path: str, json_path: str):
    """
    Converts a .jsonl file from the ingestion pipeline to a 
    standard .json file compatible with MaterialsDatabase.

    Args:
        jsonl_path: Path to the input .jsonl file.
        json_path: Path to the output .json file.
    """
    materials_dict = {}
    with open(jsonl_path, 'r') as f_in:
        for line in f_in:
            if line.strip():
                record = json.loads(line)
                material_name = record.get('substance')
                if material_name:
                    record['name'] = record.pop('substance')
                    if 'material_id' in record:
                        del record['material_id']
                    record['category'] = "element"
                    record['subcategory'] = "semiconductor"
                    materials_dict[material_name] = record
    
    with open(json_path, 'w') as f_out:
        json.dump(materials_dict, f_out, indent=2)

    print(f"Successfully converted {jsonl_path} to {json_path}")

if __name__ == "__main__":
    JSONL_PATH = "materials_lab/data/materials_project_expansion.jsonl"
    JSON_PATH = "materials_lab/data/materials_project_expansion.json"

    if not os.path.exists(JSONL_PATH):
        raise FileNotFoundError(f"Input file not found: {JSONL_PATH}")

    convert_jsonl_to_json(JSONL_PATH, JSON_PATH)
