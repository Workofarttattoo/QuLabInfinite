#!/usr/bin/env python3
"""
Utility to expand the materials database from JSON files.
"""

import json
import os
import sys
from typing import Optional

# Adjust path to allow imports if run as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from materials_lab.materials_database import MaterialsDatabase, MaterialProperties

def bulk_import_from_json(file_path: str, db: Optional[MaterialsDatabase] = None) -> MaterialsDatabase:
    """
    Import materials from a JSON file into the database.

    The JSON file should be a dictionary where keys are material names
    and values are property dictionaries matching MaterialProperties fields.
    """
    if db is None:
        db = MaterialsDatabase()

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return db

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        count = 0
        for name, props in data.items():
            if name.startswith("_"): continue

            try:
                # Ensure name is in properties
                if "name" not in props:
                    props["name"] = name

                # Filter out unknown fields to avoid TypeError
                valid_fields = MaterialProperties.__dataclass_fields__.keys()
                filtered_props = {k: v for k, v in props.items() if k in valid_fields}

                mat = MaterialProperties(**filtered_props)
                db.materials[name] = mat
                count += 1
            except Exception as e:
                print(f"Skipping '{name}': {e}")

        print(f"Successfully imported {count} materials from {file_path}")
        db.save()

    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{file_path}'")
    except Exception as e:
        print(f"Error importing: {e}")

    return db

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python expand_materials_db.py <json_file>")
        sys.exit(1)

    bulk_import_from_json(sys.argv[1])
