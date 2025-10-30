import argparse
import json
from results_validator import ResultsValidator, ValidationStatus
from pathlib import Path
import sys

# Add the project root to the python path to allow for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from ingest.schemas import RecordChem, RecordMaterial, BaseModel
from ingest.pipeline import IngestionPipeline, PydanticValidator


def get_reference_key(record: BaseModel) -> str | None:
    """
    Generate a reference key from a record.
    This is a simple implementation. A more robust solution might use a mapping file.
    """
    if isinstance(record, RecordChem):
        if "enthalpy_of_formation" in record.tags and record.substance == "H2O":
            # This key is not in reference_data.json, but shows the idea
            return "h2o_enthalpy_of_formation"
        # Add more mappings here for chemical records
    elif isinstance(record, RecordMaterial):
        if record.substance == "Copper" and record.material_id == "mp-30":
             return "copper_thermal_conductivity" # This is an example, the property doesn't match
        if record.material_id:
            # A real implementation would look up properties based on material_id
            # For now, let's assume a direct mapping for a known material for demonstration
            if record.material_id == "mp-13": # Silicon
                # This corresponds to the band gap of silicon
                return "silicon_band_gap"
    return None

def get_record_value(record: BaseModel, key: str) -> float | None:
    """
    Extract a value from a record that corresponds to a reference key.
    """
    if isinstance(record, RecordChem):
        if key == "h2o_enthalpy_of_formation":
            return record.enthalpy_j_per_mol
        # Add more value extractions here
    elif isinstance(record, RecordMaterial):
        if key == "silicon_band_gap":
            return record.band_gap_ev
        # Add more value extractions here
    return None


def main():
    parser = argparse.ArgumentParser(description="Run validation on an ingested dataset.")
    parser.add_argument("dataset_path", type=str, help="Path to the ingested dataset file (.jsonl).")
    parser.add_argument("--registry-path", type=str, default="data/registry.jsonl", help="Path to the data registry.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    validator = ResultsValidator()
    print(f"Loaded {len(validator.reference_db)} reference data points.")

    validation_results = {}
    registry_entries = []

    with open(dataset_path, 'r') as f:
        for line in f:
            record_data = json.loads(line)
            
            # Determine schema
            if 'material_id' in record_data:
                record = RecordMaterial.model_validate(record_data)
            else:
                record = RecordChem.model_validate(record_data)
            
            reference_key = get_reference_key(record)
            if not reference_key:
                continue
            
            simulated_value = get_record_value(record, reference_key)
            if simulated_value is None:
                continue

            result = validator.validate(simulated_value, reference_key)
            uid = getattr(record, 'experiment_id', None) or getattr(record, 'material_id', 'unknown')
            validation_results[f"{uid}_{reference_key}"] = result

            # Create a registry entry
            registry_entry = {
                "record_hash": record.content_hash(),
                "reference_key": reference_key,
                "validation_status": result.status.value,
                "error_percent": result.error_percent,
                "z_score": result.z_score,
            }
            registry_entries.append(registry_entry)

    print(validator.generate_report(validation_results))

    # Write to the registry
    registry_path = Path(args.registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, 'w') as f:
        for entry in registry_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(registry_entries)} entries to {registry_path}")

if __name__ == "__main__":
    main()
