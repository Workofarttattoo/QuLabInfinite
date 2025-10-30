import argparse
import json
from results_validator import ResultsValidator, ValidationStatus
from pathlib import Path
import sys

# Add the project root to the python path to allow for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from ingest.schemas import RecordChem

def main():
    parser = argparse.ArgumentParser(description="Run validation on an ingested dataset.")
    parser.add_argument("dataset_path", type=str, help="Path to the ingested dataset file (.jsonl).")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    validator = ResultsValidator()
    print(f"Loaded {len(validator.reference_db)} reference data points.")

    validation_results = {}

    with open(dataset_path, 'r') as f:
        for line in f:
            record_data = json.loads(line)
            record = RecordChem.model_validate(record_data)
            
            # This is a placeholder for a more sophisticated mapping
            # from a record to a reference key.
            if "enthalpy_of_formation" in record.tags and record.substance == "H2O":
                reference_key = "water_heat_capacity" # Incorrect, but demonstrates the concept
                simulated_value = record.enthalpy_j_per_mol / 4184 # Convert J/mol to J/g
                
                result = validator.validate(simulated_value, reference_key)
                validation_results[f"{record.experiment_id}_{reference_key}"] = result

    print(validator.generate_report(validation_results))

if __name__ == "__main__":
    main()
