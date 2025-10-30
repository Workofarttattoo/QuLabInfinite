from __future__ import annotations
from typing import Dict, Any, Optional
from pydantic import BaseModel
from validation.results_validator import ResultsValidator, ValidationStatus
import json

class DataValidator:
    """A post-processor to validate data records."""

    def __init__(self, registry_path: str = "data/registry.jsonl"):
        self.validator = ResultsValidator()
        self.registry_path = registry_path

    def __call__(self, record: BaseModel) -> Optional[BaseModel]:
        """
        Validate a record and add validation status.
        """
        if record is None:
            return None

        # This is a placeholder for how you might select a reference key.
        # In a real scenario, this would be more sophisticated.
        reference_key = f"{record.substance}_{record.tags[0]}" if hasattr(record, 'substance') and hasattr(record, 'tags') and record.tags else None

        if reference_key:
            # Assuming the main value to validate is named 'value' in the record for simplicity.
            # This would need to be adapted to the actual schema.
            value_to_validate = None
            if hasattr(record, 'enthalpy_j_per_mol'):
                value_to_validate = record.enthalpy_j_per_mol
            elif hasattr(record, 'entropy_j_per_mol_k'):
                value_to_validate = record.entropy_j_per_mol_k
            
            if value_to_validate is not None:
                validation_result = self.validator.validate(value_to_validate, reference_key)
                if not hasattr(record, 'provenance'):
                    from .schemas import Provenance
                    record.provenance = Provenance(source="Unknown")
                
                if not hasattr(record.provenance, 'extra'):
                    record.provenance.extra = {}
                
                record.provenance.extra['validation'] = validation_result.model_dump()

                # Log to registry
                self._log_to_registry(record, validation_result)

        return record

    def _log_to_registry(self, record: BaseModel, result: ValidationStatus):
        """Log validation result to the registry."""
        entry = {
            "record_id": str(hash(record.model_dump_json())),
            "validation_status": result.status.value,
            "details": result.message,
        }
        with open(self.registry_path, 'a') as f:
            f.write(json.dumps(entry) + "\n")
