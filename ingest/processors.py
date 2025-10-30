from __future__ import annotations
from typing import Dict, Any, Callable
from pydantic import BaseModel
from validation.results_validator import ResultsValidator, ValidationStatus
from .schemas import RecordChem, RecordMaterial

EV_TO_J = 1.60218e-19
G_CM3_TO_KG_M3 = 1000
A3_TO_M3 = 1e-30

class UnitConversionProcessor:
    """A processor that converts units to SI for RecordMaterial."""

    def __call__(self, record: BaseModel) -> BaseModel:
        if isinstance(record, RecordMaterial):
            if record.formation_energy_per_atom_ev is not None:
                record.formation_energy_per_atom_j = record.formation_energy_per_atom_ev * EV_TO_J
            
            if record.band_gap_ev is not None:
                record.band_gap_j = record.band_gap_ev * EV_TO_J

            if record.density_g_cm3 is not None:
                record.density_kg_m3 = record.density_g_cm3 * G_CM3_TO_KG_M3
            
            if record.volume_a3_per_atom is not None:
                record.volume_m3_per_atom = record.volume_a3_per_atom * A3_TO_M3
        
        return record

VALIDATION_RULES = [
    {
        "schema": RecordChem,
        "criteria": {"substance": "H2O", "tags": "enthalpy_of_formation"},
        "key": "h2o_enthalpy_of_formation",
        "value_field": "enthalpy_j_per_mol"
    },
    {
        "schema": RecordMaterial,
        "criteria": {"material_id": "mp-149"}, # Silicon
        "key": "silicon_band_gap",
        "value_field": "band_gap_ev"
    },
    {
        "schema": RecordMaterial,
        "criteria": {"material_id": "mp-22862"}, # NaCl
        "key": "nacl_formation_energy",
        "value_field": "formation_energy_per_atom_ev"
    },
]

class ValidationProcessor:
    """A processor that validates records against a reference database."""
    def __init__(self, registry_path: str = "data/registry.jsonl"):
        self.validator = ResultsValidator()
        self.registry_path = registry_path
        print(f"Loaded {len(self.validator.reference_db)} reference data points for validation.")

    def __call__(self, record: BaseModel) -> BaseModel | None:
        """
        Validates a record and enriches it with validation info.
        """
        for rule in VALIDATION_RULES:
            if isinstance(record, rule['schema']):
                match = all(
                    (
                        getattr(record, k) == v if k != 'tags' else
                        v in getattr(record, k)
                    )
                    for k, v in rule['criteria'].items()
                )
                if match:
                    reference_key = rule['key']
                    simulated_value = getattr(record, rule['value_field'])

                    if simulated_value is not None:
                        result = self.validator.validate(simulated_value, reference_key)
                        self.write_to_registry({
                            "record_hash": record.content_hash(),
                            "reference_key": reference_key,
                            "validation_status": result.status.value,
                            "error_percent": result.error_percent,
                            "z_score": result.z_score,
                        })
        
        return record

    def get_reference_key(self, record: BaseModel) -> str | None:
        """
        Generate a reference key from a record.
        """
        if isinstance(record, RecordChem):
            if "enthalpy_of_formation" in record.tags and record.substance == "H2O":
                return "h2o_enthalpy_of_formation"
        elif isinstance(record, RecordMaterial):
            if record.material_id == "mp-149": # Silicon
                return "silicon_band_gap"
        return None

    def get_record_value(self, record: BaseModel, key: str) -> float | None:
        """
        Extract a value from a record that corresponds to a reference key.
        """
        if isinstance(record, RecordChem):
            if key == "h2o_enthalpy_of_formation":
                return record.enthalpy_j_per_mol
        elif isinstance(record, RecordMaterial):
            if key == "silicon_band_gap":
                return record.band_gap_ev
        return None

    def write_to_registry(self, entry: Dict[str, Any]):
        """Append a validation entry to the registry file."""
        import json
        from pathlib import Path
        
        registry_path = Path(self.registry_path)
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(registry_path, 'a') as f:
            f.write(json.dumps(entry) + "\n")

class UnitConverter:
    """A processor to convert units of specified fields."""

    def __init__(self, conversions: Dict[str, Callable[[Any], Any]]):
        """
        Initialize the unit converter.
        Args:
            conversions: A dictionary mapping field names to conversion functions.
        """
        self.conversions = conversions

    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any] | None:
        """
        Apply unit conversions to a record.
        """
        if record is None:
            return None
        
        for field, conversion_func in self.conversions.items():
            if field in record and record[field] is not None:
                try:
                    record[field] = conversion_func(record[field])
                except (ValueError, TypeError) as e:
                    print(f"Could not convert field '{field}' with value {record[field]}: {e}")
                    # Decide on error handling: return None, raise exception, or just log.
                    # For now, we'll log and continue.
                    pass
        return record
