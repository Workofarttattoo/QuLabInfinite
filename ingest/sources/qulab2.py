from __future__ import annotations
from typing import Iterable
import json
from ..schemas import RecordChem, Provenance, RecordMaterial, TeleportationSchema


def load_result(file_path: str) -> RecordChem:
    """
    Load a single QuLab2.0 result file and convert it to a RecordChem object.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    provenance = Provenance(
        source="QuLab2.0",
        url=f"file://{file_path}",
        notes="Imported from QuLab2.0 teleportation experiment results.",
        extra=data.get("metadata", {})
    )

    # Adapt the teleportation data to the RecordChem schema
    # This is a temporary adaptation. A new schema would be better.
    record = RecordChem(
        substance="quantum_teleportation",
        experiment_id=data.get("experiment_id"),
        tags=[
            f"fidelity:{data.get('fidelity')}",
            f"success_probability:{data.get('success_probability')}",
            f"shots:{data.get('shots')}",
            f"execution_time_s:{data.get('execution_time')}",
        ],
        provenance=provenance,
        # These fields are required by the schema but not applicable here
        pressure_pa=0,
        temperature_k=0,
    )

    return record
