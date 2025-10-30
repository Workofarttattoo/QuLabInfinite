from __future__ import annotations
from typing import Iterable
from ..schemas import RecordChem, Provenance

def load_local_examples() -> Iterable[RecordChem]:
    prov = Provenance(source="Cantera (local examples)", license="BSD-3", notes="Load ignition-delay CSVs or mech YAML")
    # Minimal demo rows
    samples = [
        dict(substance="H2/air", phase="gas", pressure_pa=101325, temperature_k=1100.0, tags=["ignition-delay:short"]),
        dict(substance="CH4/air", phase="gas", pressure_pa=101325, temperature_k=1300.0, tags=["ignition-delay:long"]),
    ]
    for s in samples:
        yield RecordChem(**s, provenance=prov)
