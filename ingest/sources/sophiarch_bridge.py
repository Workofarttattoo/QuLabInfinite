from __future__ import annotations
from typing import Iterable
from ..schemas import RecordChem, Provenance
import json, pathlib

# Bridge: read Sophiarch forecast JSONs and emit placeholder chem records so
# QuLab can join forecasts with lab conditions.
def load_reports(path: str = "reports/sophiarch") -> Iterable[RecordChem]:
    p = pathlib.Path(path)
    prov = Provenance(source="Bayesian Sophiarch", license="INTERNAL", notes="Converted from oracle briefings")
    if not p.exists():
        return []
    for f in p.glob("*_forecast.json"):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        for outcome in data.get("outcomes", []):
            # Encode horizon probabilities as tags attached to a neutral record
            tags = [f"h:{outcome.get('horizon')}", *(f"{k}:{v}" for k,v in outcome.get('probabilities', {}).items())]
            yield RecordChem(
                substance="FORECAST_JOIN",
                phase=None,
                pressure_pa=0.0,
                temperature_k=0.0,
                volume_m3_per_mol=None,
                enthalpy_j_per_mol=None,
                tags=tags,
                provenance=prov,
            )
