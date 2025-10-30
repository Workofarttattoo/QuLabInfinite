from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field

from ingest.sources.nist_thermo import NISTThermoClient
from ingest.sources.hapi_spectroscopy import HitranClient
from ingest.sources.materials_project import MaterialsProjectClient
from ingest.sources.oqmd import OQMDClient
from validation.results_validator import validate_record


class CanonicalRecord(BaseModel):
    source: str
    key: str
    data: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class PipelineConfig:
    output_dir: str = "data"


class IngestionPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.registry_path = os.path.join(self.config.output_dir, "registry.jsonl")

        # Lazy init clients; callers can replace/inject as needed
        self.nist = NISTThermoClient()
        self.hitran = HitranClient()
        self.mp: Optional[MaterialsProjectClient] = None
        self.oqmd = OQMDClient()

        # Materials Project requires API key
        try:
            self.mp = MaterialsProjectClient()
        except Exception:
            self.mp = None

    def write_registry(self, records: List[CanonicalRecord]) -> None:
        with open(self.registry_path, "w", encoding="utf-8") as f:
            for rec in records:
                # Validate before write
                validate_record(rec.model_dump())
                f.write(json.dumps(rec.model_dump()) + "\n")

    def run(self) -> str:
        """
        Demonstration run assembling a few records from different sources.
        """
        records: List[CanonicalRecord] = []

        # NIST example (HTML payload placeholder)
        try:
            nist_payload = self.nist.get_thermo_by_formula("H2O")
            records.append(
                CanonicalRecord(source="NIST", key="H2O", data={"raw": bool(nist_payload.get("raw_payload"))})
            )
        except Exception as e:
            records.append(CanonicalRecord(source="NIST", key="H2O", data={"error": str(e)}))

        # HITRAN example (no-op compute placeholder)
        try:
            coeff = self.hitran.compute_absorption("hitran_2_1000_1100", pressure_atm=1.0, temperature_k=296.0)
            records.append(CanonicalRecord(source="HITRAN", key="CO2_band", data={"summary": coeff}))
        except Exception as e:
            records.append(CanonicalRecord(source="HITRAN", key="CO2_band", data={"error": str(e)}))

        # Materials Project (only if API key present)
        if self.mp is not None:
            try:
                mp_resp = self.mp.search_structures("Si")
                records.append(CanonicalRecord(source="MP", key="Si", data={"count": len(mp_resp.get("data", []))}))
            except Exception as e:
                records.append(CanonicalRecord(source="MP", key="Si", data={"error": str(e)}))
        else:
            records.append(CanonicalRecord(source="MP", key="Si", data={"error": "MP_API_KEY not set"}))

        # OQMD example (may require reachable endpoint)
        try:
            oqmd_resp = self.oqmd.search('select * from materials where formula="Si" limit 1')
            records.append(CanonicalRecord(source="OQMD", key="Si", data={"ok": bool(oqmd_resp)}))
        except Exception as e:
            records.append(CanonicalRecord(source="OQMD", key="Si", data={"error": str(e)}))

        self.write_registry(records)
        return self.registry_path


if __name__ == "__main__":
    path = IngestionPipeline().run()
    print(f"Wrote registry to {path}")
