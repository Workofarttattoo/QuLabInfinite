from __future__ import annotations
from typing import Iterable
import pandas as pd
from pathlib import Path

from ingest.sources.nist_thermo import NistThermoClient
from ingest.sources.hapi_spectroscopy import HapiSpectroscopyClient
from ingest.sources.materials_project import MaterialsProjectClient
from ingest.sources.oqmd import OQMDClient
from ingest.registry import Registry
from validation.results_validator import ResultsValidator


class IngestionPipeline:
    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.registry = Registry(str(self.data_dir / "registry.jsonl"))
        self.validator = ResultsValidator()

    def _write_and_register(self, name: str, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        if df.empty:
            return df
        out_path = self.data_dir / filename
        df.to_parquet(out_path, index=False)
        self.registry.add_entry(name, str(out_path), len(df))
        # Validate basic constraints and write validated copy
        validated = self.validator.validate_numeric_ranges(df)
        validated_out = self.data_dir / f"validated_{filename}"
        validated.to_parquet(validated_out, index=False)
        self.registry.add_entry(f"validated_{name}", str(validated_out), len(validated))
        return df

    def run_nist(self, identifiers: Iterable[str]) -> pd.DataFrame:
        client = NistThermoClient()
        df = client.fetch(identifiers)
        return self._write_and_register("nist_thermo", df, "nist_thermo.parquet")

    def run_hapi(self, identifiers: Iterable[str]) -> pd.DataFrame:
        client = HapiSpectroscopyClient()
        df = client.fetch(identifiers)
        return self._write_and_register("hapi_spectroscopy", df, "hapi_spectroscopy.parquet")

    def run_materials_project(self, identifiers: Iterable[str]) -> pd.DataFrame:
        client = MaterialsProjectClient()
        df = client.fetch(identifiers)
        return self._write_and_register("materials_project", df, "materials_project.parquet")

    def run_oqmd(self, identifiers: Iterable[str]) -> pd.DataFrame:
        client = OQMDClient()
        df = client.fetch(identifiers)
        return self._write_and_register("oqmd", df, "oqmd.parquet")


if __name__ == "__main__":
    ids = ["H2O", "CO2", "CH4"]
    pipeline = IngestionPipeline()
    nist = pipeline.run_nist(ids)
    hapi = pipeline.run_hapi(ids)
    mp = pipeline.run_materials_project(ids)
    oq = pipeline.run_oqmd(ids)
    print({
        "nist": len(nist),
        "hapi": len(hapi),
        "mp": len(mp),
        "oqmd": len(oq),
    })
