from __future__ import annotations
from .schemas import RecordChem, Provenance
from typing import Iterable, Dict, Any, Callable, Type, List
import pathlib, json, csv, hashlib
from pydantic import ValidationError, BaseModel
from .processors import UnitConversionProcessor, ValidationProcessor
from .schemas import RecordChem, RecordMaterial

class PydanticValidator:
    """A processor that validates records against a Pydantic schema."""
    def __init__(self, schemas: List[Type[BaseModel]]):
        self.schemas = schemas

    def __call__(self, record: Dict) -> Dict | None:
        if isinstance(record, BaseModel):
            record_dict = record.model_dump()
        else:
            record_dict = record

        for schema in self.schemas:
            try:
                return schema.model_validate(record_dict)
            except ValidationError:
                continue
        
        # If no schema matched
        print(f"Validation error for record: no matching schema found for {record_dict}")
        return None

class IngestionPipeline:
    """
    A pipeline for ingesting and processing data records.
    """
    def __init__(self, processors: List[Callable], post_processors: List[Callable] | None = None):
        self.processors = processors
        self.post_processors = post_processors or []

    def run(self, records: Iterable[Dict], out_path: str) -> str:
        """
        Run the ingestion pipeline.
        """
        processed_records = self.process_records(records)
        
        # Apply post-processors
        for post_processor in self.post_processors:
            processed_records = (post_processor(rec) for rec in processed_records)
            
        final_records = [rec for rec in processed_records if rec is not None]

        if out_path.endswith(".jsonl"):
            return write_ndjson(final_records, out_path)
        elif out_path.endswith(".csv"):
            return write_csv(final_records, out_path)
        else:
            raise ValueError(f"Unsupported output format: {out_path}")

    def process_records(self, records: Iterable[Dict]) -> Iterable[BaseModel]:
        """Process records through all processors."""
        for record in records:
            processed_record = record
            for processor in self.processors:
                if processed_record is None:
                    break
                processed_record = processor(processed_record)
            
            if processed_record is not None:
                yield processed_record

def create_default_pipeline(registry_path: str = "data/registry.jsonl") -> IngestionPipeline:
    """
    Creates an ingestion pipeline with a default set of processors.
    """
    processors = [
        PydanticValidator(schemas=[RecordChem, RecordMaterial]),
        UnitConversionProcessor(),
    ]
    post_processors = [
        ValidationProcessor(registry_path=registry_path),
    ]
    return IngestionPipeline(processors=processors, post_processors=post_processors)

def write_ndjson(records: Iterable[BaseModel], out_path: str) -> str:
    p = pathlib.Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec.model_dump(), default=str) + "\n")
    return str(p)

def write_csv(records: Iterable[BaseModel], out_path: str) -> str:
    p = pathlib.Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    first = True
    with p.open('w', encoding='utf-8', newline='') as f:
        writer = None
        for rec in records:
            row = rec.model_dump()
            # flatten provenance
            if 'provenance' in row and isinstance(row['provenance'], dict):
                prov = row.pop('provenance')
                for k, v in prov.items():
                    row[f'prov_{k}'] = v
            if first:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                first = False
            writer.writerow(row)
    return str(p)

def dataset_fingerprint(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(1024 * 1024)
            if not b: break
            h.update(b)
    return h.hexdigest()
