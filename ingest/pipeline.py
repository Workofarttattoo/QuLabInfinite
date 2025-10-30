from __future__ import annotations
from .schemas import RecordChem, Provenance
from typing import Iterable, Dict, Any, Callable, Type
import pathlib, json, csv, hashlib
from pydantic import ValidationError, BaseModel

class IngestionPipeline:
    """
    A pipeline for ingesting and processing data records.
    """
    def __init__(self, schema: Type[BaseModel], validator: Callable = None, transformer: Callable = None):
        self.schema = schema
        self.validator = validator or self.default_validator
        self.transformer = transformer or self.default_transformer

    def run(self, records: Iterable[Dict], out_path: str) -> str:
        """
        Run the ingestion pipeline.
        """
        processed_records = (self.transformer(rec) for rec in records if self.validator(rec))
        
        if out_path.endswith(".jsonl"):
            return write_ndjson(processed_records, out_path)
        elif out_path.endswith(".csv"):
            return write_csv(processed_records, out_path)
        else:
            raise ValueError(f"Unsupported output format: {out_path}")

    def default_validator(self, record: Dict) -> bool:
        """
        Default validator: checks if a record conforms to the RecordChem schema.
        """
        try:
            self.schema.model_validate(record)
            return True
        except ValidationError as e:
            print(f"Validation error for record {record.get('experiment_id', '')}: {e}")
            return False

    @staticmethod
    def default_transformer(record: BaseModel) -> BaseModel:
        """
        Default transformer: returns the record unchanged.
        """
        # In a real pipeline, this could perform unit conversions, data cleaning, etc.
        return record

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
