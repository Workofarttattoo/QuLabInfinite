from __future__ import annotations
import sqlite3
from typing import Iterable
from pydantic import BaseModel
import json
from ingest.pipeline import IngestionPipeline, PydanticValidator, DataValidator
from ingest.schemas import RecordChem
from ingest.sources import nist_thermo

def setup_database(db_path: str):
    """Set up the SQLite database and create tables."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id TEXT PRIMARY KEY,
            substance TEXT,
            phase TEXT,
            temperature_k REAL,
            pressure_pa REAL,
            enthalpy_j_per_mol REAL,
            entropy_j_per_mol_k REAL,
            provenance TEXT,
            tags TEXT
        )
        """)
        conn.commit()

def load_to_db(records: Iterable[BaseModel], db_path: str):
    """Load records into the SQLite database."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for record in records:
            if isinstance(record, RecordChem):
                cursor.execute("""
                INSERT OR REPLACE INTO records (id, substance, phase, temperature_k, pressure_pa, enthalpy_j_per_mol, entropy_j_per_mol_k, provenance, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(hash(record.model_dump_json())),
                    record.substance,
                    record.phase,
                    record.temperature_k,
                    record.pressure_pa,
                    record.enthalpy_j_per_mol,
                    record.entropy_j_per_mol_k,
                    json.dumps(record.provenance.model_dump()),
                    ",".join(record.tags or [])
                ))
        conn.commit()
    print(f"Loaded records into {db_path}")

if __name__ == "__main__":
    DB_PATH = "data/qulab.db"
    
    # Set up the database
    setup_database(DB_PATH)

    # Define the ingestion pipeline
    pipeline = IngestionPipeline(
        processors=[
            PydanticValidator(schema=RecordChem)
        ],
        post_processors=[
            DataValidator(registry_path="data/registry.jsonl")
        ]
    )

    # Run the pipeline with a data source
    records = nist_thermo.load_live()
    processed_records = list(pipeline.process_records(records))

    # Load the processed data into the database
    load_to_db(processed_records, DB_PATH)

