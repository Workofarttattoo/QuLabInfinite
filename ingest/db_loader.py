from __future__ import annotations
import os
import psycopg2
import json
from psycopg2.extras import execute_values
from ingest.schemas import RecordChem, RecordMaterial, TeleportationSchema

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=os.environ.get("DB_PORT", "5432"),
        dbname=os.environ.get("DB_NAME", "qulab"),
        user=os.environ.get("DB_USER", "user"),
        password=os.environ.get("DB_PASSWORD", "password")
    )
    return conn

def create_tables():
    """Create the necessary tables in the database if they don't exist."""
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chem_records (
        id SERIAL PRIMARY KEY,
        substance TEXT,
        phase TEXT,
        pressure_pa DOUBLE PRECISION,
        temperature_k DOUBLE PRECISION,
        volume_m3_per_mol DOUBLE PRECISION,
        enthalpy_j_per_mol DOUBLE PRECISION,
        entropy_j_per_mol_k DOUBLE PRECISION,
        composition JSONB,
        experiment_id TEXT,
        tags TEXT[],
        provenance JSONB,
        content_hash TEXT UNIQUE,
        spectrum_hdf5_ref TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS material_records (
        id SERIAL PRIMARY KEY,
        substance TEXT,
        material_id TEXT,
        phase TEXT,
        structure JSONB,
        formation_energy_per_atom_ev DOUBLE PRECISION,
        band_gap_ev DOUBLE PRECISION,
        density_g_cm3 DOUBLE PRECISION,
        volume_a3_per_atom DOUBLE PRECISION,
        formation_energy_per_atom_j DOUBLE PRECISION,
        band_gap_j DOUBLE PRECISION,
        density_kg_m3 DOUBLE PRECISION,
        volume_m3_per_atom DOUBLE PRECISION,
        tags TEXT[],
        provenance JSONB,
        content_hash TEXT UNIQUE
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS teleportation_records (
        id SERIAL PRIMARY KEY,
        experiment_id TEXT UNIQUE,
        timestamp TIMESTAMP WITH TIME ZONE,
        alpha REAL,
        beta REAL,
        fidelity REAL,
        success_probability REAL,
        shots INTEGER,
        execution_time REAL,
        measurement_results JSONB,
        classical_bits INTEGER[],
        metadata JSONB
    );
    """)

    conn.commit()
    cur.close()
    conn.close()

def load_jsonl_to_db(file_path: str, batch_size: int = 100):
    """Loads a .jsonl file into the appropriate database table using batch inserts."""
    conn = get_db_connection()
    cur = conn.cursor()

    batches = {
        "chem_records": [],
        "material_records": [],
        "teleportation_records": []
    }

    # Track columns for each table to ensure consistency in batch
    table_columns = {}

    conflict_columns = {
        "chem_records": "content_hash",
        "material_records": "content_hash",
        "teleportation_records": "experiment_id"
    }

    def flush_batch(table_name):
        if not batches[table_name]:
            return

        cols = table_columns[table_name]
        conflict_col = conflict_columns.get(table_name)

        col_names = ", ".join(cols)
        # Using ON CONFLICT DO NOTHING to handle duplicates efficiently
        sql = f"INSERT INTO {table_name} ({col_names}) VALUES %s ON CONFLICT ({conflict_col}) DO NOTHING"

        execute_values(cur, sql, batches[table_name])
        batches[table_name] = []

    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Determine if it's a RecordChem or RecordMaterial
            if "enthalpy_j_per_mol" in data or "entropy_j_per_mol_k" in data:
                record = RecordChem(**data)
                table = "chem_records"
            elif "material_id" in data:
                record = RecordMaterial(**data)
                table = "material_records"
            else:
                try:
                    record = TeleportationSchema(**data)
                    table = "teleportation_records"
                except Exception:
                    print(f"Skipping unknown record type: {data}")
                    continue

            # Prepare columns and values
            columns = [f for f in record.model_fields.keys() if f != 'id']
            values = [getattr(record, f) for f in columns]
            
            if hasattr(record, 'content_hash'):
                if 'content_hash' not in columns:
                    columns.append('content_hash')
                    values.append(record.content_hash())

            if table not in table_columns:
                table_columns[table] = columns

            # Need to handle jsonb and array fields
            for i, col in enumerate(columns):
                if isinstance(values[i], dict) or isinstance(values[i], list) and col != 'tags':
                    values[i] = json.dumps(values[i])

            batches[table].append(tuple(values))
            
            if len(batches[table]) >= batch_size:
                flush_batch(table)

    # Flush remaining records
    for table in batches:
        flush_batch(table)

    conn.commit()
    cur.close()
    conn.close()
    print(f"Finished loading {file_path} into the database.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    
    create_parser = subparsers.add_parser("create_tables")
    
    load_parser = subparsers.add_parser("load_file")
    load_parser.add_argument("file_path", help="Path to the .jsonl file to load.")

    args = parser.parse_args()

    if args.command == "create_tables":
        create_tables()
        print("Tables created successfully (if they didn't exist).")
    elif args.command == "load_file":
        load_jsonl_to_db(args.file_path)
