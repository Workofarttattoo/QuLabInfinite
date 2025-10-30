from __future__ import annotations
import os
import psycopg2
import json
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

def load_jsonl_to_db(file_path: str):
    """Loads a .jsonl file into the appropriate database table."""
    conn = get_db_connection()
    cur = conn.cursor()

    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Determine if it's a RecordChem or RecordMaterial
            # This is a bit brittle, relies on unique fields
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
            
            # Use content_hash or experiment_id to avoid duplicates
            if hasattr(record, 'content_hash'):
                cur.execute(f"SELECT id FROM {table} WHERE content_hash = %s", (record.content_hash(),))
                if cur.fetchone():
                    continue
            elif hasattr(record, 'experiment_id'):
                cur.execute(f"SELECT id FROM {table} WHERE experiment_id = %s", (record.experiment_id,))
                if cur.fetchone():
                    continue

            # Insert new record
            columns = [f for f in record.model_fields.keys() if f != 'id']
            values = [getattr(record, f) for f in columns]
            
            # Need to handle jsonb and array fields
            for i, col in enumerate(columns):
                if isinstance(values[i], dict) or isinstance(values[i], list) and col != 'tags':
                    values[i] = json.dumps(values[i])

            insert_sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))})"
            
            if hasattr(record, 'content_hash'):
                insert_sql = f"INSERT INTO {table} ({', '.join(columns)}, content_hash) VALUES ({', '.join(['%s'] * len(columns))}, %s)"
                cur.execute(insert_sql, values + [record.content_hash()])
            else:
                cur.execute(insert_sql, values)

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
